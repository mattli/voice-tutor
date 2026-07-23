"""Claim extraction: decompose a document's text into discrete, assessable claims.

This is the "rubric" that both conversation steering and future scoring will
consume.

Design constraints (kept deliberately minimal so the verifier env stays
satisfiable):
  * Imports ONLY the ``anthropic`` SDK and the standard library. No bot.py,
    app.py, pipecat, or fastapi.
  * No API key is read at import time. The Anthropic client is constructed
    lazily inside :func:`extract_claims`, mirroring documents._generate_summary.
  * Ids are purely positional and deterministic (c1, c2, c3, ...), independent
    of claim text content.

Robustness against real model output (learned from a credentialed smoke run):
  * Structured output. The decomposition uses a FORCED tool call
    (``tool_choice`` pinned to :data:`CLAIMS_TOOL`) so the model returns a
    validated ``{"claims": [...]}`` object as ``tool_use.input`` — a parsed
    dict. Markdown fences and JSON quote-escaping become structurally
    impossible rather than politely requested away.
  * Anchor resolution. The model's job is to POINT at the supporting passage,
    not reproduce it byte-perfectly. Its best-effort anchor is normalized
    (dashes, quote styles, whitespace, case) and located in the source text via
    fuzzy match; the RESOLVED verbatim source substring (plus char offsets) is
    what gets stored, so the persisted artifact stays byte-exact against the
    document — which downstream scoring depends on. An anchor that cannot be
    resolved above :data:`RESOLVE_THRESHOLD` is KEPT and flagged
    ``anchor_unresolved`` rather than discarded.
"""

import difflib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

import anthropic

# Storage root, mirroring documents.DOCUMENTS_DIR (Path.home()/".voice-tutor"/
# "documents"). Defined LOCALLY here — deliberately NOT imported/re-exported from
# documents.py — so this module's import closure stays limited to anthropic +
# stdlib (importing documents.py would drag pypdf into the closure). Referenced at
# call time inside the persistence helpers so tests can redirect it.
DOCUMENTS_DIR = Path.home() / ".voice-tutor" / "documents"

MODEL = "claude-sonnet-5"
MAX_TOKENS = 8_000
MAX_DOC_CHARS_IN = 100_000

# Minimum normalized-similarity for a drifted anchor to count as "resolved" to a
# source span. Below this, the claim is kept but flagged anchor_unresolved.
RESOLVE_THRESHOLD = 0.75

# The prompt encodes the required granularity:
#   * claims a person could articulate in 1-3 spoken sentences,
#   * not one claim per sentence, not chapter-level themes,
#   * roughly 10-40 claims for a typical document.
# The output SHAPE is enforced structurally by the forced tool call, so the
# prompt no longer pleads for bare JSON / no fences — it only shapes content.
CLAIMS_PROMPT = (
    "You are decomposing a document into a rubric of discrete, assessable "
    "claims. A claim is a single, self-contained assertion that a person could "
    "plausibly articulate in 1-3 spoken sentences.\n\n"
    "Granularity rules:\n"
    "- Do NOT emit one claim per sentence of the document — merge closely "
    "related sentences into a single articulable claim.\n"
    "- Do NOT emit broad chapter-level themes — those are too coarse to assess.\n"
    "- Aim for roughly 10 to 40 claims for a typical document; a short document "
    "may have fewer.\n\n"
    "Call the record_claims tool exactly once with the full list. For each "
    "claim provide:\n"
    "  - claim: the claim text, phrased as a standalone assertion.\n"
    "  - anchor: the supporting passage from the document, copied as closely to "
    "verbatim as you can (a contiguous span of the document's text — do not "
    "summarize or stitch together distant sentences).\n\n"
    "Document:\n__DOCUMENT_TEXT__"
)

# Forced-tool schema. tool_choice pins the model to this, so the response
# carries a validated {"claims": [{claim, anchor}, ...]} object as tool_use.input.
CLAIMS_TOOL = {
    "name": "record_claims",
    "description": (
        "Record the discrete, assessable claims decomposed from the document, "
        "in document order."
    ),
    # strict tool use: guarantees tool_use.input validates EXACTLY against this
    # schema, so `claims` is a real array of {claim, anchor} objects. Without it,
    # the model was observed double-encoding the array as a JSON string inside
    # the `claims` field. Requires additionalProperties:false + required on every
    # object.
    "strict": True,
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "claims": {
                "type": "array",
                "description": "The ordered list of claims.",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "claim": {
                            "type": "string",
                            "description": "The claim text, a standalone assertion.",
                        },
                        "anchor": {
                            "type": "string",
                            "description": (
                                "The supporting passage copied from the document."
                            ),
                        },
                    },
                    "required": ["claim", "anchor"],
                },
            }
        },
        "required": ["claims"],
    },
}


class ClaimParseError(Exception):
    """Raised when the model's response cannot be parsed/validated into claims."""


@dataclass(frozen=True)
class Claim:
    """A single assessable claim decomposed from a document.

    Attributes:
        id: Stable, purely positional id (``c1``, ``c2``, ...).
        claim: The claim text, a standalone assertion.
        anchor: The supporting passage. When resolved, this is the VERBATIM
            source substring at ``[anchor_start, anchor_end)`` — byte-exact
            against the document. When unresolved, it is the model's raw
            best-effort anchor (kept for reference, not a source substring).
        anchor_start: Start char offset of the anchor in the source, or None.
        anchor_end: End char offset (exclusive) of the anchor, or None.
        anchor_unresolved: True if the anchor could not be located in the source
            above ``RESOLVE_THRESHOLD``; the claim is kept regardless.
    """

    id: str
    claim: str
    anchor: str
    anchor_start: int | None = None
    anchor_end: int | None = None
    anchor_unresolved: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AnchorResolution:
    """Result of resolving a model anchor against the source document."""

    text: str
    start: int | None
    end: int | None
    unresolved: bool
    score: float


def _claim_id(index: int) -> str:
    """Deterministic, purely positional id for the ``index``-th (0-based) claim."""
    return f"c{index + 1}"


# --------------------------------------------------------------------------- #
# Anchor resolution: normalize model drift, locate the true source span.
# --------------------------------------------------------------------------- #

# Fold common cosmetic drift the model introduces (various dashes -> "-", curly
# quotes -> straight) so a "verbatim" passage the model lightly reformatted still
# matches the source.
_DASHES = "‐‑‒–—―−"  # hyphen/figure/en/em/minus
_SQUOTES = "‘’‚‛′`´"
_DQUOTES = "“”„‟″"


def _canon_char(ch: str) -> str:
    if ch in _DASHES:
        return "-"
    if ch in _SQUOTES:
        return "'"
    if ch in _DQUOTES:
        return '"'
    return ch


def _normalize_with_map(s: str):
    """Return (normalized_text, index_map).

    ``index_map[k]`` is the ORIGINAL index in ``s`` that produced the k-th
    normalized character — so a span located in normalized space can be mapped
    back to a byte-exact span of the original. Normalization: NFKC, canonical
    dashes/quotes, collapsed whitespace, casefold.
    """
    out: list[str] = []
    idx: list[int] = []
    prev_ws = False
    for i, ch in enumerate(s):
        for c in unicodedata.normalize("NFKC", ch):
            c = _canon_char(c)
            if c.isspace():
                if prev_ws:
                    continue
                out.append(" ")
                idx.append(i)
                prev_ws = True
            else:
                for fc in c.casefold():
                    out.append(fc)
                    idx.append(i)
                prev_ws = False
    return "".join(out), idx


def resolve_anchor(anchor: str, document_text: str) -> AnchorResolution:
    """Locate ``anchor`` in ``document_text``, tolerating cosmetic model drift.

    Fast path: exact verbatim substring (score 1.0). Else normalize both sides
    and look for an exact normalized substring (score 1.0 — cosmetic drift only).
    Else fuzzy-match; if similarity >= ``RESOLVE_THRESHOLD`` return the mapped
    verbatim source span, otherwise return the raw anchor flagged unresolved.
    """
    pos = document_text.find(anchor)
    if pos != -1:
        return AnchorResolution(anchor, pos, pos + len(anchor), False, 1.0)

    norm_src, idx = _normalize_with_map(document_text)
    na, _ = _normalize_with_map(anchor)
    na = na.strip()
    if not na:
        return AnchorResolution(anchor, None, None, True, 0.0)

    p = norm_src.find(na)
    if p != -1:
        start, end = idx[p], idx[p + len(na) - 1] + 1
        return AnchorResolution(document_text[start:end], start, end, False, 1.0)

    matcher = difflib.SequenceMatcher(None, norm_src, na, autojunk=False)
    block = matcher.find_longest_match(0, len(norm_src), 0, len(na))
    if block.size == 0:
        return AnchorResolution(anchor, None, None, True, 0.0)
    win_start = max(0, block.a - block.b)
    win_end = min(len(norm_src), win_start + len(na))
    score = difflib.SequenceMatcher(
        None, norm_src[win_start:win_end], na, autojunk=False
    ).ratio()
    if score >= RESOLVE_THRESHOLD:
        start, end = idx[win_start], idx[win_end - 1] + 1
        return AnchorResolution(
            document_text[start:end], start, end, False, round(score, 4)
        )
    return AnchorResolution(anchor, None, None, True, round(score, 4))


def claims_from_records(raw_claims, document_text: str) -> list[Claim]:
    """Validate + resolve a list of ``{"claim", "anchor"}`` records into Claims.

    The order of the returned records mirrors the input order — no sorting,
    reordering, or deduping. Each anchor is resolved against ``document_text``.

    Raises:
        ClaimParseError: if the shape is wrong or any claim text / anchor is
            missing or empty. (A non-locatable anchor is NOT an error — the claim
            is kept and flagged ``anchor_unresolved``.)
    """
    if not isinstance(raw_claims, list):
        raise ClaimParseError('"claims" must be a list')

    out: list[Claim] = []
    for i, item in enumerate(raw_claims):
        if not isinstance(item, dict):
            raise ClaimParseError(f"claim {i} is not an object")
        claim_text = item.get("claim")
        anchor = item.get("anchor")
        if not isinstance(claim_text, str) or not claim_text.strip():
            raise ClaimParseError(f"claim {i} has missing/empty claim text")
        if not isinstance(anchor, str) or not anchor.strip():
            raise ClaimParseError(f"claim {i} has missing/empty anchor")
        res = resolve_anchor(anchor, document_text)
        out.append(
            Claim(
                id=_claim_id(i),
                claim=claim_text.strip(),
                anchor=res.text,
                anchor_start=res.start,
                anchor_end=res.end,
                anchor_unresolved=res.unresolved,
            )
        )
    return out


def _extract_json_object(payload: str) -> dict:
    """Best-effort parse of a free-text model payload into a dict.

    Defensive back-compat helper for the string path (:func:`parse_claims`). The
    primary path (:func:`extract_claims`) uses structured tool output and never
    routes through here. Strips a ```json ...``` fence or grabs the outermost
    ``{...}`` before ``json.loads``.
    """
    text = payload.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    else:
        i, j = text.find("{"), text.rfind("}")
        if i != -1 and j != -1 and j > i:
            text = text[i : j + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ClaimParseError(f"response was not valid JSON: {e}") from e


def parse_claims(payload: str, document_text: str) -> list[Claim]:
    """Parse a free-text model ``payload`` into validated, resolved Claims.

    Back-compat string entry point. Tolerates markdown fences / preamble; routes
    through the same validation + anchor-resolution as the structured path.

    Raises:
        ClaimParseError: if the payload is not valid JSON of the expected shape,
            or any claim text / anchor is missing or empty.
    """
    data = _extract_json_object(payload)
    if not isinstance(data, dict) or "claims" not in data:
        raise ClaimParseError('response JSON missing "claims" key')
    return claims_from_records(data["claims"], document_text)


def _tool_input(response) -> dict:
    """Pull the record_claims tool input (a parsed dict) from an SDK response."""
    for block in getattr(response, "content", []):
        if (
            getattr(block, "type", None) == "tool_use"
            and getattr(block, "name", None) == CLAIMS_TOOL["name"]
        ):
            return block.input
    raise ClaimParseError("model did not return a record_claims tool call")


def extract_claims(document_text: str) -> list[Claim]:
    """Decompose ``document_text`` into an ordered list of :class:`Claim` records.

    Pure function core: text in -> structured claim list out. The Anthropic
    client is constructed lazily here (never at import time) so importing this
    module reads no API key and performs no network I/O. Output shape is enforced
    by a forced tool call (no fences / no JSON-escaping failure modes possible).

    Raises:
        ClaimParseError: if the model returns no tool call or a malformed record
            set (see :func:`claims_from_records`).
    """
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        thinking={"type": "disabled"},
        tools=[CLAIMS_TOOL],
        tool_choice={"type": "tool", "name": CLAIMS_TOOL["name"]},
        messages=[
            {
                "role": "user",
                "content": CLAIMS_PROMPT.replace(
                    "__DOCUMENT_TEXT__", document_text[:MAX_DOC_CHARS_IN]
                ),
            }
        ],
    )
    data = _tool_input(response)
    if not isinstance(data, dict) or "claims" not in data:
        raise ClaimParseError('record_claims input missing "claims"')
    return claims_from_records(data["claims"], document_text)


# --------------------------------------------------------------------------- #
# Sprint 1: sidecar persistence + generate-once.
#
# Mirrors documents.py's summary-sidecar pattern: documents._summary_path writes
# DOCUMENTS_DIR / f"{doc_id}.summary.txt" beside the doc, via a small path helper,
# generated once. Here the sidecar is DOCUMENTS_DIR / f"{doc_id}.claims.json",
# holding human-readable (indented) JSON. DOCUMENTS_DIR is read at call time so
# tests can redirect it into a tmp_path.
# --------------------------------------------------------------------------- #

# Top-level JSON key wrapping the claim list in the sidecar envelope.
_CLAIMS_KEY = "claims"


def _claims_path(doc_id: str) -> Path:
    """Sidecar path for ``doc_id``'s claim set, beside the document.

    Resolves ``DOCUMENTS_DIR`` at call time (module attribute lookup) so a test
    that monkeypatches ``claims.DOCUMENTS_DIR`` redirects the write.
    """
    return DOCUMENTS_DIR / f"{doc_id}.claims.json"


def _serialize(claims: list[Claim]) -> str:
    """Serialize ``claims`` to human-readable (indented, multi-line) JSON text."""
    envelope = {_CLAIMS_KEY: [c.to_dict() for c in claims]}
    return json.dumps(envelope, indent=2, ensure_ascii=False)


def _deserialize(text: str) -> list[Claim]:
    """Reconstruct :class:`Claim` records from serialized sidecar ``text``."""
    data = json.loads(text)
    raw = data[_CLAIMS_KEY] if isinstance(data, dict) else data
    return [
        Claim(
            id=item["id"],
            claim=item["claim"],
            anchor=item["anchor"],
            anchor_start=item.get("anchor_start"),
            anchor_end=item.get("anchor_end"),
            anchor_unresolved=item.get("anchor_unresolved", False),
        )
        for item in raw
    ]


def write_claims(doc_id: str, claims: list[Claim]) -> Path:
    """Persist ``claims`` to the ``{doc_id}.claims.json`` sidecar; return its path.

    Writes human-readable, indented JSON next to the document, mirroring
    documents._summary_path/save_upload's sidecar write. Creates
    ``DOCUMENTS_DIR`` if needed.
    """
    path = _claims_path(doc_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_serialize(claims))
    return path


def load_claims(doc_id: str) -> list[Claim] | None:
    """Return the cached claim set for ``doc_id``, or None if no sidecar exists."""
    path = _claims_path(doc_id)
    if not path.exists():
        return None
    return _deserialize(path.read_text())


def generate_claims(doc_id: str, document_text: str) -> list[Claim]:
    """Get-or-create the claim set for ``doc_id``, generated once per document.

    If the ``{doc_id}.claims.json`` sidecar already exists ON DISK, its claim
    records are reconstructed and returned WITHOUT invoking the Anthropic client
    (no redundant LLM call). Otherwise the (mocked-in-tests) LLM decomposition
    runs, the sidecar is written, and the fresh records are returned.

    The return type is identical on both paths: ``list[Claim]``.
    """
    cached = load_claims(doc_id)
    if cached is not None:
        return cached
    claims = extract_claims(document_text)
    write_claims(doc_id, claims)
    return claims
