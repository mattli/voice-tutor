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
    not reproduce it byte-perfectly. Its best-effort anchor is matched to the
    source only by PROVABLE substring — exact, or exact after cosmetic folding
    (dashes, quote styles, whitespace, case) — and the stored span is byte-exact
    against the document, which downstream scoring depends on. An anchor with
    genuine content drift is NOT given a guessed span; it is KEPT and flagged
    ``anchor_unresolved`` (raw text preserved, offsets null) rather than risking
    a silently-wrong span. Each claim records its ``resolution`` tier.

Offset contract for scoring: the document is canonicalized to Unicode NFC before
resolution, so ``anchor_start``/``anchor_end`` index the NFC form, and the stored
``anchor`` text is byte-exact against it. Consumers should prefer the ``anchor``
text; if slicing by offset, slice ``unicodedata.normalize("NFC", document_text)``.
See :class:`Claim` for the full contract.
"""

import hashlib
import json
import os
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
# A dense document's claim set (40-50 claims, each with a verbatim anchor) runs
# well past 8K output tokens; the old 8K cap truncated the tool call mid-JSON on
# dense docs. 16K gives comfortable headroom; extract_claims streams (required
# above ~16K to avoid SDK HTTP timeouts) and trips on a max_tokens stop reason.
# The remaining ceiling is the input bound (MAX_DOC_CHARS_IN), tracked in backlog.
MAX_TOKENS = 16_000
MAX_DOC_CHARS_IN = 100_000
# Bounded retries for RETRYABLE extraction outcomes only (empty claim list,
# transient API errors) — never for max_tokens truncation, which is deterministic.
MAX_EXTRACT_ATTEMPTS = 3

# The prompt encodes the required granularity:
#   * claims a person could articulate in 1-3 spoken sentences,
#   * not one claim per sentence, not chapter-level themes,
#   * consolidate facts the source repeats across sections,
#   * typically 10-50 claims, driven by document density.
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
    "- If the source repeats the same fact or statistic in multiple sections, "
    "express it in ONE claim only, anchored to its most substantive occurrence "
    "— do not transcribe the repetition.\n"
    "- Let the count follow the document's density: typically 10 to 50 claims; "
    "a short document may have fewer.\n\n"
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


class ClaimExtractionTruncated(ClaimParseError):
    """Raised when the model response was cut off at ``max_tokens``.

    A capped response leaves the tool-call JSON truncated (empty/partial input).
    This is DETERMINISTIC for a given document + token budget, so it is never
    retried — retrying burns tokens to hit the same wall. It signals that the
    document's claim set exceeds the output budget; raise the input/output bound.
    """


@dataclass(frozen=True)
class Claim:
    """A single assessable claim decomposed from a document.

    CONTRACT FOR SCORING / DOWNSTREAM CONSUMERS:
        ``anchor`` (the supporting-passage TEXT) is the authoritative, byte-exact
        evidence — PREFER IT. ``anchor_start``/``anchor_end`` are offsets into the
        NFC-NORMALIZED form of the source document (``unicodedata.normalize("NFC",
        document_text)``), NOT the raw bytes. For the common case (already-NFC
        text) they equal raw offsets; for a decomposed (NFD) source they do not.
        A consumer that slices by offset MUST slice the NFC form
        (``nfc_doc[anchor_start:anchor_end] == anchor``); a consumer that just
        needs the passage should use ``anchor`` directly and ignore the offsets.

    Attributes:
        id: Stable, purely positional id (``c1``, ``c2``, ...).
        claim: The claim text, a standalone assertion.
        anchor: The supporting passage. When resolved, this is the VERBATIM
            source substring at ``[anchor_start, anchor_end)`` of the NFC document
            — byte-exact. When unresolved, it is the model's raw best-effort
            anchor (kept for reference, not a source substring).
        anchor_start: Start char offset of the anchor in the NFC document, or None.
        anchor_end: End char offset (exclusive) in the NFC document, or None.
        anchor_unresolved: True if the anchor could not be located in the source
            as a provable substring; the claim is kept regardless.
        resolution: How the anchor was resolved — ``"exact"`` (verbatim
            substring), ``"normalized"`` (substring after cosmetic folding), or
            ``"unresolved"`` (genuine drift; offsets null, ``anchor`` is the raw
            model text). Lets scoring distinguish fallback strategies; the
            per-doc unresolved rate is a quality signal.
    """

    id: str
    claim: str
    anchor: str
    anchor_start: int | None = None
    anchor_end: int | None = None
    anchor_unresolved: bool = False
    resolution: str = "unresolved"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AnchorResolution:
    """Result of resolving a model anchor against the source document."""

    text: str
    start: int | None
    end: int | None
    unresolved: bool
    tier: str  # "exact" | "normalized" | "unresolved"


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
    """Return (normalized_text, index_map) for an ALREADY NFC-normalized string.

    ``index_map[k]`` is the index in ``s`` that produced the k-th normalized
    character — so a span located in normalized space maps back to a byte-exact
    span of ``s``. Applies only cosmetic folding (canonical dashes/quotes,
    collapsed whitespace, casefold). Unicode NFC composition is done by the
    caller on the WHOLE string first — per-character normalization cannot compose
    a base char with its following combining marks, which desynced NFD sources
    from NFC anchors of the same text.
    """
    out: list[str] = []
    idx: list[int] = []
    prev_ws = False
    for i, ch in enumerate(s):
        c = _canon_char(ch)
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


def resolve_anchor(
    anchor: str, document_text: str, norm_src: str | None = None, idx: list | None = None
) -> AnchorResolution:
    """Locate ``anchor`` in ``document_text`` by PROVABLE substring match only.

    ``document_text`` and ``anchor`` are NFC-normalized up front so a decomposed
    (NFD) source and a composed (NFC) anchor of the same text align. Two tiers
    produce offsets, and BOTH are provably a real source span:

      * ``"exact"``      — ``anchor`` is a verbatim substring of the NFC document.
      * ``"normalized"`` — ``anchor`` matches after cosmetic folding (dashes,
        quote styles, whitespace, case); the STORED span is the exact source
        substring at the mapped offsets, byte-exact.

    Anything else — genuine content drift (typos, dropped/added/hallucinated
    words) — is deliberately NOT resolved. Guessing an approximate span is an
    ambiguous alignment problem whose every simple heuristic silently truncates
    or over-extends the stored evidence (three such mirror bugs were found and
    removed). Instead the anchor is returned flagged ``"unresolved"`` — claim
    kept, raw anchor text preserved, offsets null — a SAFE, visible fallback
    rather than corrupt-but-confident evidence. Unresolved rate is a per-doc
    quality signal, not a failure.

    When ``norm_src``/``idx`` are supplied the caller guarantees ``document_text``
    is ALREADY NFC (see :func:`claims_from_records`), so the whole-document NFC
    pass is skipped — resolution stays O(claims + doc), not O(claims x doc).
    """
    anchor_nfc = unicodedata.normalize("NFC", anchor)
    if norm_src is None or idx is None:
        doc = unicodedata.normalize("NFC", document_text)
        norm_src, idx = _normalize_with_map(doc)
    else:
        doc = document_text  # caller guarantees this is already NFC

    # Tier 1: exact verbatim substring.
    pos = doc.find(anchor_nfc)
    if pos != -1:
        return AnchorResolution(anchor_nfc, pos, pos + len(anchor_nfc), False, "exact")

    # Tier 2: exact substring after cosmetic folding, mapped back byte-exact.
    na, _ = _normalize_with_map(anchor_nfc)
    na = na.strip()
    if na:
        p = norm_src.find(na)
        if p != -1:
            start, end = idx[p], idx[p + len(na) - 1] + 1
            return AnchorResolution(doc[start:end], start, end, False, "normalized")

    # Genuine content drift: keep the claim, flag unresolved, preserve raw anchor.
    return AnchorResolution(anchor_nfc, None, None, True, "unresolved")


def claims_from_records(raw_claims, document_text: str) -> list[Claim]:
    """Validate + resolve a list of ``{"claim", "anchor"}`` records into Claims.

    The order of the returned records mirrors the input order — no sorting,
    reordering, or deduping. The document is NFC-normalized and folded ONCE, then
    reused across all anchors (resolution is O(claims + doc), not O(claims x doc)).

    Raises:
        ClaimParseError: if the shape is wrong or any claim text / anchor is
            missing or empty. (A non-locatable anchor is NOT an error — the claim
            is kept and flagged ``anchor_unresolved``.)
    """
    if not isinstance(raw_claims, list):
        raise ClaimParseError('"claims" must be a list')

    nfc_doc = unicodedata.normalize("NFC", document_text)
    norm_src, idx = _normalize_with_map(nfc_doc)

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
        res = resolve_anchor(anchor, nfc_doc, norm_src, idx)
        out.append(
            Claim(
                id=_claim_id(i),
                claim=claim_text.strip(),
                anchor=res.text,
                anchor_start=res.start,
                anchor_end=res.end,
                anchor_unresolved=res.unresolved,
                resolution=res.tier,
            )
        )
    return out


def _tool_input(response) -> dict:
    """Pull the record_claims tool input (a parsed dict) from an SDK response."""
    for block in getattr(response, "content", []):
        if (
            getattr(block, "type", None) == "tool_use"
            and getattr(block, "name", None) == CLAIMS_TOOL["name"]
        ):
            return block.input
    raise ClaimParseError("model did not return a record_claims tool call")


# Transient API failures worth retrying (429 / 5xx / network); NOT 400/401/404.
_TRANSIENT_API_ERRORS = (
    anthropic.APIConnectionError,
    anthropic.RateLimitError,
    anthropic.InternalServerError,
)


def _stream_final_message(client, document_text: str):
    """Stream one forced record_claims turn and return the final SDK message.

    Streaming (vs a blocking create) is required at MAX_TOKENS=16K to avoid SDK
    HTTP timeouts on long generations.
    """
    with client.messages.stream(
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
    ) as stream:
        return stream.get_final_message()


def extract_claims(document_text: str) -> list[Claim]:
    """Decompose ``document_text`` into an ordered list of :class:`Claim` records.

    Pure function core: text in -> structured claim list out. The Anthropic
    client is constructed lazily here (never at import time) so importing this
    module reads no API key and performs no network I/O. Output shape is enforced
    by a forced tool call (no fences / no JSON-escaping failure modes possible).

    Retries up to ``MAX_EXTRACT_ATTEMPTS`` on RETRYABLE outcomes only — an empty
    claim list, a malformed tool input, or a transient API error. A ``max_tokens``
    truncation is deterministic and is NOT retried.

    Raises:
        ClaimExtractionTruncated: if the response was capped at ``max_tokens``.
        ClaimParseError: if, after all attempts, the model returned no usable
            record set (malformed input or persistently empty).
    """
    client = anthropic.Anthropic()
    last_error: Exception | None = None
    for _attempt in range(MAX_EXTRACT_ATTEMPTS):
        try:
            response = _stream_final_message(client, document_text)
        except _TRANSIENT_API_ERRORS as e:  # transient -> retry
            last_error = e
            continue

        # Truncation first: a capped response has partial/empty tool JSON. Raise a
        # named, non-retryable error rather than letting it look like empty input.
        if getattr(response, "stop_reason", None) == "max_tokens":
            raise ClaimExtractionTruncated(
                f"model response hit max_tokens ({MAX_TOKENS}); this document's "
                "claim set exceeds the output budget — raise the bound, do not retry"
            )

        try:
            data = _tool_input(response)
            if not isinstance(data, dict) or not isinstance(data.get("claims"), list):
                raise ClaimParseError("record_claims input missing a claims list")
            result = claims_from_records(data["claims"], document_text)
        except ClaimParseError as e:  # malformed but not truncated -> retry
            last_error = e
            continue

        if not result:  # empty claim list -> retryable degenerate response
            last_error = ClaimParseError("model returned an empty claim list")
            continue
        return result

    raise last_error if last_error else ClaimParseError("claim extraction failed")


# --------------------------------------------------------------------------- #
# Sprint 1: sidecar persistence + generate-once.
#
# Mirrors documents.py's summary-sidecar pattern: documents._summary_path writes
# DOCUMENTS_DIR / f"{doc_id}.summary.txt" beside the doc, via a small path helper,
# generated once. Here the sidecar is DOCUMENTS_DIR / f"{doc_id}.claims.json",
# holding human-readable (indented) JSON. DOCUMENTS_DIR is read at call time so
# tests can redirect it into a tmp_path.
# --------------------------------------------------------------------------- #

# Top-level JSON keys in the sidecar envelope.
_CLAIMS_KEY = "claims"
_HASH_KEY = "source_hash"


def _hash_source(document_text: str) -> str:
    """Stable content hash of the input text — the cache-integrity key.

    A cached rubric is only valid for the exact document it was generated from.
    The served document can drift (e.g. a vault page edited after the rubric was
    cached), so the hash lets get-or-create detect skew and regenerate rather
    than serve a rubric that silently disagrees with the current text.
    """
    return hashlib.sha256(document_text.encode("utf-8")).hexdigest()


def _claims_path(doc_id: str) -> Path:
    """Sidecar path for ``doc_id``'s claim set, beside the document.

    Resolves ``DOCUMENTS_DIR`` at call time (module attribute lookup) so a test
    that monkeypatches ``claims.DOCUMENTS_DIR`` redirects the write.
    """
    return DOCUMENTS_DIR / f"{doc_id}.claims.json"


def _serialize(claims: list[Claim], source_hash: str | None = None) -> str:
    """Serialize ``claims`` to human-readable (indented, multi-line) JSON text.

    The envelope carries a ``source_hash`` (the hash of the document the claims
    were generated from) so the cache can detect document drift.
    """
    envelope = {_HASH_KEY: source_hash, _CLAIMS_KEY: [c.to_dict() for c in claims]}
    return json.dumps(envelope, indent=2, ensure_ascii=False)


def _records_to_claims(data) -> list[Claim]:
    """Reconstruct :class:`Claim` records from a parsed sidecar envelope/list.

    Raises ClaimParseError (not KeyError) on a record missing a required field,
    so callers can degrade cleanly. A record lacking offsets is coerced to
    ``anchor_unresolved`` rather than deserializing as resolved-with-null-span
    (which downstream would slice as ``document_text[None:None]`` — the whole
    document).
    """
    raw = data[_CLAIMS_KEY] if isinstance(data, dict) else data
    if not isinstance(raw, list):
        raise ClaimParseError("sidecar has no claims list")
    out: list[Claim] = []
    for item in raw:
        try:
            cid, claim, anchor = item["id"], item["claim"], item["anchor"]
        except (KeyError, TypeError) as e:
            raise ClaimParseError(f"malformed sidecar record: {e!r}") from e
        start = item.get("anchor_start")
        end = item.get("anchor_end")
        unresolved = (
            bool(item.get("anchor_unresolved", False)) or start is None or end is None
        )
        resolution = "unresolved" if unresolved else item.get("resolution", "normalized")
        out.append(Claim(cid, claim, anchor, start, end, unresolved, resolution))
    return out


def _deserialize(text: str) -> list[Claim]:
    """Reconstruct :class:`Claim` records from serialized sidecar ``text``."""
    return _records_to_claims(json.loads(text))


def write_claims(
    doc_id: str, claims: list[Claim], source_hash: str | None = None
) -> Path:
    """Persist ``claims`` to the ``{doc_id}.claims.json`` sidecar; return its path.

    Writes human-readable, indented JSON next to the document, mirroring
    documents._summary_path/save_upload's sidecar write. Creates ``DOCUMENTS_DIR``
    if needed. The write is ATOMIC (temp file + os.replace) so an interrupted
    write never leaves a half-written sidecar that would poison the cache. Pass
    ``source_hash`` to stamp the sidecar with the hash of the source document.
    """
    path = _claims_path(doc_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(_serialize(claims, source_hash), encoding="utf-8")
    os.replace(tmp, path)  # atomic on the same filesystem
    return path


def load_claims(doc_id: str) -> list[Claim] | None:
    """Return the cached claim set for ``doc_id``, or None if no sidecar exists."""
    path = _claims_path(doc_id)
    if not path.exists():
        return None
    return _deserialize(path.read_text(encoding="utf-8"))


def _cached_source_hash(doc_id: str) -> str | None:
    """Return the ``source_hash`` stamped on ``doc_id``'s sidecar, or None.

    None when no sidecar exists, when a sidecar predates hashing (no stamp), or
    when the sidecar is unreadable/corrupt — in every case the cache cannot be
    verified, so callers regenerate rather than crash.
    """
    path = _claims_path(doc_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    return data.get(_HASH_KEY) if isinstance(data, dict) else None


def generate_claims(doc_id: str, document_text: str) -> list[Claim]:
    """Get-or-create the claim set for ``doc_id``, generated once per document.

    Cache integrity is keyed on the document's content hash: the sidecar is
    reused (NO LLM call) only when its stamped ``source_hash`` matches the hash
    of ``document_text``. If the sidecar is absent, its hash disagrees with the
    current text (the served document drifted since the rubric was cached), OR it
    is corrupt/unreadable, the (mocked-in-tests) LLM decomposition re-runs and the
    sidecar is rewritten. The return type is identical on both paths: ``list[Claim]``.
    """
    current_hash = _hash_source(document_text)
    path = _claims_path(doc_id)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get(_HASH_KEY) == current_hash:
                return _records_to_claims(data)  # cache hit — single read
        except (ValueError, OSError, ClaimParseError):
            pass  # corrupt / unreadable / malformed -> regenerate
    claims = extract_claims(document_text)
    write_claims(doc_id, claims, source_hash=current_hash)
    return claims
