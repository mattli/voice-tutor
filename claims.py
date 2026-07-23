"""Claim extraction: decompose a document's text into discrete, assessable claims.

This is the "rubric" that both conversation steering and future scoring will
consume. Sprint 0 ships ONLY the pure decomposition core: document text in ->
ordered list of discrete claim records out.

Design constraints (kept deliberately minimal so the verifier env stays
satisfiable):
  * Imports ONLY the ``anthropic`` SDK and the standard library. No bot.py,
    app.py, pipecat, or fastapi.
  * No API key is read at import time. The Anthropic client is constructed
    lazily inside :func:`extract_claims`, mirroring documents._generate_summary.
  * Ids are purely positional and deterministic (c1, c2, c3, ...), independent
    of claim text content.
  * The parser validates the model's structured output and RAISES on malformed
    responses (empty claim text, empty anchor, or an anchor that is not a
    verbatim substring of the document) rather than silently skipping.
"""

import json
from dataclasses import dataclass, asdict

import anthropic

MODEL = "claude-sonnet-4-5-20250929"
MAX_TOKENS = 8_000
MAX_DOC_CHARS_IN = 100_000

# The prompt encodes the required granularity:
#   * claims a person could articulate in 1-3 spoken sentences,
#   * not one claim per sentence, not chapter-level themes,
#   * roughly 10-40 claims for a typical document.
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
    "For each claim, provide:\n"
    '  - "claim": the claim text, phrased as a standalone assertion.\n'
    '  - "anchor": the supporting quote/passage copied VERBATIM from the '
    "document (an exact substring — do not paraphrase, do not add ellipses).\n\n"
    "Respond with ONLY a JSON object of the form "
    '{"claims": [{"claim": "...", "anchor": "..."}, ...]} — no preamble, no '
    "markdown fences, no commentary.\n\n"
    "Document:\n__DOCUMENT_TEXT__"
)


class ClaimParseError(Exception):
    """Raised when the model's response cannot be parsed/validated into claims."""


@dataclass(frozen=True)
class Claim:
    """A single assessable claim decomposed from a document.

    Attributes:
        id: Stable, purely positional id (``c1``, ``c2``, ...).
        claim: The claim text, a standalone assertion.
        anchor: The supporting quote/passage, a verbatim substring of the doc.
    """

    id: str
    claim: str
    anchor: str

    def to_dict(self) -> dict:
        return asdict(self)


def _claim_id(index: int) -> str:
    """Deterministic, purely positional id for the ``index``-th (0-based) claim."""
    return f"c{index + 1}"


def _extract_text_payload(response) -> str:
    """Pull the raw text payload from an Anthropic SDK-shaped response.

    Mirrors documents._generate_summary: ``response.content[0].text``.
    """
    try:
        return response.content[0].text
    except (AttributeError, IndexError, TypeError) as e:
        raise ClaimParseError(f"unexpected response shape: {e!r}") from e


def parse_claims(payload: str, document_text: str) -> list[Claim]:
    """Parse a model text ``payload`` into validated, ordered :class:`Claim`s.

    The order of the returned records mirrors the order of claims in the
    payload — no sorting, reordering, or deduping is performed.

    Raises:
        ClaimParseError: if the payload is not valid JSON of the expected shape,
            if any claim text or anchor is missing/empty, or if any anchor is
            not a verbatim substring of ``document_text``.
    """
    text = payload.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ClaimParseError(f"response was not valid JSON: {e}") from e

    if not isinstance(data, dict) or "claims" not in data:
        raise ClaimParseError('response JSON missing "claims" key')
    raw_claims = data["claims"]
    if not isinstance(raw_claims, list):
        raise ClaimParseError('"claims" must be a list')

    claims: list[Claim] = []
    for i, item in enumerate(raw_claims):
        if not isinstance(item, dict):
            raise ClaimParseError(f"claim {i} is not an object")
        claim_text = item.get("claim")
        anchor = item.get("anchor")
        if not isinstance(claim_text, str) or not claim_text.strip():
            raise ClaimParseError(f"claim {i} has missing/empty claim text")
        if not isinstance(anchor, str) or not anchor.strip():
            raise ClaimParseError(f"claim {i} has missing/empty anchor")
        if anchor not in document_text:
            raise ClaimParseError(
                f"claim {i} anchor is not a verbatim substring of the document"
            )
        claims.append(Claim(id=_claim_id(i), claim=claim_text.strip(), anchor=anchor))

    return claims


def extract_claims(document_text: str) -> list[Claim]:
    """Decompose ``document_text`` into an ordered list of :class:`Claim` records.

    Pure function core: text in -> structured claim list out. The Anthropic
    client is constructed lazily here (never at import time) so importing this
    module reads no API key and performs no network I/O.

    Raises:
        ClaimParseError: if the model response cannot be validated (see
            :func:`parse_claims`).
    """
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[
            {
                "role": "user",
                "content": CLAIMS_PROMPT.replace(
                    "__DOCUMENT_TEXT__", document_text[:MAX_DOC_CHARS_IN]
                ),
            }
        ],
    )
    payload = _extract_text_payload(response)
    return parse_claims(payload, document_text)
