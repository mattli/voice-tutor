"""Coverage judge: judge which document claims a study session covered.

Sprint 0 scope (this file): the OFFLINE foundations everything else builds on —
the input contract (claim list + indexed transcript) and the strict-JSON
transport defenses this repo has learned the hard way. There is NO live model
call implemented here yet; the actual Haiku invocation, the v2 judge prompt +
its hash, ``union_coverage``, and the CLI are later sprints.

Design constraints (mirroring ``claims.py`` so the verifier env stays
satisfiable and hermetic tests never touch the network):

  * This module imports ONLY the standard library at module scope. It does NOT
    import ``bot``, ``app``, ``documents``, ``pipecat``, ``fastapi``, or the
    ``anthropic`` SDK at import time. ``import coverage_judge`` reads no API key
    and performs no network I/O.
  * The transport-defense parsing functions (:func:`strip_code_fences`,
    :func:`parse_verdicts`, and the completeness checks) are PURE and
    model-independent — no client, no I/O. They take text / already-parsed data
    and a claim-id set, and return validated verdicts or raise a typed error.
  * A future model-invocation seam will construct ``anthropic.Anthropic()``
    LAZILY (never at import time, mirroring ``claims.extract_claims`` /
    ``documents._generate_summary``). Sprint 0 does not implement or exercise it.

The verdict object shape is fixed by the goal:
    {"claim_id": str, "covered": bool, "turns": [int, ...]}
and the completeness rule is: every input claim id must appear EXACTLY ONCE in
the model's verdict list (regardless of its ``covered`` value), so a truncated
or padded response is caught structurally rather than silently accepted.
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

# --------------------------------------------------------------------------- #
# Input contract: claims and indexed transcripts.
# --------------------------------------------------------------------------- #

# Top-level envelope key of a `.claims.json` sidecar (see claims.py _CLAIMS_KEY).
_CLAIMS_KEY = "claims"
# Top-level envelope key of a transcript file (see the ~/.voice-tutor transcript
# shape: {"session_start", "session_end", "turn_count", "turns": [...]}).
_TURNS_KEY = "turns"


class CoverageInputError(Exception):
    """Raised when a claim list or transcript is malformed.

    A single typed, catchable error for every input-validation failure, so the
    transport/CLI layer can surface a clear message instead of leaking a raw
    ``KeyError`` / ``IndexError`` / ``TypeError`` from deep inside a loader.
    """


@dataclass(frozen=True)
class Claim:
    """A single claim to judge coverage for: a stable id + its assertion text."""

    id: str
    text: str


@dataclass(frozen=True)
class Turn:
    """One transcript turn, normalized to a stable integer ``index`` space.

    ``index`` is the value a verdict's ``turns: [int]`` list references. See
    :func:`load_transcript` for how the index space is established (assigned
    contiguously when absent, preserved when supplied).
    """

    index: int
    role: str
    content: str


def _claim_records(data: Any) -> list:
    """Return the raw list of claim records from a sidecar envelope or bare list.

    Accepts the repo's `.claims.json` envelope ``{"claims": [...]}`` or a bare
    ``[...]`` list. Raises :class:`CoverageInputError` on any other top-level
    shape (mirrors claims._records_to_claims, but with a typed error instead of
    a bare ``KeyError``).
    """
    if isinstance(data, dict):
        raw = data.get(_CLAIMS_KEY)
        if not isinstance(raw, list):
            raise CoverageInputError(
                'claim list envelope must have a "claims" list'
            )
        return raw
    if isinstance(data, list):
        return data
    raise CoverageInputError(
        "claim list must be a list or a {\"claims\": [...]} envelope, "
        f"got {type(data).__name__}"
    )


def load_claims(data: Any) -> list[Claim]:
    """Normalize a claim list into an ordered list of :class:`Claim`.

    ``data`` is either the repo's `.claims.json` envelope (``{"claims": [{"id",
    "claim", ...}]}``) or a bare list of the same records. The sidecar's
    ``claim`` field is normalized to the claim ``text`` (a bare ``text`` field is
    also accepted). Input ORDER is preserved — no sorting or deduping — since the
    judge and downstream consumers key on the ids, not position.

    Raises:
        CoverageInputError: on a wrong top-level shape, an empty claim list, a
            record that is not an object, a missing/blank id, a missing/blank
            text, or a duplicate claim id.
    """
    raw = _claim_records(data)
    if not raw:
        raise CoverageInputError("claim list is empty")

    out: list[Claim] = []
    seen: set[str] = set()
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise CoverageInputError(f"claim {i} is not an object")
        cid = item.get("id")
        if not isinstance(cid, str) or not cid.strip():
            raise CoverageInputError(f"claim {i} has a missing/blank id")
        # Accept the sidecar's "claim" field (primary) or a bare "text" field.
        text = item.get("claim", item.get("text"))
        if not isinstance(text, str) or not text.strip():
            raise CoverageInputError(f"claim {cid!r} has missing/blank text")
        cid = cid.strip()
        if cid in seen:
            raise CoverageInputError(f"duplicate claim id {cid!r}")
        seen.add(cid)
        out.append(Claim(id=cid, text=text.strip()))
    return out


def claim_ids(claims: list[Claim]) -> list[str]:
    """Return the ordered list of claim ids (a small convenience for callers)."""
    return [c.id for c in claims]


def load_transcript(data: Any) -> list[Turn]:
    """Normalize a transcript into an indexed list of :class:`Turn`.

    ``data`` is either the repo's transcript envelope (``{"turns": [{"role",
    "content", ...}]}``) or a bare list of turn objects. Each turn is normalized
    to ``{index:int, role, content}``, guaranteeing a stable integer index space
    so a verdict's ``turns: [int]`` can reference transcript positions.

    Index rule:
      * If NO turn carries an ``index``, assign contiguous 0-based indices in
        list order (the real ~/.voice-tutor transcripts have no index field).
      * If indices ARE supplied, they must ALL be ints and unique, and are
        preserved as-is.
      * A partially-supplied index set (some turns have an index, some do not)
        is malformed — it is ambiguous whether to trust or overwrite, so it is
        rejected rather than silently guessed.

    Raises:
        CoverageInputError: on a wrong top-level shape, a turn that is not an
            object, a missing/non-string role or content, a non-int supplied
            index, a partially-supplied index set, or a duplicate supplied index.
    """
    if isinstance(data, dict):
        raw = data.get(_TURNS_KEY)
        if not isinstance(raw, list):
            raise CoverageInputError('transcript envelope must have a "turns" list')
    elif isinstance(data, list):
        raw = data
    else:
        raise CoverageInputError(
            "transcript must be a list or a {\"turns\": [...]} envelope, "
            f"got {type(data).__name__}"
        )

    # First pass: validate role/content and collect any supplied indices, so we
    # can decide the all-or-none index policy before assigning anything.
    supplied: list[int | None] = []
    for i, turn in enumerate(raw):
        if not isinstance(turn, dict):
            raise CoverageInputError(f"turn {i} is not an object")
        role = turn.get("role")
        content = turn.get("content")
        if not isinstance(role, str) or not role:
            raise CoverageInputError(f"turn {i} has a missing/invalid role")
        if not isinstance(content, str):
            raise CoverageInputError(f"turn {i} has missing/invalid content")
        if "index" in turn:
            idx = turn["index"]
            # bool is an int subclass; reject it so True/False can't pose as 1/0.
            if not isinstance(idx, int) or isinstance(idx, bool):
                raise CoverageInputError(
                    f"turn {i} has a non-integer index {idx!r}"
                )
            supplied.append(idx)
        else:
            supplied.append(None)

    have_index = [s is not None for s in supplied]
    if any(have_index) and not all(have_index):
        raise CoverageInputError(
            "transcript has a partially-supplied index set: either every turn "
            "must carry an integer index or none may"
        )

    if all(have_index):
        if len(set(supplied)) != len(supplied):
            raise CoverageInputError("transcript has duplicate turn indices")
        indices = [int(s) for s in supplied]  # type: ignore[arg-type]
    else:
        indices = list(range(len(raw)))  # contiguous 0-based, list order

    return [
        Turn(index=indices[i], role=turn["role"], content=turn["content"])
        for i, turn in enumerate(raw)
    ]


# --------------------------------------------------------------------------- #
# Transport defenses: strict-JSON verdict parsing.
#
# These are PURE, model-independent functions. They take model TEXT (or already-
# parsed data) plus the set of claim ids being judged, and return validated
# verdicts or raise a typed parse error. No Anthropic client, no I/O — so the
# hermetic suite exercises every fence / truncation / id-mismatch case offline.
# --------------------------------------------------------------------------- #


class VerdictParseError(Exception):
    """Base class for every strict-JSON verdict parsing/validation failure.

    A single defined base so a caller (and the future bounded-retry loop) can
    catch ONE exception type for any malformed model output — invalid JSON,
    truncation, or a claim-id mismatch — rather than a raw
    ``json.JSONDecodeError`` leaking out. The subclasses below let callers
    distinguish WHY a response was rejected (e.g. retry-vs-fail decisions).
    """


class VerdictJSONError(VerdictParseError):
    """The model text was not valid JSON (after markdown-fence stripping)."""


class VerdictShapeError(VerdictParseError):
    """A verdict entry had the wrong shape/type (not {claim_id, covered, turns})."""


class VerdictCountError(VerdictParseError):
    """The verdict count did not equal the claim count — truncated or padded.

    The primary truncation defense: a response cut off mid-list yields fewer
    verdicts than there are claims, and this is caught structurally.
    """


class UnknownClaimIdError(VerdictParseError):
    """A verdict referenced a claim_id that is not in the input claim set."""


class DuplicateClaimIdError(VerdictParseError):
    """The same claim_id appeared more than once in the verdict list."""


class MissingClaimIdError(VerdictParseError):
    """An input claim_id had no verdict in the model's output."""


class InvalidTurnCitationError(VerdictParseError):
    """A verdict's ``turns`` citation is not supported by the transcript.

    Two cases, both of which would otherwise yield a SILENTLY WRONG coverage
    number rather than a loud failure:
      * a cited turn index does not exist in the judged transcript (a
        hallucinated citation);
      * ``covered: true`` with an EMPTY ``turns`` list — the judge prompt's
        contract is "no citable turn => covered: false and turns: []", so an
        uncited claim of coverage is a contract violation, not a valid verdict.

    A model-output defect (hence a :class:`VerdictParseError`, retryable within
    the bound), not an operator input error.
    """


class MassCitationDowngradeError(VerdictParseError):
    """EVERY claim the model asserted coverage for was downgraded by repair.

    The per-claim repair in :func:`repair_turn_citations` contains the damage of
    ONE miscited claim. But when the model's citations are in a DIFFERENT INDEX
    SPACE than the transcript it was given (1-based instead of 0-based, line
    numbers, ids carried over from another rendering), every citation is bogus,
    every ``covered: true`` is downgraded, and the repair produces a
    perfectly-shaped verdict set reading ZERO CLAIMS COVERED.

    That is not a coverage number — it is a transport failure wearing one. And
    because coverage is APPEND-ONLY (see ``coverage_store.write_sidecar``),
    writing it makes the false zero PERMANENT: the session can never be
    re-judged without an explicit ``--force``. A monotonic bar that can only be
    corrected by hand is worse than no bar.

    So this case is refused rather than repaired: a retryable parse error, which
    the bounded retry re-issues once, and — if the second attempt does the same —
    surfaces as a judge failure, i.e. NO COVERAGE DATA for that session. No data
    is recoverable later; a permanent wrong zero is not.

    Deliberately narrow: it fires only when SEVERAL claims
    (:data:`MIN_MASS_DOWNGRADE_CLAIMS`) were downgraded AND nothing survived. A
    model that honestly finds nothing covered downgrades nothing and is
    unaffected, and a single miscited claim is still repaired, not refused.
    """


class VerdictTruncatedError(Exception):
    """The model's reply was cut off at the output token cap (max_tokens).

    DELIBERATELY NOT a :class:`VerdictParseError`. That base class is the
    RETRYABLE family — :func:`judge_coverage` catches it and tries again — and
    truncation is the one failure that must NEVER be retried: it is
    DETERMINISTIC, so re-sending the identical prompt at temperature 0
    re-truncates identically and simply doubles the bill.

    Keeping it outside the retryable hierarchy makes that guarantee STRUCTURAL
    rather than positional. Previously it held only because the raise sits
    outside the ``try``; any future ``except VerdictParseError: retry`` would
    have silently re-enabled the double-billing. Now the type prevents it.

    The remedy is a larger ``max_tokens`` or fewer claims per call.
    """


@dataclass(frozen=True)
class Verdict:
    """A per-claim coverage verdict from the judge.

    ``turns`` is the list of transcript indices where the claim was covered; it
    is empty for a ``covered: false`` verdict (an empty list is valid, not an
    error).

    ``reason`` is the judge's brief per-claim justification (the decomposition
    the v2 prompt requires). It is PERSISTED — for judgment machinery the stated
    rationale is the most valuable field for auditing a verdict after the fact,
    so it is carried through to the output rather than discarded. ``None`` when
    the model omitted it (the field is optional, not required).
    """

    claim_id: str
    covered: bool
    turns: list[int] = field(default_factory=list)
    reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "covered": self.covered,
            "turns": list(self.turns),
            "reason": self.reason,
        }


def strip_code_fences(text: str) -> str:
    """Strip a surrounding markdown code fence from a model response, if present.

    Handles ```` ```json ... ``` ````, plain ```` ``` ... ``` ```` (with or
    without a language tag), and surrounding whitespace. A no-op on already-clean
    JSON text. This is deliberately a SEPARATE step from parsing so the two
    functions compose cleanly and each is independently testable.
    """
    s = text.strip()
    if not s.startswith("```"):
        return s
    # Drop the opening fence line (```lang or bare ```), keeping the rest.
    first_newline = s.find("\n")
    if first_newline == -1:
        # A lone "```..." with no body — nothing recoverable; return stripped.
        return s
    body = s[first_newline + 1 :]
    # Drop a trailing closing fence if present.
    stripped = body.rstrip()
    if stripped.endswith("```"):
        stripped = stripped[: stripped.rfind("```")]
    return stripped.strip()


def loads_json(text: str) -> Any:
    """Fence-strip then ``json.loads``, raising :class:`VerdictJSONError`.

    Wraps the raw ``json.JSONDecodeError`` in the module's parse-error hierarchy
    so callers catch a single defined type rather than a stdlib error.
    """
    cleaned = strip_code_fences(text)
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError) as e:
        raise VerdictJSONError(f"model output was not valid JSON: {e}") from e


def _verdict_records(data: Any) -> list:
    """Extract the raw verdict list from a parsed payload (envelope or list).

    Accepts either a bare list of verdict objects or a ``{"verdicts": [...]}``
    envelope, so the judge prompt can ask for either shape without the parser
    caring. Raises :class:`VerdictShapeError` otherwise.
    """
    if isinstance(data, dict):
        raw = data.get("verdicts")
        if not isinstance(raw, list):
            raise VerdictShapeError('verdict payload envelope must have a "verdicts" list')
        return raw
    if isinstance(data, list):
        return data
    raise VerdictShapeError(
        "verdict payload must be a list or a {\"verdicts\": [...]} envelope, "
        f"got {type(data).__name__}"
    )


def _coerce_verdict(item: Any) -> Verdict:
    """Validate one raw verdict record into a typed :class:`Verdict`.

    Raises :class:`VerdictShapeError` on a missing field or wrong type. An empty
    ``turns`` list is valid (a covered:false verdict). ``bool`` is checked
    strictly (an int like 1 is NOT accepted for ``covered``).
    """
    if not isinstance(item, dict):
        raise VerdictShapeError(f"verdict is not an object: {item!r}")
    if "claim_id" not in item:
        raise VerdictShapeError("verdict is missing claim_id")
    cid = item["claim_id"]
    if not isinstance(cid, str) or not cid.strip():
        raise VerdictShapeError(f"verdict has a missing/blank claim_id: {cid!r}")
    covered = item.get("covered")
    if not isinstance(covered, bool):
        raise VerdictShapeError(
            f"verdict {cid!r} covered must be a bool, got {covered!r}"
        )
    turns = item.get("turns", [])
    if not isinstance(turns, list):
        raise VerdictShapeError(f"verdict {cid!r} turns must be a list, got {turns!r}")
    norm_turns: list[int] = []
    for t in turns:
        if not isinstance(t, int) or isinstance(t, bool):
            raise VerdictShapeError(
                f"verdict {cid!r} turns must be a list of ints, got element {t!r}"
            )
        norm_turns.append(t)
    return Verdict(
        claim_id=cid.strip(),
        covered=covered,
        turns=norm_turns,
        reason=_coerce_reason(item.get("reason")),
    )


# Cap on a coerced rationale, so a pathological model reply cannot bloat the
# sidecar. Generous — a real per-claim reason is a sentence or two.
_MAX_REASON_CHARS = 2000


def _coerce_reason(reason: Any) -> str | None:
    """Normalize the judge's per-claim rationale to a string (or ``None``).

    ``reason`` is AUDITING METADATA, not the answer. Rejecting a whole verdict
    set because the model returned the rationale in the wrong type cost the
    session its entire coverage number for a harmless model quirk — the run
    failed, a retry was burned, and a recoverable hiccup became "no number at
    all". So a non-string rationale is COERCED (``str``, capped at
    :data:`_MAX_REASON_CHARS`) rather than raised on: the verdict — the part that
    actually matters — is always kept.

    ``None`` (the model omitted it) passes through as ``None``; a string is kept
    as-is apart from the length cap. Pure.
    """
    if reason is None:
        return None
    text = reason if isinstance(reason, str) else str(reason)
    if len(text) > _MAX_REASON_CHARS:
        return text[:_MAX_REASON_CHARS] + "…[truncated]"
    return text


def verify_turn_citations(verdicts: list[Verdict], valid_turn_indices) -> None:
    """Assert every verdict's ``turns`` citation refers to a turn that EXISTS.

    Scope note: this is an EXISTENCE check, not a semantic one. It cannot verify
    that a cited turn actually supports the claim (that judgement is the model's),
    nor does it currently require the cited turn to be an assistant turn.

    The defense against a SILENTLY WRONG coverage credit — the failure mode that
    inflates the number rather than raising. Two rules, both from the judge
    prompt's own contract:

      * every cited index must exist in ``valid_turn_indices`` (the judged
        transcript's real index space) — a hallucinated citation is rejected;
      * ``covered: true`` must cite at least one turn — the prompt says "no
        citable turn => covered: false and turns: []", so an uncited coverage
        claim violates the contract.

    A ``covered: false`` verdict with empty ``turns`` is the normal, valid case.
    Raises :class:`InvalidTurnCitationError` (a retryable parse error).
    """
    valid = set(valid_turn_indices)
    for v in verdicts:
        unknown = [t for t in v.turns if t not in valid]
        if unknown:
            raise InvalidTurnCitationError(
                f"verdict {v.claim_id!r} cites turn(s) {unknown} that do not exist "
                f"in the judged transcript (valid indices: {len(valid)} turns)"
            )
        if v.covered and not v.turns:
            raise InvalidTurnCitationError(
                f"verdict {v.claim_id!r} is covered:true with no cited turn — the "
                "judge contract requires a citation for coverage"
            )


@dataclass(frozen=True)
class CitationRepair:
    """A record of ONE claim's citation being repaired rather than failing the run.

    Emitted by :func:`repair_turn_citations`. ``dropped_turns`` are cited indices
    that do not exist in the judged transcript; ``downgraded`` is True when the
    repair left a ``covered: true`` verdict with no citation at all, so the claim
    was flipped to not-covered (the conservative direction — a repair can never
    invent coverage).
    """

    claim_id: str
    dropped_turns: list[int] = field(default_factory=list)
    downgraded: bool = False

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "dropped_turns": list(self.dropped_turns),
            "downgraded": self.downgraded,
        }


def repair_turn_citations(
    verdicts: list[Verdict], valid_turn_indices
) -> tuple[list[Verdict], list[CitationRepair]]:
    """Repair unsupported citations claim-by-claim instead of failing the run.

    The blast-radius fix for the strict :func:`verify_turn_citations`: on a real
    63-claim answer key, ONE hallucinated turn index rejected all 63 verdicts,
    burned a retry, and (on a second failure) left the session with no coverage
    number at all. A model that miscites one claim has still judged the other 62
    perfectly well, so the damage is contained to the claim that was miscited:

      * a cited index that does not exist in ``valid_turn_indices`` is DROPPED
        from that verdict's ``turns``;
      * if dropping leaves a ``covered: true`` verdict with NO citation, that one
        claim is DOWNGRADED to ``covered: false`` (the judge prompt's contract is
        "no citable turn => not covered"), as is a ``covered: true`` that cited
        nothing to begin with.

    Every repair moves coverage DOWN, never up, so a repaired verdict set can
    never credit coverage the transcript does not support — the property that
    makes repairing safer than rejecting-and-retrying. Verdicts needing no repair
    are returned unchanged (same objects).

    Returns ``(verdicts, repairs)``; ``repairs`` is empty when nothing was
    touched, and is persisted into the verdict object so a repaired number is
    auditable rather than silent. Pure: no I/O, no model call.

    Scope limit: repair contains the damage of a FEW bad citations. The case
    where EVERY coverage claim is downgraded is a different failure (an
    index-space mismatch, not a miscitation) and is refused by
    :func:`judge_coverage` via :func:`is_mass_citation_downgrade` rather than
    repaired — a verdict set repaired down to zero would otherwise be persisted
    as a permanent false zero under the append-only policy.
    """
    valid = set(valid_turn_indices)
    out: list[Verdict] = []
    repairs: list[CitationRepair] = []
    for v in verdicts:
        dropped = [t for t in v.turns if t not in valid]
        if not dropped and not (v.covered and not v.turns):
            out.append(v)
            continue
        kept = [t for t in v.turns if t in valid]
        downgraded = v.covered and not kept
        out.append(
            Verdict(
                claim_id=v.claim_id,
                covered=False if downgraded else v.covered,
                turns=kept,
                reason=v.reason,
            )
        )
        repairs.append(
            CitationRepair(
                claim_id=v.claim_id, dropped_turns=dropped, downgraded=downgraded
            )
        )
    return out, repairs


# How many independently-downgraded claims it takes to call a verdict set an
# index-space mismatch rather than a miscitation. TWO, not one: a single claim
# whose citation was bogus is exactly the case per-claim repair exists to
# contain, and one asserted-then-downgraded claim carries no evidence
# distinguishing "the model slipped on its only coverage claim" from "every
# index is wrong". Two or more claims independently citing indices that all fail,
# with nothing left standing, is the corroborated signature.
MIN_MASS_DOWNGRADE_CLAIMS = 2


def is_mass_citation_downgrade(
    verdicts: list[Verdict], repairs: list[CitationRepair]
) -> bool:
    """True when repair downgraded EVERY coverage claim the model made, en masse.

    The signature of a citation INDEX-SPACE mismatch rather than a one-off
    miscitation: at least :data:`MIN_MASS_DOWNGRADE_CLAIMS` claims were
    downgraded, and after repair not a single claim remains covered. See
    :class:`MassCitationDowngradeError` for why that case must fail loudly
    instead of being persisted as a permanent zero.

    Pure. Returns False when nothing was downgraded (including the ordinary
    "model honestly found nothing covered" case, which downgrades nothing), and
    False for a single downgraded claim — repair contains that one, by design.
    """
    downgraded = sum(1 for r in repairs if r.downgraded)
    if downgraded < MIN_MASS_DOWNGRADE_CLAIMS:
        return False
    return not any(v.covered for v in verdicts)


def verify_claim_id_coverage(present_ids, expected_ids) -> None:
    """Assert ``present_ids`` is a bijection onto the DISTINCT ``expected_ids``.

    The pure completeness check underlying :func:`parse_verdicts`, factored out
    so the three id-level defenses are independently reachable and testable
    regardless of the count gate that guards the full parse path:

      * an id in ``present_ids`` not in ``expected_ids`` -> :class:`UnknownClaimIdError`;
      * an id repeated in ``present_ids`` -> :class:`DuplicateClaimIdError`;
      * a distinct expected id absent from ``present_ids`` -> :class:`MissingClaimIdError`.

    ``present_ids`` is the ordered sequence of claim_ids the model returned;
    ``expected_ids`` is the input claim-id set (duplicates collapsed to distinct).
    Returns ``None`` when every distinct expected id is present exactly once.
    """
    expected_set = set(expected_ids)
    seen: set[str] = set()
    for cid in present_ids:
        if cid not in expected_set:
            raise UnknownClaimIdError(f"unknown claim_id {cid!r}")
        if cid in seen:
            raise DuplicateClaimIdError(f"duplicate claim_id {cid!r}")
        seen.add(cid)
    missing = [cid for cid in expected_set if cid not in seen]
    if missing:
        raise MissingClaimIdError(
            f"no verdict for claim id(s): {', '.join(sorted(missing))}"
        )


@dataclass(frozen=True)
class ParsedVerdicts:
    """A validated verdict list plus any citation repairs applied to it."""

    verdicts: list[Verdict]
    repairs: list[CitationRepair] = field(default_factory=list)


def parse_verdicts(
    data: Any, expected_ids, valid_turn_indices=None, *, repair_citations: bool = True
) -> list[Verdict]:
    """Validate a parsed verdict payload against the ``expected_ids`` claim set.

    ``data`` is already-parsed JSON (a list or ``{"verdicts": [...]}`` envelope);
    use :func:`loads_json` first if you have model text. ``expected_ids`` is the
    ordered/iterable set of input claim ids. The returned verdicts are ordered to
    match ``expected_ids`` so the output is deterministic regardless of the order
    the model emitted them in.

    Enforces the completeness contract — every DISTINCT expected id appears
    EXACTLY ONCE. Each failure raises its own subclass so callers can act on the
    specific diagnosis:
      * a verdict count that does not equal the expected count ->
        :class:`VerdictCountError` (the truncation/padding signature: a response
        cut off mid-list yields fewer verdicts than there are claims);
      * a claim_id not in ``expected_ids`` -> :class:`UnknownClaimIdError`;
      * a repeated claim_id in the output -> :class:`DuplicateClaimIdError`;
      * a distinct expected id with no verdict -> :class:`MissingClaimIdError`.

    Note on completeness vs. count: the count check compares against the number
    of expected slots ``len(expected)``, so a short (truncated) list trips
    :class:`VerdictCountError` first. The distinct-id map then catches unknown
    and duplicate ids; the final missing check catches a count-correct response
    that still fails to cover a distinct expected id (reachable when
    ``expected_ids`` itself carries a duplicate, i.e. distinct-count < slot-count).
    ``expected_ids`` is normally duplicate-free (claim ids are unique per
    :func:`load_claims`).

    Citation handling (when ``valid_turn_indices`` is supplied):
      * ``repair_citations=True`` (the default, and what the judge + CLI use):
        an unsupported citation is repaired PER CLAIM via
        :func:`repair_turn_citations` — the bad index is dropped and, if that
        leaves an uncited coverage claim, that ONE claim is downgraded to
        not-covered. One miscited claim can no longer discard the other 62.
      * ``repair_citations=False``: the strict :func:`verify_turn_citations`
        raises :class:`InvalidTurnCitationError` for the whole set (kept for
        callers that want the all-or-nothing contract).
    Use :func:`parse_verdicts_detailed` to see WHICH claims were repaired; this
    function returns the verdicts alone.
    """
    return parse_verdicts_detailed(
        data, expected_ids, valid_turn_indices, repair_citations=repair_citations
    ).verdicts


def parse_verdicts_detailed(
    data: Any, expected_ids, valid_turn_indices=None, *, repair_citations: bool = True
) -> ParsedVerdicts:
    """:func:`parse_verdicts`, additionally returning the citation repairs applied.

    The repairs are what the judge stamps into its verdict object (and hence the
    session's coverage sidecar), so a repaired coverage number is auditable
    instead of silently different from what the model returned.
    """
    expected = list(expected_ids)
    # Distinct expected ids, first-seen order preserved for deterministic output.
    distinct: list[str] = []
    seen_expected: set[str] = set()
    for cid in expected:
        if cid not in seen_expected:
            seen_expected.add(cid)
            distinct.append(cid)
    raw = _verdict_records(data)

    # Truncation/padding: the verdict count must equal the number of expected
    # slots. A short list (the truncation signature) trips here first.
    if len(raw) != len(expected):
        raise VerdictCountError(
            f"expected {len(expected)} verdicts (one per claim), got {len(raw)} "
            "— response is truncated or padded"
        )

    verdicts = [_coerce_verdict(item) for item in raw]
    # Delegate the unknown/duplicate/missing id defenses to the shared, pure
    # completeness check so both call sites use identical semantics.
    verify_claim_id_coverage([v.claim_id for v in verdicts], distinct)

    # Citation validation, when the caller supplies the transcript's index space.
    # Optional so the pure parser stays usable without a transcript, but
    # judge_coverage ALWAYS supplies it — an unvalidated citation is the one
    # defect class that yields a wrong number instead of an error.
    repairs: list[CitationRepair] = []
    if valid_turn_indices is not None:
        if repair_citations:
            verdicts, repairs = repair_turn_citations(verdicts, valid_turn_indices)
        else:
            verify_turn_citations(verdicts, valid_turn_indices)

    by_id = {v.claim_id: v for v in verdicts}
    return ParsedVerdicts(
        verdicts=[by_id[cid] for cid in distinct], repairs=repairs
    )


def parse_verdicts_text(
    text: str, expected_ids, valid_turn_indices=None, *, repair_citations: bool = True
) -> list[Verdict]:
    """Fence-strip + JSON-parse ``text``, then validate against ``expected_ids``.

    The full transport-defense pipeline as one call: markdown-fence stripping,
    JSON parsing (raising the module's :class:`VerdictJSONError`, never a raw
    ``json.JSONDecodeError``), the completeness/id validation of
    :func:`parse_verdicts`, and — when ``valid_turn_indices`` is supplied — the
    per-claim citation repair of :func:`repair_turn_citations` (or, with
    ``repair_citations=False``, the strict :func:`verify_turn_citations`).
    """
    return parse_verdicts(
        loads_json(text),
        expected_ids,
        valid_turn_indices,
        repair_citations=repair_citations,
    )


def parse_verdicts_text_detailed(
    text: str, expected_ids, valid_turn_indices=None, *, repair_citations: bool = True
) -> ParsedVerdicts:
    """:func:`parse_verdicts_text`, returning the citation repairs alongside."""
    return parse_verdicts_detailed(
        loads_json(text),
        expected_ids,
        valid_turn_indices,
        repair_citations=repair_citations,
    )


# --------------------------------------------------------------------------- #
# Judge-prompt versioning + hashing.
#
# The judge prompt is a versioned deliverable: every verdict carries a
# ``judge_prompt_hash`` so a coverage result is always traceable to the exact
# prompt that produced it. There is ONE public hash function, :func:`prompt_hash`,
# reused for BOTH the v1 provenance check and the authored v2 prompt — no second,
# copy-pasted hash routine.
#
# The scheme is the one the eval-set records for v1 (judge-prompt-v1.md, line 4):
#   "Prompt hash (sha256[:16] of system + user template)".
# i.e. sha256 of the prompt text (UTF-8), truncated to the first 16 hex chars
# (a 64-bit digest). :func:`prompt_hash` implements exactly that scheme and is
# PURE: no I/O, no network, no clock, no global mutable state — the same input
# always yields the same output.
# --------------------------------------------------------------------------- #

# Provenance of the v1 judge prompt, as recorded in the eval-set folder
# (judge-prompt-v1.md and every *.coverage.json ``judge_prompt_hash``). v1 is a
# fixed INPUT to VERIFY, never a target to reproduce or tune toward.
V1_PROMPT_HASH = "632b73a34b1a22b1"

# Length of the truncated hex digest (64-bit); factored out so the v1 and v2
# hashes are produced identically.
_PROMPT_HASH_HEX_LEN = 16


def prompt_hash(text: str) -> str:
    """Return the repo-scheme prompt hash: ``sha256(text)`` truncated to 16 hex.

    The single, public, PURE hash function for judge prompts. Deterministic:
    the same ``text`` always yields the same digest; it performs no I/O and
    depends on no external state. Used for BOTH the v1 provenance verification
    (hashing the copied v1 fixture bytes) and the authored v2 prompt — one
    implementation, reused, so v1 and v2 hashes are directly comparable.

    ``text`` is encoded as UTF-8 before hashing (a fixed, deterministic
    encoding), so the hash of a ``str`` is stable across processes/platforms.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:_PROMPT_HASH_HEX_LEN]


# --------------------------------------------------------------------------- #
# The v2 judge prompt — the AUTHORED deliverable of this sprint.
#
# v2 keeps v1's strictness against topic adjacency AND adds an explicit
# CONTENT-MATCH requirement: "covered" means the tutor's explanation is
# consistent with the claim's SPECIFIC assertions (same substance) — not merely
# the same shape, topic, or shared keywords. An explanation that substitutes
# different specifics (different list members, a different mechanism, or
# contradicting details) is NOT covered even if it is fluent and topically
# adjacent. This closes the proven v1 failure mode (crediting "fluent wrongness",
# e.g. the c30 case where the tutor taught the doc's sub-example triad as the
# headline three-way split and v1 credited it).
#
# GENERALIZED after independent review (2026-08-03). The first cut of these rules
# was written around the ONE case it had to fix: it carried c30's structural
# fingerprint ("an A/B/C three-way split, PLUS a list of named patterns"), a rule
# that existed only for c30's second enumeration ("if the claim bundles two
# separate enumerations..."), a clause aimed at the fixture's recap turns ("even
# if the recap uses the claim's exact words"), and a verbal photograph of c30's
# exact failure ("teaching a sub-example's inner list AS IF it were the headline
# split"). Those are eval-set specifics, not principles, and a prompt tuned to its
# own answer key proves nothing about unseen documents — so they were removed and
# each rule restated at the general level. The motivating case is recorded HERE,
# in a comment the model never sees, rather than in the prompt itself.
#
# There is NO byte-reproduction target for v2: it is authored here and hashed
# with the SAME :func:`prompt_hash` function; its new hash is what every verdict
# records once the live judge is wired (a later sprint).
# --------------------------------------------------------------------------- #

# A versioned tag embedded in the prompt so the authored version is self-identifying.
JUDGE_PROMPT_V2_VERSION = "v2"

JUDGE_PROMPT_V2_SYSTEM = """\
You are a STRICT coverage judge for a voice study-tutor experiment (judge prompt \
version v2). You are given a list of factual CLAIMS extracted from a source \
document, and the full transcript of one tutoring session about that document. \
Your job: for EACH claim, decide whether the TUTOR (the assistant role) actually \
EXPLAINED that claim's specific assertion to the student, with real \
comprehensiveness.

Rules for a verdict of "covered": true
- CONTENT MATCH (the v2 requirement): the tutor's explanation must be CONSISTENT \
  WITH THE CLAIM'S SPECIFIC ASSERTIONS -- the same substance -- not merely the \
  same shape, the same topic, or shared keywords. Matching a claim's topic, \
  phrasing shape, or vocabulary is NOT sufficient on its own.
- An explanation that SUBSTITUTES DIFFERENT SPECIFICS is NOT covered, even if it \
  is fluent, confident, and topically adjacent. In particular, if the tutor \
  gives different list members, a different mechanism, a different cause, \
  different numbers, or otherwise contradicts the claim's specific details, mark \
  it NOT covered -- fluent wrongness is not coverage.
- ENUMERATIONS AND MULTI-PART CLAIMS (conjunctive) -- decompose the claim into \
  ALL its distinct named parts and require the tutor to actually convey EVERY \
  ONE of them. A multi-part claim is a CONJUNCTION: it is covered only if ALL its \
  parts are delivered. Delivering a different set, the right NUMBER of wrong \
  members, or only some of the parts while omitting the rest, is NOT covered.
- NAMING/LISTING IS NOT EXPLAINING -- naming a term, or reciting a list of \
  labels, without conveying what each part MEANS or asserts, is NOT coverage of a \
  claim whose substance is those parts' content. A bare name-drop is a passing \
  mention, not coverage.
- LEVEL / SCOPE MATCH -- the tutor must convey the claim's assertion at the \
  claim's OWN level, with the claim's OWN specifics. Explaining a lower-level \
  sub-item, inner example, or detail does not cover a claim about a higher-level \
  structure, and vice versa.
- The tutor must convey the claim's ACTUAL ASSERTION -- its specific substance -- \
  not merely mention the topic, name a term, or say something topic-adjacent.
- A passing mention, a one-word reference, a heading read aloud, or a remark that \
  is merely about the same general subject is NOT coverage.
- If the tutor discussed the general area from outside/general knowledge without \
  conveying THIS claim's specific point, that is NOT coverage.
- Coverage is about what the TUTOR explained. A student asserting something, or a \
  claim being merely implied, does not count. The explanation must be the tutor's.
- Every "covered": true MUST cite, in "turns", the COMPLETE list of transcript \
  turn indices (the [N] labels) -- assistant turns -- that together constitute \
  the explanation. If you cannot point to a specific tutor turn that explains the \
  claim, it is NOT covered.
- No citable turn => "covered": false and "turns": [].

Be strict. When in doubt, mark not covered. Topic adjacency is the most common \
trap -- reject it. Fluent-but-wrong specifics (right shape, wrong substance) are \
the second trap -- reject them too.

PER-CLAIM PROCEDURE (apply to EVERY claim before deciding): \
(1) Identify the claim's CENTRAL ASSERTION -- the main point it makes -- and, \
separately, any ENUMERATION it commits to (a set of members the claim explicitly \
names). Do NOT treat a rhetorical \
restatement, an illustrative metaphor, an example aside, or a paraphrase of the \
same single point as an extra required part -- those are ONE assertion, not many. \
(2) A single-point claim is covered when the tutor conveys that central \
assertion's substance (the exact wording need not match). \
(3) A claim that commits to an ENUMERATION is covered ONLY IF the tutor delivers \
the substance of EACH named member of that enumeration -- not a different set, \
not the right count of wrong members, not just some members while omitting the \
rest, and not a mere name-drop. If any named enumeration member is missing, \
substituted, or only name-dropped, mark "covered": false, and do not let a \
strong match on the topic or on one member carry the members that are absent. \
Apply this enumeration strictness ONLY to genuine named enumerations, so a \
single-point claim is not failed for lacking parts it never enumerated.

OUTPUT: a single strict JSON object, no prose, no markdown fences. For EACH \
claim include a brief "reason" that names the claim's load-bearing parts and \
states, for each, whether the tutor delivered it -- then the boolean must follow \
that reasoning (covered ONLY if every part is delivered):
{"verdicts": [{"claim_id": "<id>", "reason": "<parts + which were delivered>", \
"covered": true|false, "turns": [<int>, ...]}, ...]}
Include EVERY claim id exactly once, in the order given."""

JUDGE_PROMPT_V2_USER_TEMPLATE = """\
CLAIMS (id -- claim text):
{claims_block}

TRANSCRIPT (each line is "[index] ROLE: content"; only assistant turns are the tutor):
{transcript_block}

Produce the strict JSON coverage object now. Every claim id exactly once; every \
"covered": true must (a) cite all constituent assistant turn indices AND (b) match \
the claim's specific substance, not just its topic, shape, or keywords."""

# The full v2 prompt text that :func:`prompt_hash` hashes: system + user template
# joined with a blank line, mirroring the "system + user template" scheme the
# eval-set records for v1. Fixed, deterministic composition (no runtime data).
JUDGE_PROMPT_V2 = JUDGE_PROMPT_V2_SYSTEM + "\n\n" + JUDGE_PROMPT_V2_USER_TEMPLATE

# The authored v2 hash, computed with the SAME public hash function. Stable and
# deterministic; there is no target value it is tuned to hit.
JUDGE_PROMPT_V2_HASH = prompt_hash(JUDGE_PROMPT_V2)


# --------------------------------------------------------------------------- #
# Sprint 1: the single-invocation judge + verdict assembly.
#
# One model call per invocation (Haiku, temperature 0) that judges the claim
# list against an indexed transcript using the v2 prompt above. The output is a
# complete verdict set (every claim id exactly once) plus provenance metadata,
# assembled ON the Sprint-0 transport-defense core: markdown-fence stripping and
# strict verdict parsing are delegated to :func:`parse_verdicts_text`, so the
# truncation/count-mismatch and unknown/duplicate/missing id defenses use exactly
# one implementation. A bounded retry re-issues the call on any malformed output
# (a caught :class:`VerdictParseError`); after the bound is exhausted the last
# typed parse error is re-raised rather than returning a partial verdict.
#
# The Anthropic client is constructed LAZILY inside :func:`judge_coverage`
# (mirroring claims.extract_claims / bot.generate_session_summary) so
# ``import coverage_judge`` still reads no API key and performs no network I/O.
# The ``anthropic`` import is likewise function-local, never at module scope.
# --------------------------------------------------------------------------- #

# The judge model + sampling. Haiku (5x cheaper than the live Sonnet), temperature
# 0 for a deterministic-as-possible verdict. The id matches the repo's other Haiku
# calls (bot.py / documents.py); overridable via JudgeConfig for tests/wiring.
JUDGE_MODEL = "claude-haiku-4-5-20251001"
JUDGE_TEMPERATURE = 0
# Each verdict carries a brief per-claim "reason" (the decomposition the v2 prompt
# asks for) plus the boolean + turns, so a 63/71-claim doc's strict-JSON verdict
# list runs to several thousand tokens. Generous headroom so a full verdict set is
# never truncated at the token cap (that would masquerade as a VerdictCountError
# and burn retries).
JUDGE_MAX_TOKENS = 16_000
# Total model calls per invocation: the initial call + a bounded number of retries
# on malformed output. Small and finite so a persistently-malformed model can
# never loop unbounded (goal: "a bounded retry on malformed output").
MAX_JUDGE_ATTEMPTS = 2


class JudgeError(Exception):
    """Raised when the judge invocation cannot produce a valid verdict set.

    Distinct from the input-validation (:class:`CoverageInputError`) and
    transport-parse (:class:`VerdictParseError`) hierarchies: this wraps a
    live-call failure the retry loop could not recover from. The originating
    parse error is chained via ``__cause__`` so the specific diagnosis (JSON,
    count/truncation, id-mismatch) is preserved.
    """


@dataclass(frozen=True)
class JudgeConfig:
    """Configuration for one judge invocation: model + judge-prompt version.

    Defaults to the module's Haiku model and the authored v2 prompt. ``model`` is
    the id stamped into the verdict metadata AND sent to the API. The prompt
    fields default to the v2 constants; ``judge_prompt_hash`` defaults to the v2
    hash so every verdict is traceable to the exact prompt that produced it.
    """

    model: str = JUDGE_MODEL
    temperature: float = JUDGE_TEMPERATURE
    max_tokens: int = JUDGE_MAX_TOKENS
    max_attempts: int = MAX_JUDGE_ATTEMPTS
    prompt_version: str = JUDGE_PROMPT_V2_VERSION
    system_prompt: str = JUDGE_PROMPT_V2_SYSTEM
    user_template: str = JUDGE_PROMPT_V2_USER_TEMPLATE
    judge_prompt_hash: str = JUDGE_PROMPT_V2_HASH


def _render_claims_block(claims: list[Claim]) -> str:
    """Render the claim list as ``<id> -- <text>`` lines for the user prompt."""
    return "\n".join(f"{c.id} -- {c.text}" for c in claims)


def _render_transcript_block(turns: list[Turn]) -> str:
    """Render the indexed transcript as ``[index] ROLE: content`` lines.

    The ``[index]`` labels are exactly the integers a verdict's ``turns`` list
    references, so the model can cite constituent turns unambiguously.
    """
    return "\n".join(f"[{t.index}] {t.role.upper()}: {t.content}" for t in turns)


def build_judge_messages(
    claims: list[Claim], turns: list[Turn], config: JudgeConfig
) -> list[dict]:
    """Compose the single user message for the judge call from config + inputs.

    Pure and model-independent (no client, no I/O): fills the config's user
    template with the rendered claim + transcript blocks. Factored out so a test
    can assert the payload carries the v2 prompt text without issuing a call.
    """
    user = config.user_template.format(
        claims_block=_render_claims_block(claims),
        transcript_block=_render_transcript_block(turns),
    )
    return [{"role": "user", "content": user}]


def _extract_text(response: Any) -> str:
    """Pull the assistant text out of an Anthropic messages response.

    Concatenates the text of every text content block (mirroring the repo's
    ``resp.content[0].text`` usage, but tolerant of multiple blocks). Raises
    :class:`VerdictJSONError` if there is no text at all, so an empty/blank
    response is treated as malformed output the retry loop can act on.
    """
    parts: list[str] = []
    for block in getattr(response, "content", None) or []:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    joined = "".join(parts).strip()
    if not joined:
        raise VerdictJSONError("model response contained no text content")
    return joined


def _doc_id_from_claims(claims: Any) -> str | None:
    """Derive a document identity from a claim-map payload, when it carries one.

    The repo's ``.claims.json`` sidecars carry ``source_hash`` — a content hash of
    the source document — which is exactly the per-document identity the union
    guard needs (two different documents necessarily differ). Returns ``None`` for
    a bare claim list, which carries no document identity.
    """
    if isinstance(claims, dict):
        for key in ("doc_id", "doc_key", "source_hash"):
            value = claims.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def judge_coverage(
    claims: Any,
    transcript: Any,
    config: JudgeConfig | None = None,
    *,
    client: Any = None,
    doc_id: str | None = None,
) -> dict:
    """Judge coverage for one session: one model call, one complete verdict set.

    ``claims`` / ``transcript`` are raw inputs (sidecar/envelope or bare list) —
    they are normalized via :func:`load_claims` / :func:`load_transcript`, so
    input-validation failures surface as :class:`CoverageInputError` before any
    call fires. The model is invoked ONCE on the happy path (Haiku, temperature
    0) with the v2 prompt; its strict-JSON reply is run through the Sprint-0
    transport defenses (:func:`parse_verdicts_text`: fence stripping, JSON
    parsing, truncation/count check, unknown/duplicate/missing id checks). On any
    :class:`VerdictParseError` the call is retried up to ``config.max_attempts``
    total; if still malformed, the last parse error is re-raised (wrapped in
    :class:`JudgeError` only when it was a live-call failure).

    The Anthropic client is constructed LAZILY here (never at import time); pass
    ``client=`` to inject a mock in hermetic tests so no network call is made.

    Returns:
        A verdict object:
            {
              "verdicts": [
                {"claim_id", "covered", "turns", "reason"}, ... one per claim
              ],
              "judged_at": <ISO-8601 UTC timestamp>,
              "model": <config.model>,
              "judge_prompt_hash": <config.judge_prompt_hash>,
              "doc_id": <document identity, or None when the claim payload
                         carried none — see union_coverage's merge guard>,
            }
        with exactly one verdict per input claim id, ordered as the input claims.
        ``reason`` is the model's per-claim rationale (None when it omitted one).
    """
    cfg = config or JudgeConfig()
    claim_list = load_claims(claims)
    turn_list = load_transcript(transcript)
    expected_ids = claim_ids(claim_list)
    messages = build_judge_messages(claim_list, turn_list, cfg)
    # The transcript's real index space — every verdict citation is validated
    # against it, so a hallucinated turn cannot become a silent coverage credit.
    valid_turns = [t.index for t in turn_list]
    # Document identity for the cross-document merge guard (see union_coverage).
    resolved_doc_id = doc_id if doc_id is not None else _doc_id_from_claims(claims)

    if client is None:
        # Lazy, function-local import + construction: importing this module reads
        # no API key and performs no I/O (mirrors claims.extract_claims).
        import anthropic  # noqa: PLC0415 - intentional lazy import

        client = anthropic.Anthropic()

    last_parse_error: VerdictParseError | None = None
    for _attempt in range(cfg.max_attempts):
        response = client.messages.create(
            model=cfg.model,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            system=cfg.system_prompt,
            messages=messages,
        )
        # max_tokens truncation is DETERMINISTIC: the identical prompt at
        # temperature 0 truncates identically, so retrying only doubles the bill
        # for the same failure. Diagnose it explicitly and fail fast, before the
        # generic parse path can mistake it for ordinary bad JSON.
        if getattr(response, "stop_reason", None) == "max_tokens":
            raise VerdictTruncatedError(
                f"model reply hit the {cfg.max_tokens}-token output cap and was cut "
                "off mid-verdict (not retried: identical prompt at temperature 0 "
                "would truncate identically — raise max_tokens or judge fewer "
                "claims per call)"
            )
        try:
            text = _extract_text(response)
            parsed = parse_verdicts_text_detailed(text, expected_ids, valid_turns)
            # A wholesale citation failure (every coverage claim downgraded) is
            # NOT a coverage number — see MassCitationDowngradeError. Raised
            # inside the try so it retries within the bound like any other
            # malformed output, and, if the retry fails too, degrades to no
            # coverage rather than persisting a permanent zero.
            if is_mass_citation_downgrade(parsed.verdicts, parsed.repairs):
                raise MassCitationDowngradeError(
                    f"all {sum(1 for r in parsed.repairs if r.downgraded)} coverage "
                    "claim(s) were downgraded because none of their cited turns exist "
                    f"in the judged transcript ({len(valid_turns)} turns, indices "
                    f"{valid_turns[:1]}..{valid_turns[-1:]}) — the citations are in a "
                    "different index space, not a per-claim miscitation"
                )
        except VerdictParseError as e:
            # Malformed output (invalid JSON, truncated/short verdict list, an
            # unknown/duplicate/missing claim id, or an unsupported turn
            # citation). Remember it and retry within the bound.
            last_parse_error = e
            continue
        return _assemble_verdict(
            parsed.verdicts, cfg, doc_id=resolved_doc_id, repairs=parsed.repairs
        )

    # Bound exhausted with every attempt malformed: re-raise the last typed parse
    # error so the caller sees the specific diagnosis (never a partial verdict).
    assert last_parse_error is not None  # loop ran at least once (max_attempts>=1)
    raise last_parse_error


def _assemble_verdict(
    verdicts: list[Verdict],
    config: JudgeConfig,
    *,
    doc_id: str | None = None,
    repairs: list[CitationRepair] | None = None,
) -> dict:
    """Assemble the final verdict object (verdicts + provenance metadata).

    ``verdicts`` is the validated, expected-order list from
    :func:`parse_verdicts`. Stamps ``judged_at`` (UTC, ISO-8601),
    ``model``, and ``judge_prompt_hash`` (the v2 hash) so every result is
    traceable to the exact prompt + model that produced it.
    """
    # Local imports keep the module's top-level import closure stdlib-only per the
    # Sprint-0 isolation contract (no datetime dependency leaks at module scope).
    from datetime import datetime, timezone

    return {
        "verdicts": [v.to_dict() for v in verdicts],
        "judged_at": datetime.now(timezone.utc).isoformat(),
        "model": config.model,
        "judge_prompt_hash": config.judge_prompt_hash,
        # Document identity: claim ids are per-document sequentials (c1..cN) that
        # COLLIDE across documents, so union_coverage needs this to refuse a
        # cross-document merge. None when the claim payload carried no identity.
        "doc_id": doc_id,
        # Per-claim citation repairs applied to this set (normally empty). Kept
        # so a repaired coverage number is auditable rather than silent.
        "citation_repairs": [r.to_dict() for r in (repairs or [])],
    }


# --------------------------------------------------------------------------- #
# Sprint 2: union_coverage — the pure cross-session merge.
#
# A study document is judged across MULTIPLE sessions; a claim is "covered" if
# ANY session covered it (union by claim id). Percentage is DERIVED AT READ TIME
# per the design doc — the union stores only the covered id set and the universe
# of judged ids, and computes the percentage on demand — so it never drifts from
# a stale stored count.
#
# This function is PURE: no file I/O, no model/LLM call, no network, no clock,
# no global mutable state. It accepts the verdict SETS this module's own judge
# produces (the {verdicts, judged_at, model, judge_prompt_hash} object from
# :func:`judge_coverage`) and, for convenience, a bare per-claim verdict list or
# a {"verdicts": [...]} envelope.
# --------------------------------------------------------------------------- #

# Fixed rounding convention for the derived percentage: 1 decimal place. Factored
# out so the single documented convention is applied in exactly one place — and
# PUBLIC because coverage_store derives its own display percentage (covered over
# the claim MAP's size, not over the judged universe) and must round identically;
# two modules rounding a user-facing number by different conventions is exactly
# the kind of drift a shared constant exists to prevent.
COVERAGE_PERCENTAGE_DECIMALS = 1


def _verdict_list_of(verdict_set: Any) -> list:
    """Return the raw per-claim verdict list from one merged input item.

    Accepts, uniformly, any of the shapes a caller might hand :func:`union_coverage`:
      * the full verdict OBJECT this module's judge returns
        (``{"verdicts": [...], "judged_at", "model", "judge_prompt_hash"}``);
      * a bare list of per-claim verdict records (``[{claim_id, covered, turns}, ...]``);
      * a list of typed :class:`Verdict` instances.
    Raises :class:`CoverageInputError` on any other shape so a malformed session
    result is a clear, typed failure rather than a silent empty merge.
    """
    if isinstance(verdict_set, dict):
        raw = verdict_set.get("verdicts")
        if not isinstance(raw, list):
            raise CoverageInputError(
                'verdict set must have a "verdicts" list'
            )
        return raw
    if isinstance(verdict_set, list):
        return verdict_set
    raise CoverageInputError(
        "verdict set must be a verdict object or a list of verdicts, "
        f"got {type(verdict_set).__name__}"
    )


def _verdict_fields(item: Any) -> tuple[str, bool]:
    """Return ``(claim_id, covered)`` from one per-claim verdict (dict or typed).

    Only the fields :func:`union_coverage` needs; ``turns`` is not consulted for
    the union. Raises :class:`CoverageInputError` on a missing/blank claim id or a
    non-bool covered flag.
    """
    if isinstance(item, Verdict):
        return item.claim_id, item.covered
    if not isinstance(item, dict):
        raise CoverageInputError(f"verdict is not an object: {item!r}")
    cid = item.get("claim_id")
    if not isinstance(cid, str) or not cid.strip():
        raise CoverageInputError(f"verdict has a missing/blank claim_id: {cid!r}")
    covered = item.get("covered")
    if not isinstance(covered, bool):
        raise CoverageInputError(
            f"verdict {cid!r} covered must be a bool, got {covered!r}"
        )
    return cid.strip(), covered


def union_coverage(verdict_sets: Any, *, allow_unidentified: bool = False) -> dict:
    """Merge coverage verdicts across sessions by UNION of covered claim ids.

    ``verdict_sets`` is an iterable of per-session verdict sets, each in any shape
    :func:`_verdict_list_of` accepts (the full judge object, a bare verdict list,
    or a list of typed :class:`Verdict`). A claim id is covered in the union if it
    is covered in ANY session, even when other sessions mark it not-covered.

    Returns a dict::

        {"covered_ids": [<claim id>, ...], "judged_ids": [<claim id>, ...],
         "percentage": <float>}

    where ``covered_ids`` is the sorted, de-duplicated union of covered ids,
    ``judged_ids`` is the sorted, de-duplicated set of every claim id judged in
    any set (covered or not — the percentage's denominator, returned so a caller
    merging ONE SET AT A TIME can rebuild the same universe rather than
    re-deriving it from verdict internals), and ``percentage`` is DERIVED AT READ
    TIME as::

        100 * (# distinct covered ids) / (# distinct judged ids across all sets)

    rounded to a single fixed convention (:data:`COVERAGE_PERCENTAGE_DECIMALS`
    decimal places). The denominator counts EVERY distinct claim id judged in any
    session (covered or not), so a not-covered claim still contributes to the
    universe. Empty input (no sets, or sets with empty verdict lists) yields
    ``covered_ids == []`` and ``percentage == 0.0`` with no divide-by-zero.

    Document-identity guard: claim ids are per-document sequentials that collide
    across documents, so merging sessions judged against DIFFERENT documents
    produces false coverage. Refused cases: more than one distinct declared
    ``doc_id``; a mix of declaring and non-declaring sets; and MORE THAN ONE set
    where none declares an identity. That last case is exempted by
    ``allow_unidentified=True``, which asserts the caller knows the sets share a
    document (used for pure merge arithmetic on bare verdict lists). A single set
    and empty input never need the opt-in.

    Pure: performs no file I/O, no model/LLM call, no network access, and depends
    on no external mutable state.
    """
    covered: set[str] = set()
    judged: set[str] = set()
    # Document-identity guard. Claim ids are per-document sequentials (c1..c63,
    # c1..c71) sharing ONE namespace, so merging sessions judged against different
    # documents makes document A's c15 absorb document B's c15 — a false-positive
    # coverage credit in a user-facing number, with no error. Refuse it.
    declared: set[str] = set()
    undeclared = 0
    for verdict_set in verdict_sets:
        set_doc_id = verdict_set.get("doc_id") if isinstance(verdict_set, dict) else None
        if isinstance(set_doc_id, str) and set_doc_id.strip():
            declared.add(set_doc_id.strip())
        else:
            undeclared += 1
        for item in _verdict_list_of(verdict_set):
            cid, is_covered = _verdict_fields(item)
            judged.add(cid)
            if is_covered:
                covered.add(cid)

    if len(declared) > 1:
        raise CoverageInputError(
            "refusing to merge verdict sets from DIFFERENT documents "
            f"(doc_ids: {sorted(declared)}) — claim ids are per-document and would "
            "collide, producing a false coverage number. Union one document at a time."
        )
    if declared and undeclared:
        # Some sets identify their document and some do not: we cannot PROVE they
        # are the same document, and guessing is exactly the silent-wrongness risk.
        raise CoverageInputError(
            f"refusing to merge {undeclared} unlabelled verdict set(s) with sets "
            f"declaring doc_id {sorted(declared)} — cannot verify they are the same "
            "document. Stamp doc_id on every set (judge_coverage does this)."
        )
    if not declared and undeclared > 1 and not allow_unidentified:
        # Every set is unlabelled: nothing proves they came from the SAME document,
        # so this is the guard's last bypass — strip the metadata and the merge
        # silently succeeds. A single set is exempt (one set cannot be a CROSS-
        # document merge), as is empty input. Callers doing pure merge arithmetic
        # on known-same-document data opt in explicitly rather than by omission.
        raise CoverageInputError(
            f"refusing to merge {undeclared} verdict sets that declare no document "
            "identity — nothing proves they are the same document, and claim ids "
            "collide across documents. Use verdict objects from judge_coverage "
            "(which stamp doc_id), or pass allow_unidentified=True to assert they "
            "are the same document."
        )

    total = len(judged)
    if total == 0:
        percentage = 0.0
    else:
        percentage = round(100.0 * len(covered) / total, COVERAGE_PERCENTAGE_DECIMALS)
    return {
        "covered_ids": sorted(covered),
        "judged_ids": sorted(judged),
        "percentage": percentage,
    }


# --------------------------------------------------------------------------- #
# Sprint 2: the standalone CLI.
#
#   python -m coverage_judge --claims <file> --transcript <file> --out <file>
#                            [--cost-out <file>] [--model <id>]
#                            [--prompt-version <ver>]
#
# Reads a claim list + indexed transcript from disk, runs ONE judge invocation
# (the real Haiku call in production; a mocked client injected in hermetic tests),
# and writes the verdict object to ``--out`` as JSON. ``--cost-out`` writes a
# small cost-accounting JSON (calls + model, plus token/cost fields when the SDK
# reports usage). No ledger writes are wired into any app path — the ``--cost-out``
# JSON is the entire cost surface, per the sprint constraint.
# --------------------------------------------------------------------------- #

# The known judge-prompt versions the CLI accepts, each mapping to its authored
# (system, user_template, hash). Sprint 2 ships v2 (the authored deliverable).
# An unknown --prompt-version is REJECTED, never blindly echoed into metadata.
_KNOWN_PROMPT_VERSIONS = {
    JUDGE_PROMPT_V2_VERSION: (
        JUDGE_PROMPT_V2_SYSTEM,
        JUDGE_PROMPT_V2_USER_TEMPLATE,
        JUDGE_PROMPT_V2_HASH,
    ),
}


class CLIError(Exception):
    """Raised for a user-facing CLI failure (bad path, unknown version, etc.).

    Caught by :func:`main`, which prints the message to stderr and returns a
    non-zero exit code, so an operator sees a clear message rather than a raw
    traceback and no bogus ``--out`` file is written.
    """


def _config_for_version(prompt_version: str, model: str) -> JudgeConfig:
    """Build a :class:`JudgeConfig` for a KNOWN prompt version, or reject.

    Raises :class:`CLIError` if ``prompt_version`` is not a known module prompt,
    so an unsupported version fails loudly instead of being echoed into a
    verdict's metadata.
    """
    entry = _KNOWN_PROMPT_VERSIONS.get(prompt_version)
    if entry is None:
        known = ", ".join(sorted(_KNOWN_PROMPT_VERSIONS))
        raise CLIError(
            f"unknown --prompt-version {prompt_version!r} (known: {known})"
        )
    system_prompt, user_template, judge_prompt_hash = entry
    return JudgeConfig(
        model=model,
        prompt_version=prompt_version,
        system_prompt=system_prompt,
        user_template=user_template,
        judge_prompt_hash=judge_prompt_hash,
    )


def _read_json_file(path: str, label: str) -> Any:
    """Read + JSON-parse a file, raising :class:`CLIError` on any failure.

    ``label`` (e.g. "claims", "transcript") is woven into the message so an
    operator knows which input was bad. A missing file, an unreadable file, or
    invalid JSON all surface as one typed, user-facing error.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError as e:
        raise CLIError(f"{label} file not found: {path}") from e
    except OSError as e:
        raise CLIError(f"could not read {label} file {path}: {e}") from e
    except (json.JSONDecodeError, ValueError) as e:
        raise CLIError(f"{label} file {path} is not valid JSON: {e}") from e


def _write_json_file(path: str, obj: Any) -> None:
    """Serialize ``obj`` to ``path`` as pretty JSON (UTF-8), or raise CLIError."""
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
    except OSError as e:
        raise CLIError(f"could not write output file {path}: {e}") from e


def _assert_writable(path: str, label: str) -> None:
    """Raise :class:`CLIError` unless ``path`` looks writable, BEFORE any call.

    The judge call costs money; discovering an unwritable output path afterwards
    throws that spend away along with the verdict. Checked up front instead.
    """
    import os

    if not path:
        raise CLIError(f"{label} path is empty")
    # A directory where a file belongs: os.path.exists/os.access both pass, so
    # without this the (paid) call fires and only THEN does the write fail.
    if os.path.isdir(path):
        raise CLIError(f"{label} path is a directory, not a file: {path}")
    directory = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(directory):
        raise CLIError(f"{label} directory does not exist: {directory}")
    if not os.access(directory, os.W_OK):
        raise CLIError(f"{label} directory is not writable: {directory}")
    # A broken symlink: writing follows the link, so the TARGET's directory is
    # what must exist. (lexists && !exists is exactly "dangling symlink".)
    if os.path.lexists(path) and not os.path.exists(path):
        if not os.path.isdir(os.path.dirname(os.path.realpath(path))):
            raise CLIError(f"{label} path is a broken symlink: {path}")
    elif os.path.exists(path) and not os.access(path, os.W_OK):
        raise CLIError(f"{label} file exists and is not writable: {path}")


@dataclass
class UsageTally:
    """Mutable running tally of model calls + reported token usage.

    Shared by the CLI and the app wiring so there is ONE implementation of "how
    many calls did this judge invocation make, and what did they cost". Tokens
    ACCUMULATE across attempts (a retry is exactly when spend spikes, so keeping
    only the last response under-reports precisely when the number matters).

    The ``calls_reporting_*`` counters exist for the partial-measurement problem:
    if only some attempts report usage, the bare sum looks like a complete count,
    and a field never observed at all would otherwise be emitted as a confident
    ``0``. :meth:`is_complete` distinguishes a real measurement from an
    incomplete one, and :func:`_cost_record` OMITS a field no call reported.
    """

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    calls_reporting_input: int = 0
    calls_reporting_output: int = 0

    def record_response(self, response: Any) -> None:
        """Fold one model response's ``usage`` block into the tally."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        in_tokens = getattr(usage, "input_tokens", None)
        out_tokens = getattr(usage, "output_tokens", None)
        # bool is a subclass of int; exclude it as verdict-shape validation does,
        # so a stray True cannot count as 1 token.
        if isinstance(in_tokens, int) and not isinstance(in_tokens, bool):
            self.input_tokens += in_tokens
            self.calls_reporting_input += 1
        if isinstance(out_tokens, int) and not isinstance(out_tokens, bool):
            self.output_tokens += out_tokens
            self.calls_reporting_output += 1

    def is_complete(self) -> bool:
        """True iff EVERY call made reported both token counts."""
        return (
            self.calls > 0
            and self.calls_reporting_input == self.calls
            and self.calls_reporting_output == self.calls
        )


class CountingClient:
    """Wrap an Anthropic-shaped client, tallying calls + usage into a :class:`UsageTally`.

    Transparent at the one boundary that matters (``messages.create``), so the
    judge is unaware of it and no key material passes through. Used by the CLI
    and by the app wiring, which both need the ACTUAL call count — including the
    calls made by attempts that then failed.
    """

    def __init__(self, inner: Any, tally: UsageTally | None = None):
        self._inner = inner
        self.tally = tally if tally is not None else UsageTally()
        self.messages = _CountingMessages(self)

    @property
    def inner(self) -> Any:
        return self._inner


class _CountingMessages:
    def __init__(self, owner: CountingClient):
        self._owner = owner

    def create(self, **kwargs):
        tally = self._owner.tally
        tally.calls += 1
        response = self._owner.inner.messages.create(**kwargs)
        tally.record_response(response)
        return response


def _cost_record(
    tally: UsageTally,
    config: JudgeConfig,
    *,
    status: str = "ok",
    error: str | None = None,
) -> dict:
    """Assemble the ``--cost-out`` cost-accounting record from a :class:`UsageTally`.

    Always carries the model, the number of model calls issued, and the run's
    ``status`` — this record is written whether the run SUCCEEDED OR FAILED
    (a failed run that burned two attempts is exactly when spend spiked, and the
    previous write-on-success-only behaviour dropped it).

    Token fields are emitted ONLY when at least one call actually reported them,
    so an unobserved count is absent rather than a confident ``0``, and
    ``usage_complete`` states whether every call reported usage — a reader can
    tell an incomplete measurement from a real one. NO ledger write happens here
    (the JSON file is the whole cost surface for the CLI).
    """
    record: dict[str, Any] = {
        "model": config.model,
        "calls": tally.calls,
        "status": status,
    }
    if tally.calls_reporting_input:
        record["input_tokens"] = tally.input_tokens
    if tally.calls_reporting_output:
        record["output_tokens"] = tally.output_tokens
    record["usage_complete"] = tally.is_complete()
    if not record["usage_complete"]:
        # Name exactly how incomplete, so a partial sum is never read as a total.
        record["calls_reporting_input_tokens"] = tally.calls_reporting_input
        record["calls_reporting_output_tokens"] = tally.calls_reporting_output
    if error is not None:
        record["error"] = error
    return record


def run_cli(
    claims_path: str,
    transcript_path: str,
    out_path: str,
    *,
    model: str = JUDGE_MODEL,
    prompt_version: str = JUDGE_PROMPT_V2_VERSION,
    cost_out_path: str | None = None,
    client: Any = None,
) -> dict:
    """End-to-end CLI body: read inputs, judge once, write verdict (+ cost).

    Reads and JSON-parses the ``claims_path`` and ``transcript_path`` inputs
    (raising :class:`CLIError` on a missing/malformed file BEFORE any model call),
    resolves the prompt version to a known module prompt (rejecting an unknown
    one), runs ONE judge invocation via :func:`judge_coverage` (the model call is
    injected as ``client`` in tests, so no network in the hermetic suite), writes
    the verdict object to ``out_path``, and — when ``cost_out_path`` is given —
    writes a cost-accounting JSON. Returns the verdict object.
    """
    cfg = _config_for_version(prompt_version, model)
    claims_data = _read_json_file(claims_path, "claims")
    transcript_data = _read_json_file(transcript_path, "transcript")

    # Input validation (CoverageInputError) fires here, before the write, so a
    # bad claim list / transcript never yields a bogus --out file.
    try:
        load_claims(claims_data)
        load_transcript(transcript_data)
    except CoverageInputError as e:
        raise CLIError(str(e)) from e

    # Output paths are checked for writability BEFORE the (paid) judge call, so an
    # unwritable --out never discards a call that has already been billed.
    _assert_writable(out_path, "output")
    if cost_out_path is not None:
        _assert_writable(cost_out_path, "cost-out")

    judge_client = client
    if judge_client is None:
        import anthropic  # noqa: PLC0415 - intentional lazy import (no import-time key)

        judge_client = anthropic.Anthropic()

    # Wrap the injected/real client so we observe the exact call count and the
    # per-attempt token usage, without leaking that plumbing into judge_coverage.
    counting = CountingClient(judge_client)

    # The cost record is written in a `finally`: a run that burned attempts and
    # THEN failed is exactly when spend spiked, and writing only on success
    # dropped it — recording spend when it did not spike and losing it when it
    # did, the inverse of the intent.
    try:
        verdict = judge_coverage(
            claims_data, transcript_data, config=cfg, client=counting
        )
    except Exception as e:
        if cost_out_path is not None:
            try:
                _write_json_file(
                    cost_out_path,
                    _cost_record(
                        counting.tally,
                        cfg,
                        status="failed",
                        error=f"{type(e).__name__}: {e}",
                    ),
                )
            except CLIError:
                # A failed cost write must not MASK the judge failure that is the
                # real diagnosis; the original error still propagates.
                pass
        raise

    _write_json_file(out_path, verdict)
    if cost_out_path is not None:
        _write_json_file(cost_out_path, _cost_record(counting.tally, cfg))

    return verdict


def build_arg_parser():
    """Construct the module's argparse parser (factored out for testability)."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="coverage_judge",
        description="Judge document-claim coverage for one study session.",
    )
    parser.add_argument("--claims", required=True, help="path to a claim-list JSON file")
    parser.add_argument(
        "--transcript", required=True, help="path to an indexed-transcript JSON file"
    )
    parser.add_argument("--out", required=True, help="path to write the verdict JSON")
    parser.add_argument(
        "--cost-out",
        dest="cost_out",
        default=None,
        help="optional path to write a cost-accounting JSON",
    )
    parser.add_argument(
        "--model", default=JUDGE_MODEL, help="model id to stamp + invoke"
    )
    parser.add_argument(
        "--prompt-version",
        dest="prompt_version",
        default=JUDGE_PROMPT_V2_VERSION,
        help="judge-prompt version (known: %s)" % ", ".join(sorted(_KNOWN_PROMPT_VERSIONS)),
    )
    return parser


def main(argv: Any = None, *, client: Any = None) -> int:
    """CLI entrypoint: parse args, run the judge, return a process exit code.

    Returns 0 on success and 2 on ANY handled failure, printing a one-line
    message to stderr rather than a traceback. The handled set is deliberately
    broad — every failure a real run can hit:
      * :class:`CLIError` — bad input file, unknown prompt version, unwritable
        output path;
      * :class:`CoverageInputError` — malformed claim list / transcript;
      * :class:`VerdictParseError` — the model's reply could not be validated
        (invalid JSON, id mismatch, unsupported turn citation);
      * :class:`VerdictTruncatedError` — the reply hit the output token cap;
        caught by name because it is deliberately OUTSIDE the retryable family;
      * :class:`JudgeError` and any other ``Exception`` — notably the Anthropic
        SDK's errors (auth, rate limit, network), which are not ours to enumerate.
    ``KeyboardInterrupt``/``SystemExit`` are BaseExceptions and deliberately still
    propagate. ``client`` is injected by hermetic tests to mock the model.
    """
    import sys

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        run_cli(
            args.claims,
            args.transcript,
            args.out,
            model=args.model,
            prompt_version=args.prompt_version,
            cost_out_path=args.cost_out,
            client=client,
        )
    except (
        CLIError,
        CoverageInputError,
        VerdictParseError,
        VerdictTruncatedError,  # named explicitly: no longer a VerdictParseError
        JudgeError,
    ) as e:
        print(f"coverage_judge: error: {e}", file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001 - CLI boundary: never show a traceback
        # Anything else (Anthropic SDK auth/rate-limit/network errors, unexpected
        # faults). The type is named so the message stays diagnosable.
        print(f"coverage_judge: error: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess in tests
    import sys

    sys.exit(main())
