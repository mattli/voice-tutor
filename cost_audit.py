"""Pure, Pipecat-free cost-log auditing helpers.

Sprint 0 foundation: this module is the new home of the pricing constants the
cost logger uses, plus pure recomputation helpers that re-derive each cost-log
row's ``cost_*_usd`` fields from that row's own usage fields. It imports nothing
beyond the standard library (``json`` / ``pathlib``) so it can be imported and
tested with no Pipecat, no network, and no provider API keys — ``import bot`` /
``import app`` are unwinnable in the verifier env, so the audit must never reach
for them.

Constant ownership was RELOCATED here from ``bot.py`` (values byte-identical to
the pre-move working tree); ``bot.py`` re-imports every price name from this
module so the logger's behavior is unchanged. That keeps a single source of
truth for the rates the audit recomputes against.

The recomputation helpers are total and deterministic: they read only the
fields on the row passed in, do no I/O and no network, and return identical
outputs for identical inputs. They intentionally do NOT round — callers compare
a recomputed value against the row's stored (already-rounded) value within
``COST_TOLERANCE_USD`` rather than trying to reproduce the logger's rounding.
"""

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Pricing constants (relocated verbatim from bot.py).
#
# Prices last verified 2026-04-15 against official pricing pages and
# cross-checked with the 2026-04-14 session's provider dashboards.
# Sources: claude.com/pricing, deepgram.com/pricing, cartesia.ai/pricing.
# Values here are byte-identical to the pre-relocation bot.py definitions;
# bot.py now re-imports these names so the logger's math is unchanged.
# ---------------------------------------------------------------------------
PRICE_ANTHROPIC_INPUT_PER_MTOK = 3.00
PRICE_ANTHROPIC_OUTPUT_PER_MTOK = 15.00
PRICE_ANTHROPIC_CACHE_WRITE_PER_MTOK = 3.75
PRICE_ANTHROPIC_CACHE_READ_PER_MTOK = 0.30
# Haiku 4.5 powers the post-session summary + analysis calls and the
# study-artifact generation call.
PRICE_ANTHROPIC_HAIKU_INPUT_PER_MTOK = 1.00
PRICE_ANTHROPIC_HAIKU_OUTPUT_PER_MTOK = 5.00
PRICE_DEEPGRAM_NOVA3_PER_MIN = 0.0077
# Cartesia bills 1 credit per character submitted to the TTS WebSocket;
# $5 / 100_000 credits on Pro plan = $0.00005 per character.
PRICE_CARTESIA_PER_CHAR = 5.00 / 100_000

# Default location of the append-only JSONL cost ledger the CLI audits. Read at
# CALL time (not bound into a local at import) so a test can monkeypatch this
# attribute to a per-test tmp ledger — same pattern as sessions.py /
# grounding.py / documents.py.
COST_LOG_JSONL_PATH = (
    Path.home() / "second-brain" / "products" / "voice-tutor" / "validation" / "cost-log.jsonl"
)

# Absolute tolerance (in USD) for comparing a recomputed cost against the row's
# stored value. Stored costs are ``round(x, 4)`` and ``stt_minutes_billed`` is
# ``round(x, 2)``; recomputing STT from the already-rounded minutes reproduces
# the stored value closely, but summed components accumulate up to ~a few units
# in the 4th decimal place. 5e-5 is loose enough to absorb that stored rounding
# without ever flagging a genuinely-correct row as a mismatch, while staying far
# tighter than any real math error we'd want to catch. A single module-level
# constant so helpers, tests, and the later CLI all share one epsilon.
COST_TOLERANCE_USD = 5e-5

# One MTok = one million tokens; per-MTok rates divide token counts by this.
_TOKENS_PER_MTOK = 1_000_000


def _num(value) -> float:
    """Coerce a possibly-missing usage field to a float, treating None as 0.

    Pure: no I/O. A missing field (legacy rows) or an explicit null counts as
    zero usage of that class, which is the same thing the logger would have
    recorded had the field existed.
    """
    if value is None:
        return 0.0
    return float(value)


def recompute_llm_cost(row: dict) -> float:
    """Recompute a session row's live-LLM (Sonnet) cost from its token fields.

    Uses all four token classes at their four distinct per-MTok rates:
    ``llm_uncached_input_tokens`` × INPUT, ``llm_cache_read_tokens`` ×
    CACHE_READ, ``llm_cache_write_tokens`` × CACHE_WRITE, ``llm_output_tokens``
    × OUTPUT. Pure and deterministic.
    """
    return (
        _num(row.get("llm_uncached_input_tokens")) / _TOKENS_PER_MTOK * PRICE_ANTHROPIC_INPUT_PER_MTOK
        + _num(row.get("llm_cache_read_tokens")) / _TOKENS_PER_MTOK * PRICE_ANTHROPIC_CACHE_READ_PER_MTOK
        + _num(row.get("llm_cache_write_tokens")) / _TOKENS_PER_MTOK * PRICE_ANTHROPIC_CACHE_WRITE_PER_MTOK
        + _num(row.get("llm_output_tokens")) / _TOKENS_PER_MTOK * PRICE_ANTHROPIC_OUTPUT_PER_MTOK
    )


def recompute_stt_cost(row: dict) -> float:
    """Recompute a session row's STT cost from its stored billed minutes.

    Uses the row's already-rounded ``stt_minutes_billed`` field (the field the
    goal names) × PRICE_DEEPGRAM_NOVA3_PER_MIN — NOT a re-derivation from
    ``session_duration_sec`` — so a correct row's recomputed STT matches the
    stored value within tolerance. Pure and deterministic.
    """
    return _num(row.get("stt_minutes_billed")) * PRICE_DEEPGRAM_NOVA3_PER_MIN


def recompute_tts_cost(row: dict) -> float:
    """Recompute a session row's TTS cost from its ``tts_chars`` field.

    ``tts_chars`` × PRICE_CARTESIA_PER_CHAR. Pure and deterministic.
    """
    return _num(row.get("tts_chars")) * PRICE_CARTESIA_PER_CHAR


def recompute_post_session_cost(row: dict) -> float:
    """Recompute a session row's post-session (Haiku) cost from its token fields.

    ``post_session_input_tokens`` × HAIKU_INPUT + ``post_session_output_tokens``
    × HAIKU_OUTPUT. Pure and deterministic. A legacy row without these fields
    yields 0.0.
    """
    return (
        _num(row.get("post_session_input_tokens")) / _TOKENS_PER_MTOK * PRICE_ANTHROPIC_HAIKU_INPUT_PER_MTOK
        + _num(row.get("post_session_output_tokens")) / _TOKENS_PER_MTOK * PRICE_ANTHROPIC_HAIKU_OUTPUT_PER_MTOK
    )


def recompute_session_costs(row: dict) -> dict:
    """Recompute every cost component + total for a session row from its usage.

    Returns a mapping with keys ``cost_llm_usd``, ``cost_stt_usd``,
    ``cost_tts_usd``, ``cost_post_session_usd``, and ``cost_total_usd`` (the sum
    of the four components). Values are the raw un-rounded recomputations;
    callers compare them against the row's stored values within
    ``COST_TOLERANCE_USD``. Pure and deterministic.
    """
    llm = recompute_llm_cost(row)
    stt = recompute_stt_cost(row)
    tts = recompute_tts_cost(row)
    post = recompute_post_session_cost(row)
    return {
        "cost_llm_usd": llm,
        "cost_stt_usd": stt,
        "cost_tts_usd": tts,
        "cost_post_session_usd": post,
        "cost_total_usd": llm + stt + tts + post,
    }


def recompute_artifact_cost(row: dict) -> float:
    """Recompute an artifact row's ``cost_usd`` from its own token fields.

    Artifact generation runs on Haiku, so this uses the HAIKU per-MTok rates
    (``input_tokens`` × HAIKU_INPUT + ``output_tokens`` × HAIKU_OUTPUT) — NOT
    the Sonnet input/output rates. Pure and deterministic.
    """
    return (
        _num(row.get("input_tokens")) / _TOKENS_PER_MTOK * PRICE_ANTHROPIC_HAIKU_INPUT_PER_MTOK
        + _num(row.get("output_tokens")) / _TOKENS_PER_MTOK * PRICE_ANTHROPIC_HAIKU_OUTPUT_PER_MTOK
    )


def costs_match(recomputed: float, stored, tolerance: float = COST_TOLERANCE_USD) -> bool:
    """Return whether ``recomputed`` and ``stored`` agree within ``tolerance``.

    A missing/non-numeric stored value never matches. Pure and deterministic.
    """
    try:
        stored_val = float(stored)
    except (TypeError, ValueError):
        return False
    return abs(recomputed - stored_val) <= tolerance


# ===========================================================================
# Sprint 1: row parsing + validation checks over the JSONL ledger.
#
# Everything below layers audit CHECKS on top of the pure recompute core above.
# It stays stdlib-only and reads the ledger from the module-level
# ``COST_LOG_JSONL_PATH`` at CALL time so tests can monkeypatch that attribute
# (same pattern as sessions.py / grounding.py / documents.py). No app/bot/
# pipecat/network/API-key dependency is introduced.
# ===========================================================================

# Stable, enumerated finding categories. Every finding a check raises carries
# exactly one of these as its ``category``. Kept as module-level constants so
# tests and the CLI can reference them without magic strings.
CATEGORY_MALFORMED = "malformed"
CATEGORY_COST_MISMATCH = "cost_mismatch"
CATEGORY_ORPHAN_ARTIFACT = "orphan_artifact"

# The full set, in a deterministic order, so the CLI can print per-category
# counts even for categories with zero findings.
FINDING_CATEGORIES = (
    CATEGORY_MALFORMED,
    CATEGORY_COST_MISMATCH,
    CATEGORY_ORPHAN_ARTIFACT,
)

# Cost fields recomputed + compared per session row, mapped to the recompute
# helper key that produces each. Order is the print/report order.
_SESSION_COST_FIELDS = (
    "cost_llm_usd",
    "cost_stt_usd",
    "cost_tts_usd",
    "cost_post_session_usd",
    "cost_total_usd",
)


class Finding:
    """A single audit finding: one problem discovered on one ledger line.

    Attributes:
      - ``category``: one of ``FINDING_CATEGORIES`` (a stable enumerated kind).
      - ``line_number``: the 1-based line number in the ledger (counting every
        physical line, including malformed ones).
      - ``reason``: a human-readable explanation naming the offending detail.

    Pure data — constructing a Finding does no I/O.
    """

    __slots__ = ("category", "line_number", "reason")

    def __init__(self, category: str, line_number: int, reason: str):
        self.category = category
        self.line_number = line_number
        self.reason = reason

    def __repr__(self) -> str:
        return f"Finding(category={self.category!r}, line_number={self.line_number}, reason={self.reason!r})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Finding):
            return NotImplemented
        return (
            self.category == other.category
            and self.line_number == other.line_number
            and self.reason == other.reason
        )


class AuditResult:
    """Structured summary the audit returns (and the CLI renders).

    Attributes:
      - ``rows_read``: total physical lines read from the ledger (blank lines
        excluded — they are not rows).
      - ``rows_valid``: number of rows that produced NO finding of any category.
      - ``findings``: list of :class:`Finding`, in line order.
      - ``category_counts``: dict mapping every category in
        ``FINDING_CATEGORIES`` to its finding count (0 when none).
    """

    __slots__ = ("rows_read", "rows_valid", "findings", "category_counts")

    def __init__(self, rows_read, rows_valid, findings, category_counts):
        self.rows_read = rows_read
        self.rows_valid = rows_valid
        self.findings = findings
        self.category_counts = category_counts


def _is_number(value) -> bool:
    """True iff ``value`` is a real numeric (int/float, not bool, not str)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _read_ledger_lines(path: Path):
    """Yield ``(line_number, raw_text)`` for each non-blank line of the ledger.

    1-based line numbers count every physical line so a finding's line number
    matches what a human sees in an editor. Fully-blank / whitespace-only lines
    are skipped (they are not rows) but still consume a line number. An absent
    ledger yields nothing.
    """
    if not path.exists():
        return
    with path.open() as f:
        for line_number, raw in enumerate(f, start=1):
            if raw.strip() == "":
                continue
            yield line_number, raw


def _classify_and_check_row(line_number: int, raw: str):
    """Parse + check one ledger line, returning ``(entry_or_None, findings)``.

    ``entry_or_None`` is the parsed dict when the line is a well-formed JSON
    object (used later for the whole-file orphan pass), else ``None``.
    ``findings`` is the (possibly empty) list of per-row findings for this line,
    covering malformed and cost_mismatch. Orphan detection needs the whole file
    and is applied separately.
    """
    findings: list[Finding] = []

    # (a)/(b): non-JSON or non-object → malformed.
    try:
        entry = json.loads(raw)
    except Exception:
        findings.append(
            Finding(
                CATEGORY_MALFORMED,
                line_number,
                "line is not valid JSON",
            )
        )
        return None, findings
    if not isinstance(entry, dict):
        findings.append(
            Finding(
                CATEGORY_MALFORMED,
                line_number,
                f"line is valid JSON but not an object (got {type(entry).__name__})",
            )
        )
        return None, findings

    kind = entry.get("kind")

    # Legacy rows are defined STRICTLY as absence of the ``kind`` field. They
    # are tolerated: skipped by every error check, never flagged.
    if "kind" not in entry:
        return entry, findings

    if kind == "session":
        # (c) malformed iff it can't even be classified/recomputed: needs an
        # identity key. session_id is the minimal identity key.
        if "session_id" not in entry:
            findings.append(
                Finding(
                    CATEGORY_MALFORMED,
                    line_number,
                    "session row missing required 'session_id'",
                )
            )
            return None, findings
        # Cost-mismatch: compare each stored cost_*_usd against recompute.
        recomputed = recompute_session_costs(entry)
        for field in _SESSION_COST_FIELDS:
            stored = entry.get(field)
            expected = recomputed[field]
            if not costs_match(expected, stored):
                if not _is_number(stored):
                    reason = (
                        f"{field}: stored value absent/non-numeric "
                        f"(recomputed {expected:.6f})"
                    )
                else:
                    reason = (
                        f"{field}: stored {float(stored):.6f} != "
                        f"recomputed {expected:.6f}"
                    )
                findings.append(
                    Finding(CATEGORY_COST_MISMATCH, line_number, reason)
                )
        return entry, findings

    if kind == "artifact":
        # (c) malformed iff missing an identity/recompute key.
        missing = [
            key
            for key in ("session_id", "input_tokens", "output_tokens")
            if key not in entry
        ]
        if missing:
            findings.append(
                Finding(
                    CATEGORY_MALFORMED,
                    line_number,
                    "artifact row missing required "
                    + ", ".join(f"'{m}'" for m in missing),
                )
            )
            return None, findings
        # Cost-mismatch: stored cost_usd vs recomputed from tokens.
        expected = recompute_artifact_cost(entry)
        stored = entry.get("cost_usd")
        if not costs_match(expected, stored):
            if not _is_number(stored):
                reason = (
                    f"cost_usd: stored value absent/non-numeric "
                    f"(recomputed {expected:.6f})"
                )
            else:
                reason = (
                    f"cost_usd: stored {float(stored):.6f} != "
                    f"recomputed {expected:.6f}"
                )
            findings.append(
                Finding(CATEGORY_COST_MISMATCH, line_number, reason)
            )
        return entry, findings

    # A ``kind`` present but not in {session, artifact}: not something we audit
    # for cost (unknown row type), and not legacy. Treat as a valid, unchecked
    # row — never flagged. It still participates in the file so it doesn't break
    # numbering, but carries no findings.
    return entry, findings


def audit_cost_log(path: Path | None = None) -> AuditResult:
    """Audit the cost-log ledger and return a structured :class:`AuditResult`.

    Reads the ledger from ``path`` (default: the module-level
    ``COST_LOG_JSONL_PATH``, read at CALL time so tests can monkeypatch it).
    Runs, per row, the malformed and cost_mismatch checks, then a whole-file
    two-phase orphan pass. Pure w.r.t. app/bot/pipecat/network — only stdlib and
    the ledger file are touched.

    Legacy rows (no ``kind``) are tolerated and never flagged. A malformed line
    does not abort the audit; every subsequent line is still processed.
    """
    if path is None:
        path = COST_LOG_JSONL_PATH

    findings: list[Finding] = []
    # Track, for the orphan pass: every session_id seen on a session row, and
    # every artifact occurrence (line_number, session_id).
    session_ids: set = set()
    artifacts: list[tuple[int, object]] = []
    lines_with_findings: set[int] = set()
    rows_read = 0

    for line_number, raw in _read_ledger_lines(path):
        rows_read += 1
        entry, row_findings = _classify_and_check_row(line_number, raw)
        for f in row_findings:
            findings.append(f)
            lines_with_findings.add(f.line_number)

        # Feed the whole-file orphan pass from well-formed rows only.
        if isinstance(entry, dict) and "kind" in entry:
            if entry.get("kind") == "session":
                session_ids.add(entry.get("session_id"))
            elif entry.get("kind") == "artifact" and "session_id" in entry:
                # Only artifacts that passed malformed classification reach here
                # (missing-session_id artifacts returned entry=None already).
                artifacts.append((line_number, entry.get("session_id")))

    # Phase 2: orphan detection using the fully-populated session_ids set, so a
    # forward reference (artifact before its session) is NOT an orphan.
    orphan_findings: list[Finding] = []
    for line_number, sid in artifacts:
        if sid not in session_ids:
            orphan_findings.append(
                Finding(
                    CATEGORY_ORPHAN_ARTIFACT,
                    line_number,
                    f"artifact row references session_id={sid!r} with no matching session row",
                )
            )
            lines_with_findings.add(line_number)

    findings.extend(orphan_findings)
    # Keep findings in a stable line-then-category order for deterministic output.
    findings.sort(key=lambda f: (f.line_number, FINDING_CATEGORIES.index(f.category)))

    category_counts = {cat: 0 for cat in FINDING_CATEGORIES}
    for f in findings:
        category_counts[f.category] += 1

    rows_valid = rows_read - len(lines_with_findings)

    return AuditResult(
        rows_read=rows_read,
        rows_valid=rows_valid,
        findings=findings,
        category_counts=category_counts,
    )


def format_report(result: AuditResult) -> str:
    """Render an :class:`AuditResult` as the human-readable CLI summary.

    Shows rows read, rows valid, per-category failure counts, and — for each
    finding — its 1-based line number together with its category and reason.
    Pure: builds and returns a string, does no I/O.
    """
    lines = []
    lines.append(f"cost-log audit — {COST_LOG_JSONL_PATH}")
    lines.append(f"rows read:  {result.rows_read}")
    lines.append(f"rows valid: {result.rows_valid}")
    lines.append("per-check failure counts:")
    for cat in FINDING_CATEGORIES:
        lines.append(f"  {cat}: {result.category_counts.get(cat, 0)}")
    if result.findings:
        lines.append("findings:")
        for f in result.findings:
            lines.append(f"  line {f.line_number} [{f.category}]: {f.reason}")
    else:
        lines.append("findings: none")
    return "\n".join(lines)


def main(argv=None) -> int:
    """CLI entry point: audit the module-level ledger and print the report.

    Returns a process exit code (0 always — findings are diagnostic output, not
    a failure of the audit tool itself). Renders the SAME AuditResult object the
    tests grade, so stdout is a view of that structured result.
    """
    result = audit_cost_log()
    print(format_report(result))
    return 0


if __name__ == "__main__":
    import sys as _sys

    raise SystemExit(main(_sys.argv[1:]))
