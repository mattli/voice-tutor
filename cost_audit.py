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
