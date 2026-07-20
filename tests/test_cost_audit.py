"""Hermetic tests for the pure cost_audit recomputation core (sprint 0).

These tests pin the recomputation math against known usage->cost fixtures and
prove the module is Pipecat-free / network-free / key-free: it imports cleanly
with all provider API keys unset and without importing bot or app.

Everything here is stdlib-only and touches no real filesystem or network. The
recomputation helpers read only the row dict passed in, so no fixture ledger is
needed to exercise them.
"""

import ast
import os
import subprocess
import sys
from pathlib import Path

import cost_audit

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COST_AUDIT_PATH = PROJECT_ROOT / "cost_audit.py"


# ---------------------------------------------------------------------------
# c1 / c5 / c6: pure import — no keys, no bot/app/pipecat/third-party imports.
# ---------------------------------------------------------------------------
def test_imports_with_api_keys_unset_and_no_bot_import():
    """`import cost_audit` succeeds in a subprocess with all provider keys
    unset and API-key access impossible, and afterward neither bot nor app nor
    pipecat has been imported."""
    env = {k: v for k, v in os.environ.items()}
    for key in ("ANTHROPIC_API_KEY", "DEEPGRAM_API_KEY", "CARTESIA_API_KEY"):
        env.pop(key, None)
    # Force any accidental key read to be observable as absent.
    code = (
        "import sys\n"
        "import cost_audit\n"
        "assert 'bot' not in sys.modules, 'cost_audit imported bot'\n"
        "assert 'app' not in sys.modules, 'cost_audit imported app'\n"
        "assert 'pipecat' not in sys.modules, 'cost_audit imported pipecat'\n"
        "assert 'anthropic' not in sys.modules, 'cost_audit imported anthropic'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"import under unset keys failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )


def test_source_has_no_forbidden_imports():
    """Static proof: cost_audit's module-scope imports are stdlib only — no
    bot/app/pipecat/anthropic and no third-party top-level module."""
    tree = ast.parse(COST_AUDIT_PATH.read_text())
    stdlib = set(getattr(sys, "stdlib_module_names", set())) | {"json", "pathlib"}
    forbidden = {"bot", "app", "pipecat", "anthropic"}
    tops: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                tops.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                continue
            tops.add((node.module or "").split(".")[0])
    tops.discard("")
    assert tops & forbidden == set(), f"forbidden imports present: {tops & forbidden}"
    non_stdlib = tops - stdlib
    assert non_stdlib == set(), f"non-stdlib imports present: {non_stdlib}"


# ---------------------------------------------------------------------------
# c2: the eight verbatim pricing constants exist and are numeric.
# ---------------------------------------------------------------------------
_PRICE_NAMES = (
    "PRICE_ANTHROPIC_INPUT_PER_MTOK",
    "PRICE_ANTHROPIC_OUTPUT_PER_MTOK",
    "PRICE_ANTHROPIC_CACHE_READ_PER_MTOK",
    "PRICE_ANTHROPIC_CACHE_WRITE_PER_MTOK",
    "PRICE_ANTHROPIC_HAIKU_INPUT_PER_MTOK",
    "PRICE_ANTHROPIC_HAIKU_OUTPUT_PER_MTOK",
    "PRICE_DEEPGRAM_NOVA3_PER_MIN",
    "PRICE_CARTESIA_PER_CHAR",
)


def test_all_price_constants_present_and_numeric():
    for name in _PRICE_NAMES:
        assert hasattr(cost_audit, name), f"missing constant {name}"
        val = getattr(cost_audit, name)
        assert isinstance(val, (int, float)) and not isinstance(val, bool), name


def test_price_constants_verbatim_values():
    # Pin the relocated values byte-for-byte against the pre-move bot.py RHS.
    assert cost_audit.PRICE_ANTHROPIC_INPUT_PER_MTOK == 3.00
    assert cost_audit.PRICE_ANTHROPIC_OUTPUT_PER_MTOK == 15.00
    assert cost_audit.PRICE_ANTHROPIC_CACHE_WRITE_PER_MTOK == 3.75
    assert cost_audit.PRICE_ANTHROPIC_CACHE_READ_PER_MTOK == 0.30
    assert cost_audit.PRICE_ANTHROPIC_HAIKU_INPUT_PER_MTOK == 1.00
    assert cost_audit.PRICE_ANTHROPIC_HAIKU_OUTPUT_PER_MTOK == 5.00
    assert cost_audit.PRICE_DEEPGRAM_NOVA3_PER_MIN == 0.0077
    assert cost_audit.PRICE_CARTESIA_PER_CHAR == 5.00 / 100_000


def test_bot_reexports_same_constants(imported_bot):
    """bot.py re-imports the relocated constants; every price bot exposes is the
    same object/value as cost_audit's (verbatim, no drift). Uses the
    pipecat-stub fixture so `import bot` works Pipecat-free."""
    bot = imported_bot
    for name in _PRICE_NAMES:
        assert hasattr(bot, name), f"bot no longer exposes {name}"
        assert getattr(bot, name) == getattr(cost_audit, name), name
        # Re-imported names are the identical objects.
        assert getattr(bot, name) is getattr(cost_audit, name), name


# ---------------------------------------------------------------------------
# c7: single shared tolerance constant, >= 5e-5.
# ---------------------------------------------------------------------------
def test_tolerance_constant_present_and_loose_enough():
    assert hasattr(cost_audit, "COST_TOLERANCE_USD")
    assert isinstance(cost_audit.COST_TOLERANCE_USD, float)
    assert cost_audit.COST_TOLERANCE_USD >= 5e-5


# ---------------------------------------------------------------------------
# c3: session recomputation — non-zero cache + post-session tokens.
# ---------------------------------------------------------------------------
# A study session+artifact-style row with every token class non-zero. Hand
# computed against the relocated rates:
#   llm  = 12_000/1e6*3.00 + 400_000/1e6*0.30 + 8_000/1e6*3.75 + 5_000/1e6*15.00
#        = 0.036 + 0.12 + 0.03 + 0.075 = 0.261
#   stt  = 4.25 * 0.0077                                = 0.0327250
#   tts  = 9_000 * (5/100_000)                          = 0.45
#   post = 20_000/1e6*1.00 + 3_000/1e6*5.00 = 0.02 + 0.015 = 0.035
#   total = 0.261 + 0.032725 + 0.45 + 0.035            = 0.778725
_SESSION_ROW = {
    "kind": "session",
    "mode": "study",
    "tts_chars": 9_000,
    "stt_minutes_billed": 4.25,
    "llm_uncached_input_tokens": 12_000,
    "llm_cache_read_tokens": 400_000,
    "llm_cache_write_tokens": 8_000,
    "llm_output_tokens": 5_000,
    "post_session_input_tokens": 20_000,
    "post_session_output_tokens": 3_000,
    "cost_llm_usd": 0.261,
    "cost_stt_usd": 0.0327,
    "cost_tts_usd": 0.45,
    "cost_post_session_usd": 0.035,
    "cost_total_usd": 0.7787,
}


def test_session_llm_component_uses_four_distinct_rates():
    tol = cost_audit.COST_TOLERANCE_USD
    got = cost_audit.recompute_llm_cost(_SESSION_ROW)
    assert abs(got - 0.261) <= tol
    # Cache tokens are load-bearing: omitting them changes the answer.
    without_cache = dict(_SESSION_ROW)
    without_cache["llm_cache_read_tokens"] = 0
    without_cache["llm_cache_write_tokens"] = 0
    assert abs(cost_audit.recompute_llm_cost(without_cache) - 0.261) > tol


def test_session_stt_uses_stored_billed_minutes():
    tol = cost_audit.COST_TOLERANCE_USD
    got = cost_audit.recompute_stt_cost(_SESSION_ROW)
    assert abs(got - 0.032725) <= tol


def test_session_tts_uses_chars():
    tol = cost_audit.COST_TOLERANCE_USD
    assert abs(cost_audit.recompute_tts_cost(_SESSION_ROW) - 0.45) <= tol


def test_session_post_session_uses_haiku_rates():
    tol = cost_audit.COST_TOLERANCE_USD
    got = cost_audit.recompute_post_session_cost(_SESSION_ROW)
    assert abs(got - 0.035) <= tol
    # Non-zero post-session tokens genuinely contribute.
    assert got > 0


def test_session_full_recompute_components_and_total():
    tol = cost_audit.COST_TOLERANCE_USD
    got = cost_audit.recompute_session_costs(_SESSION_ROW)
    assert abs(got["cost_llm_usd"] - 0.261) <= tol
    assert abs(got["cost_stt_usd"] - 0.032725) <= tol
    assert abs(got["cost_tts_usd"] - 0.45) <= tol
    assert abs(got["cost_post_session_usd"] - 0.035) <= tol
    assert abs(got["cost_total_usd"] - 0.778725) <= tol
    # Total is exactly the sum of the four components.
    assert abs(
        got["cost_total_usd"]
        - (
            got["cost_llm_usd"]
            + got["cost_stt_usd"]
            + got["cost_tts_usd"]
            + got["cost_post_session_usd"]
        )
    ) <= tol


def test_session_recompute_matches_stored_within_tolerance():
    """A correct session row's recomputed components compare equal to its stored
    (already-rounded) cost_*_usd via the shared tolerance — no false mismatch."""
    got = cost_audit.recompute_session_costs(_SESSION_ROW)
    assert cost_audit.costs_match(got["cost_llm_usd"], _SESSION_ROW["cost_llm_usd"])
    assert cost_audit.costs_match(got["cost_stt_usd"], _SESSION_ROW["cost_stt_usd"])
    assert cost_audit.costs_match(got["cost_tts_usd"], _SESSION_ROW["cost_tts_usd"])
    assert cost_audit.costs_match(
        got["cost_post_session_usd"], _SESSION_ROW["cost_post_session_usd"]
    )
    assert cost_audit.costs_match(got["cost_total_usd"], _SESSION_ROW["cost_total_usd"])


def test_legacy_session_row_missing_fields_treated_as_zero():
    """A legacy row (no cache/post-session fields) recomputes without raising;
    absent usage classes count as zero."""
    legacy = {
        "tts_chars": 1_000,
        "stt_minutes_billed": 1.0,
        "llm_uncached_input_tokens": 1_000,
        "llm_output_tokens": 500,
    }
    got = cost_audit.recompute_session_costs(legacy)
    tol = cost_audit.COST_TOLERANCE_USD
    # llm = 1000/1e6*3 + 500/1e6*15 = 0.003 + 0.0075 = 0.0105
    assert abs(got["cost_llm_usd"] - 0.0105) <= tol
    assert abs(got["cost_post_session_usd"] - 0.0) <= tol


# ---------------------------------------------------------------------------
# c4: artifact recomputation — Haiku rates, would fail under Sonnet.
# ---------------------------------------------------------------------------
# input=60_000, output=4_000
#   haiku  = 60_000/1e6*1.00 + 4_000/1e6*5.00 = 0.06 + 0.02 = 0.08
#   sonnet = 60_000/1e6*3.00 + 4_000/1e6*15.00 = 0.18 + 0.06 = 0.24 (different!)
_ARTIFACT_ROW = {
    "kind": "artifact",
    "input_tokens": 60_000,
    "output_tokens": 4_000,
    "cost_usd": 0.08,
}


def test_artifact_cost_uses_haiku_rates_not_sonnet():
    tol = cost_audit.COST_TOLERANCE_USD
    got = cost_audit.recompute_artifact_cost(_ARTIFACT_ROW)
    assert abs(got - 0.08) <= tol
    # Explicitly prove it is NOT the Sonnet-rate answer.
    sonnet = (
        60_000 / 1_000_000 * cost_audit.PRICE_ANTHROPIC_INPUT_PER_MTOK
        + 4_000 / 1_000_000 * cost_audit.PRICE_ANTHROPIC_OUTPUT_PER_MTOK
    )
    assert abs(got - sonnet) > tol
    assert cost_audit.costs_match(got, _ARTIFACT_ROW["cost_usd"])


# ---------------------------------------------------------------------------
# c5: helpers are total & deterministic — identical inputs, identical outputs.
# ---------------------------------------------------------------------------
def test_helpers_are_deterministic():
    a = cost_audit.recompute_session_costs(_SESSION_ROW)
    b = cost_audit.recompute_session_costs(_SESSION_ROW)
    assert a == b
    assert cost_audit.recompute_artifact_cost(_ARTIFACT_ROW) == cost_audit.recompute_artifact_cost(
        _ARTIFACT_ROW
    )
    assert cost_audit.recompute_llm_cost(_SESSION_ROW) == cost_audit.recompute_llm_cost(_SESSION_ROW)
