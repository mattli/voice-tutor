"""Hermetic tests for the pure usage_ledger module.

usage_ledger.UsageLedger holds a single tutoring session's usage counters, the
per-hop dedup bookkeeping, and the cost `summary()` math — all Pipecat-free, so
it is tested here with plain values (no pipecat frames, no network, no keys, no
filesystem). bot.py's UsageAccumulator(BaseObserver) is a thin adapter over it.

Background (2026-07-22 runtime evidence): Pipecat fires on_push_frame once per
processor hop, and one pipeline-wide observer sees every hop, so the legacy
accumulator counted each frame's usage once PER HOP — measured LLM tokens 5.00x,
STT audio 8.00x, TTS audio ~2.63x, all id-stable. UsageLedger dedups by frame
id so each frame's usage counts exactly once.
"""

import os
import subprocess
import sys
from pathlib import Path

import usage_ledger
from usage_ledger import UsageLedger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
USAGE_LEDGER_PATH = PROJECT_ROOT / "usage_ledger.py"


# ---------------------------------------------------------------------------
# Purity: imports with keys unset, without pulling in bot/app/pipecat.
# ---------------------------------------------------------------------------
def test_imports_pure_no_keys_no_bot_no_pipecat():
    env = {k: v for k, v in os.environ.items()}
    for key in ("ANTHROPIC_API_KEY", "DEEPGRAM_API_KEY", "CARTESIA_API_KEY"):
        env.pop(key, None)
    code = (
        "import sys\n"
        "import usage_ledger\n"
        "assert 'bot' not in sys.modules, 'usage_ledger imported bot'\n"
        "assert 'app' not in sys.modules, 'usage_ledger imported app'\n"
        "assert 'pipecat' not in sys.modules, 'usage_ledger imported pipecat'\n"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=str(PROJECT_ROOT), env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


# ---------------------------------------------------------------------------
# The core fix: dedup counts each frame id exactly once (once per FRAME, not
# once per hop). This is the regression guard for the 5.00x LLM multi-count.
# ---------------------------------------------------------------------------
def test_dedup_on_counts_llm_frame_once_across_hops():
    led = UsageLedger(dedup=True)
    # One LLM response observed across 5 processor hops (same frame id).
    for _hop in range(5):
        if led.should_count(101):
            led.add_llm_usage(prompt_tokens=10, cache_read=1000,
                              cache_write=100, completion=50)
    assert led.cache_read_tokens == 1000     # once, NOT 5000
    assert led.cache_write_tokens == 100
    assert led.uncached_input_tokens == 10
    assert led.output_tokens == 50


def test_dedup_off_reproduces_legacy_multicount():
    # With the flag off, behavior is the legacy per-hop multi-count (5x here),
    # so the escape hatch demonstrably restores old behavior.
    led = UsageLedger(dedup=False)
    for _hop in range(5):
        if led.should_count(101):
            led.add_llm_usage(prompt_tokens=10, cache_read=1000,
                              cache_write=100, completion=50)
    assert led.cache_read_tokens == 5000     # legacy 5x
    assert led.output_tokens == 250


def test_should_count_returns_true_once_then_false():
    led = UsageLedger(dedup=True)
    assert led.should_count(7) is True
    assert led.should_count(7) is False
    assert led.should_count(7) is False
    assert led.should_count(8) is True       # a different frame still counts


def test_distinct_frames_each_counted_once():
    led = UsageLedger(dedup=True)
    for fid in (201, 202, 203):              # 3 turns, each seen twice
        for _hop in range(2):
            if led.should_count(fid):
                led.add_llm_usage(prompt_tokens=0, cache_read=100,
                                  cache_write=0, completion=0)
    assert led.cache_read_tokens == 300      # 3 frames x 100, not x2 hops


# ---------------------------------------------------------------------------
# Frame-level dedup: one MetricsFrame can carry BOTH LLM and TTS usage. Dedup is
# decided once per frame id; both metrics must apply on that single count.
# ---------------------------------------------------------------------------
def test_one_frame_with_llm_and_tts_applies_both_once():
    led = UsageLedger(dedup=True)
    for _hop in range(4):
        if led.should_count(55):
            led.add_llm_usage(prompt_tokens=5, cache_read=0,
                              cache_write=0, completion=3)
            led.add_tts_chars(120)
    assert led.output_tokens == 3            # once
    assert led.tts_chars == 120              # once, not dropped and not x4


# ---------------------------------------------------------------------------
# Audio observability counters dedup the same way.
# ---------------------------------------------------------------------------
def test_add_llm_usage_tolerates_none_cache_fields():
    # The adapter feeds u.cache_read_input_tokens / u.cache_creation_input_tokens
    # straight through, and pipecat can hand those back as None. The `or 0` path
    # must count them as zero, not crash. (This is the real production input path.)
    led = UsageLedger(dedup=True)
    if led.should_count(9):
        led.add_llm_usage(prompt_tokens=12, cache_read=None,
                          cache_write=None, completion=4)
    assert led.cache_read_tokens == 0
    assert led.cache_write_tokens == 0
    assert led.uncached_input_tokens == 12
    assert led.output_tokens == 4


def test_frame_with_no_usage_metrics_consumes_id_and_counts_nothing():
    # Mirrors a MetricsFrame with empty .data (or only non-usage metrics): the
    # adapter still calls should_count(frame.id) once (deduping the id) but applies
    # no counters. The id is consumed; all totals stay zero.
    led = UsageLedger(dedup=True)
    assert led.should_count(500) is True     # first hop consumes the id
    assert led.should_count(500) is False    # later hops deduped, still no counting
    assert led.uncached_input_tokens == 0
    assert led.output_tokens == 0
    assert led.tts_chars == 0


def test_stt_and_tts_audio_dedup():
    led = UsageLedger(dedup=True)
    for _hop in range(8):                    # STT audio measured at 8.00x
        if led.should_count(300):
            led.add_stt_audio(0.02)
    for _hop in range(3):
        if led.should_count(301):
            led.add_tts_audio(0.05)
    assert led.stt_audio_sec == 0.02
    assert led.tts_audio_sec == 0.05


# ---------------------------------------------------------------------------
# Bounded seen-set: a frame re-sighted within the hop-span window is still
# deduped; ids only recur as "new" after the cap has evicted them. The measured
# max hop-span (295) is far below any sane cap, so real sessions never evict
# early. This test pins the boundary behavior with a tiny cap.
# ---------------------------------------------------------------------------
def test_bounded_set_dedups_within_window():
    led = UsageLedger(dedup=True, seen_id_cap=4)
    # Interleave: frame A seen, three others intervene (still within cap), A again.
    assert led.should_count(1) is True
    for other in (2, 3, 4):
        assert led.should_count(other) is True
    assert led.should_count(1) is False      # A still remembered (cap not exceeded by A's span)


def test_bounded_set_evicts_oldest_beyond_cap():
    led = UsageLedger(dedup=True, seen_id_cap=3)
    for fid in (1, 2, 3):
        assert led.should_count(fid) is True
    assert led.should_count(4) is True       # inserting 4 evicts oldest (1) -> seen {2,3,4}
    assert led.should_count(1) is True       # 1 was evicted, so new again -> evicts 2 -> {3,4,1}
    assert led.should_count(3) is False      # 3 still within the window
    # (Documents FIFO eviction; real caps sit ~14x above the measured 295 span.)


def test_bound_never_undercounts_below_measured_hop_span():
    # Safety property that matters in production: with cap >> max hop-span, a
    # frame's later hops are never evicted before they are seen, so no undercount.
    led = UsageLedger(dedup=True, seen_id_cap=4096)
    # Simulate frame id 100 spanning 295 intervening distinct ids between hops.
    assert led.should_count(100) is True
    for other in range(200, 200 + 295):
        led.should_count(other)
    assert led.should_count(100) is False    # still deduped across a 295-id span


# ---------------------------------------------------------------------------
# summary(): cost math moved verbatim from bot.py. Pin key mappings (no swapped
# price lines) and the STT wall-clock basis, using the real price constants.
# ---------------------------------------------------------------------------
def test_summary_all_zero_is_zero_cost():
    led = UsageLedger()
    s = led.summary(session_duration_sec=0.0)
    assert s["total_cost_usd"] == 0.0
    assert s["llm"]["cost_usd"] == 0.0
    assert s["stt"]["cost_usd"] == 0.0
    assert s["tts"]["cost_usd"] == 0.0


def test_summary_maps_each_counter_to_its_cost_line():
    from cost_audit import (
        PRICE_ANTHROPIC_CACHE_READ_PER_MTOK,
        PRICE_CARTESIA_PER_CHAR,
        PRICE_DEEPGRAM_NOVA3_PER_MIN,
    )
    led = UsageLedger()
    led.cache_read_tokens = 1_000_000        # exactly 1 Mtok cache-read
    led.tts_chars = 1000
    s = led.summary(session_duration_sec=60.0)  # exactly 1 minute
    assert s["llm"]["cache_read_tokens"] == 1_000_000
    assert s["llm"]["cost_usd"] == round(PRICE_ANTHROPIC_CACHE_READ_PER_MTOK, 4)
    assert s["stt"]["minutes"] == 1.0
    assert s["stt"]["cost_usd"] == round(PRICE_DEEPGRAM_NOVA3_PER_MIN, 4)
    assert s["tts"]["chars"] == 1000
    assert s["tts"]["cost_usd"] == round(1000 * PRICE_CARTESIA_PER_CHAR, 4)


def test_summary_stt_uses_wall_clock_not_observed_audio():
    # STT billing basis is session_duration_sec/60, NOT stt_audio_sec (which is
    # observability only). Setting a wildly different observed value must not move
    # the STT cost.
    led = UsageLedger()
    led.stt_audio_sec = 9999.0
    s = led.summary(session_duration_sec=120.0)
    assert s["stt"]["minutes"] == 2.0
    assert s["stt"]["audio_sec_observed"] == 9999.0   # reported, not billed
