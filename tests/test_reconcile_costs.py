"""Unit tests for the PURE ledger-summing + reconciliation-math core of
``reconcile_costs.py``.

Scope is deliberately narrow (per the tool's design): only the pure, hermetic
parts are tested here — ledger parsing, per-provider summing, the local-time
date-range filter (including the artifact→session join), token-bucket pricing,
and the ReconLine/verdict math. The network fetchers (Anthropic/Deepgram/
Cartesia HTTP) are NOT tested — they're verified by running the tool for real
against the provider APIs. No network, no credentials, no clock here.
"""

from datetime import datetime

import cost_audit
import reconcile_costs as rc


# --- fixtures: representative ledger rows -----------------------------------

def _session_row(**over):
    row = {
        "kind": "session",
        "session_id": "s1",
        "session_start": "2026-04-26T16:17:47.912037",
        "session_end": "2026-04-26T16:18:26.027041",
        "tts_chars": 2416,
        "stt_minutes_billed": 0.64,
        "llm_uncached_input_tokens": 6400,
        "llm_cache_read_tokens": 100,
        "llm_cache_write_tokens": 200,
        "llm_output_tokens": 775,
        "post_session_input_tokens": 50,
        "post_session_output_tokens": 30,
        "cost_llm_usd": 0.0308,
        "cost_stt_usd": 0.0049,
        "cost_tts_usd": 0.1208,
        "cost_post_session_usd": 0.0002,
    }
    row.update(over)
    return row


def _artifact_row(**over):
    row = {
        "kind": "artifact",
        "session_id": "s1",
        "input_tokens": 576,
        "output_tokens": 100,
        "cost_usd": 0.0011,
    }
    row.update(over)
    return row


def _legacy_row(**over):
    # Legacy rows have no "kind"; carry tts_credits (not tts_chars) and no
    # post_session/artifact fields.
    row = {
        "session_id": "2026-04-15T163747",
        "session_start": "2026-04-15T16:37:47.158740",
        "session_end": "2026-04-15T16:58:33.234349",
        "tts_credits": 45757.0,
        "stt_minutes_billed": 20.77,
        "llm_uncached_input_tokens": 1165,
        "llm_cache_read_tokens": 9073840,
        "llm_cache_write_tokens": 291205,
        "llm_output_tokens": 16990,
        "cost_llm_usd": 4.0725,
        "cost_stt_usd": 0.1599,
        "cost_tts_usd": 2.2879,
    }
    row.update(over)
    return row


# --- summarize_ledger: token attribution ------------------------------------

def test_session_row_fills_live_bucket():
    t = rc.summarize_ledger([_session_row()])
    assert t.live_tokens.uncached_input == 6400
    assert t.live_tokens.cache_read == 100
    assert t.live_tokens.cache_write == 200
    assert t.live_tokens.output == 775
    # post_session tokens land in the haiku bucket.
    assert t.haiku_tokens.uncached_input == 50
    assert t.haiku_tokens.output == 30
    assert t.session_rows == 1


def test_artifact_row_fills_haiku_bucket():
    t = rc.summarize_ledger([_artifact_row()])
    assert t.haiku_tokens.uncached_input == 576
    assert t.haiku_tokens.output == 100
    assert t.haiku_tokens.cache_read == 0
    assert t.artifact_rows == 1
    # An artifact contributes nothing to the live/Sonnet bucket.
    assert t.live_tokens.total() == 0


def test_session_plus_artifact_haiku_accumulates():
    t = rc.summarize_ledger([_session_row(), _artifact_row()])
    # 50 (post-session) + 576 (artifact) input; 30 + 100 output.
    assert t.haiku_tokens.uncached_input == 626
    assert t.haiku_tokens.output == 130


def test_legacy_row_counts_as_session_for_live_tokens():
    t = rc.summarize_ledger([_legacy_row()])
    assert t.live_tokens.uncached_input == 1165
    assert t.live_tokens.cache_read == 9073840
    assert t.live_tokens.output == 16990
    # No post_session fields -> haiku bucket empty.
    assert t.haiku_tokens.total() == 0
    assert t.session_rows == 1


# --- summarize_ledger: Deepgram + Cartesia ----------------------------------

def test_deepgram_minutes_sum():
    t = rc.summarize_ledger([_session_row(), _session_row(stt_minutes_billed=1.36)])
    assert t.stt_minutes == 0.64 + 1.36


def test_cartesia_chars_modern_and_legacy():
    t = rc.summarize_ledger([_session_row(), _legacy_row()])
    # modern tts_chars (2416) + legacy tts_credits (45757) treated as chars.
    assert t.tts_chars == 2416 + 45757.0
    # 1 credit == 1 char.
    assert t.tts_credits == t.tts_chars


def test_recorded_usd_per_provider():
    t = rc.summarize_ledger([_session_row()])
    assert abs(t.recorded_anthropic_usd - (0.0308 + 0.0002)) < 1e-9
    assert abs(t.recorded_deepgram_usd - 0.0049) < 1e-9
    assert abs(t.recorded_cartesia_usd - 0.1208) < 1e-9


def test_missing_fields_default_to_zero():
    t = rc.summarize_ledger([{"kind": "session", "session_id": "x"}])
    assert t.live_tokens.total() == 0
    assert t.stt_minutes == 0
    assert t.tts_chars == 0


# --- date-range filtering + artifact join -----------------------------------

def test_range_filters_out_of_window_session():
    early = _session_row(session_id="early", session_start="2026-01-01T10:00:00")
    late = _session_row(session_id="late", session_start="2026-06-01T10:00:00")
    start = datetime(2026, 5, 1)
    end = datetime(2026, 7, 1)
    t = rc.summarize_ledger([early, late], start, end)
    # Only the late session's tokens are counted.
    assert t.session_rows == 1
    assert t.live_tokens.uncached_input == 6400


def test_artifact_joins_session_time_for_range():
    # Artifact has no timestamp; it must inherit its session's time. Session s1
    # is 2026-04-26, inside the window -> artifact included.
    rows = [_session_row(), _artifact_row()]
    t = rc.summarize_ledger(rows, datetime(2026, 4, 1), datetime(2026, 5, 1))
    assert t.artifact_rows == 1
    assert t.haiku_tokens.uncached_input == 50 + 576


def test_artifact_excluded_when_its_session_out_of_range():
    rows = [_session_row(), _artifact_row()]
    # Window excludes the April session -> its artifact drops too.
    t = rc.summarize_ledger(rows, datetime(2026, 5, 1), datetime(2026, 6, 1))
    assert t.artifact_rows == 0
    assert t.session_rows == 0
    assert t.haiku_tokens.total() == 0


def test_no_bounds_keeps_everything():
    rows = [_session_row(), _artifact_row(), _legacy_row()]
    t = rc.summarize_ledger(rows, None, None)
    assert t.session_rows == 2  # session + legacy
    assert t.artifact_rows == 1


# --- pricing a token bucket matches cost_audit's recompute ------------------

def test_price_live_bucket_matches_cost_audit_recompute():
    row = _session_row()
    bucket = rc.TokenBucket(
        uncached_input=row["llm_uncached_input_tokens"],
        cache_read=row["llm_cache_read_tokens"],
        cache_write=row["llm_cache_write_tokens"],
        output=row["llm_output_tokens"],
    )
    assert abs(rc.price_token_bucket(bucket, "live") - cost_audit.recompute_llm_cost(row)) < 1e-12


def test_price_haiku_bucket_matches_artifact_recompute():
    row = _artifact_row()
    bucket = rc.TokenBucket(uncached_input=row["input_tokens"], output=row["output_tokens"])
    assert abs(rc.price_token_bucket(bucket, "haiku") - cost_audit.recompute_artifact_cost(row)) < 1e-12


# --- reconciliation math ----------------------------------------------------

def test_reconline_abs_and_pct():
    line = rc.ReconLine("x", "tokens", ledger=100.0, provider=110.0)
    assert line.abs_diff == 10.0
    assert abs(line.pct_diff - 10.0) < 1e-9


def test_reconline_pct_none_when_ledger_zero():
    line = rc.ReconLine("x", "tokens", ledger=0.0, provider=5.0)
    assert line.pct_diff is None
    # No baseline and provider != 0 -> not within tolerance.
    assert not line.within(1.0)
    # Both zero -> within.
    assert rc.ReconLine("x", "tokens", 0.0, 0.0).within(1.0)


def test_within_tolerance_band():
    line = rc.ReconLine("x", "$", ledger=100.0, provider=100.5)
    assert line.within(1.0)
    assert not line.within(0.1)


def test_verdict_all_within_is_match():
    lines = [
        rc.ReconLine("a", "tokens", 100, 100.5),
        rc.ReconLine("b", "tokens", 200, 201),
    ]
    assert rc.verdict_for(lines, 1.0) == "MATCH"


def test_verdict_one_out_is_discrepancy():
    lines = [
        rc.ReconLine("a", "tokens", 100, 100.5),
        rc.ReconLine("b", "tokens", 200, 260),
    ]
    assert rc.verdict_for(lines, 1.0) == "DISCREPANCY"


# --- model classification + timestamp parsing -------------------------------

def test_classify_model():
    assert rc.classify_model("claude-haiku-4-5-20251001") == "haiku"
    assert rc.classify_model("claude-sonnet-4-5-20250929") == "live"
    assert rc.classify_model(None) == "live"


def test_parse_local_ts_naive():
    dt = rc.parse_local_ts("2026-04-26T16:17:47.912037")
    assert dt == datetime(2026, 4, 26, 16, 17, 47, 912037)
    assert dt.tzinfo is None
    assert rc.parse_local_ts(None) is None
    assert rc.parse_local_ts("not-a-date") is None


def test_ledger_local_bounds():
    lo, hi = rc.ledger_local_bounds([_legacy_row(), _session_row()])
    assert lo == datetime(2026, 4, 15, 16, 37, 47, 158740)
    assert hi == datetime(2026, 4, 26, 16, 18, 26, 27041)


# --- ledger loading ---------------------------------------------------------

def test_load_ledger_rows_skips_blank_and_malformed(tmp_path):
    p = tmp_path / "cost-log.jsonl"
    p.write_text(
        '{"kind": "session", "session_id": "s1"}\n'
        "\n"
        "not json\n"
        "[1,2,3]\n"  # valid JSON but not an object -> skipped
        '{"kind": "artifact", "session_id": "s1"}\n'
    )
    rows = rc.load_ledger_rows(p)
    assert len(rows) == 2
    assert rows[0]["kind"] == "session"
    assert rows[1]["kind"] == "artifact"
