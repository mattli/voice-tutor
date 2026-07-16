"""Coverage driver for session_state.py moved helpers.

This module imports ONLY session_state (never bot), so it can run under a
coverage subprocess in any environment — including a bare Pipecat-free venv —
without pulling pypdf/pipecat/anthropic. It exercises every reachable branch of
each moved helper so branch coverage over the moved spans reaches 100%.

Run indirectly by test_session_state_coverage.py; also collectable as a normal
test module (its assertions duplicate the pinned characterization literals).
"""

import json
import locale
from pathlib import Path

import pytest

import session_state as ss


@pytest.fixture(autouse=True)
def _locale_c():
    saved = locale.setlocale(locale.LC_TIME)
    for cand in ("C", "en_US.UTF-8", "C.UTF-8"):
        try:
            locale.setlocale(locale.LC_TIME, cand)
            break
        except locale.Error:
            continue
    else:
        pytest.skip("no deterministic locale")
    yield
    locale.setlocale(locale.LC_TIME, saved)


@pytest.fixture
def tmp_consts(tmp_path, monkeypatch):
    root = tmp_path / ".voice-tutor"
    monkeypatch.setattr(ss, "VOICE_TUTOR_DIR", root)
    monkeypatch.setattr(ss, "TRANSCRIPTS_DIR", root / "transcripts")
    monkeypatch.setattr(ss, "PROFILE_PATH", root / "profile.md")
    monkeypatch.setattr(ss, "MEMORY_PATH", root / "memory.md")
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_cover_load_profile_both_branches(tmp_consts):
    assert ss.load_profile() == ""
    ss.PROFILE_PATH.write_text("p")
    assert ss.load_profile() == "p"


def test_cover_load_memory_both_branches(tmp_consts):
    assert ss.load_memory() == ""
    ss.MEMORY_PATH.write_text("m")
    assert ss.load_memory() == "m"


def test_cover_append_to_memory_both_branches(tmp_consts):
    # created branch
    ss.append_to_memory({"session_start": "2026-04-14T09:05:00", "turns": []}, "x")
    first = ss.MEMORY_PATH.read_text()
    assert first.startswith("# Memory")
    # append branch
    ss.append_to_memory({"session_start": "2026-04-14T09:05:00", "turns": []}, "y")
    second = ss.MEMORY_PATH.read_text()
    assert second.startswith(first)
    assert second.count("# Memory") == 1


def test_cover_format_memory_date_boundaries():
    assert ss._format_memory_date("2026-01-01T00:07:00") == "January 1, 2026 — 12:07 AM"
    assert ss._format_memory_date("2026-12-25T12:30:00") == "December 25, 2026 — 12:30 PM"
    assert ss._format_memory_date("2026-04-14T09:05:00") == "April 14, 2026 — 9:05 AM"


def test_cover_format_session_time_boundaries():
    assert ss._format_session_time("2026-01-01T00:07:00") == "January 1, 2026 at 12:07am"
    assert ss._format_session_time("2026-12-25T12:30:00") == "December 25, 2026 at 12:30pm"
    assert ss._format_session_time("2026-04-14T09:05:00") == "April 14, 2026 at 9:05am"


def test_cover_format_full_transcript_block():
    t = {"session_start": "2026-04-14T09:05:00",
         "turns": [{"role": "assistant", "content": "Hi"}, {"role": "user", "content": "Hello there"}]}
    assert ss._format_full_transcript_block(t) == (
        "## Session from April 14, 2026 at 9:05am\n  You: Hi\n  Matt: Hello there"
    )
    assert ss._format_full_transcript_block(t, header_suffix=" (most recent)").startswith(
        "## Session from April 14, 2026 at 9:05am (most recent)\n"
    )


def test_cover_load_most_recent_all_branches(tmp_consts):
    # missing dir -> None
    assert ss.load_most_recent_transcript_block() is None
    # dir exists, no eligible files (.usage.json only) -> None
    ss.TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    (ss.TRANSCRIPTS_DIR / "x.usage.json").write_text("{}")
    assert ss.load_most_recent_transcript_block() is None
    # populated -> newest formatted block
    sample = {"session_start": "2026-04-14T09:05:00", "turns": [{"role": "assistant", "content": "Hi"}]}
    (ss.TRANSCRIPTS_DIR / "2026-04-14.json").write_text(json.dumps(sample))
    out = ss.load_most_recent_transcript_block()
    assert out.startswith("## Session from April 14, 2026 at 9:05am (most recent)")
