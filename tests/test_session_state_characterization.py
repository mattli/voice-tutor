"""c8 + c9 + c10: characterization of the relocated helpers.

Pins the CURRENT observed behavior of each moved helper with hardcoded literals
against the shared CASES corpus, exercises every reachable branch (using the
hermetic tmp-dir fixture), and proves both import paths (bot.<h> and
session_state.<h>) agree.
"""

import json

import pytest

import session_state as ss
from tests.session_state_cases import (
    CASES,
    MIDNIGHT_TRANSCRIPT,
    MOVED_HELPERS,
    NOON_TRANSCRIPT,
    SAMPLE_TRANSCRIPT,
)


# ---------------------------------------------------------------------------
# c8: representative return-value pins (hardcoded literals)
# ---------------------------------------------------------------------------
def test_format_memory_date_representative(deterministic_locale):
    assert ss._format_memory_date("2026-04-14T09:05:00") == "April 14, 2026 — 9:05 AM"


def test_format_session_time_representative(deterministic_locale):
    assert ss._format_session_time("2026-04-14T09:05:00") == "April 14, 2026 at 9:05am"


def test_format_full_transcript_block_representative(deterministic_locale):
    expected = (
        "## Session from April 14, 2026 at 9:05am\n"
        "  You: Hi\n"
        "  Matt: Hello there"
    )
    assert ss._format_full_transcript_block(SAMPLE_TRANSCRIPT) == expected


def test_format_full_transcript_block_with_suffix(deterministic_locale):
    expected = (
        "## Session from April 14, 2026 at 9:05am (most recent)\n"
        "  You: Hi\n"
        "  Matt: Hello there"
    )
    assert ss._format_full_transcript_block(
        SAMPLE_TRANSCRIPT, header_suffix=" (most recent)"
    ) == expected


def test_format_full_transcript_block_empty_turns(deterministic_locale):
    # No turns -> just the header line (join of empty list is "").
    assert ss._format_full_transcript_block(
        {"session_start": "2026-04-14T09:05:00", "turns": []}
    ) == "## Session from April 14, 2026 at 9:05am\n"


def test_load_profile_present(session_state_tmp):
    ss.PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    ss.PROFILE_PATH.write_text("I am Matt.\n")
    assert ss.load_profile() == "I am Matt.\n"


def test_load_memory_present(session_state_tmp):
    ss.MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    ss.MEMORY_PATH.write_text("# Memory\n\nstuff\n")
    assert ss.load_memory() == "# Memory\n\nstuff\n"


# ---------------------------------------------------------------------------
# c9: branch/edge coverage — each required branch pair
# ---------------------------------------------------------------------------
# (1) load_profile: exists vs absent
def test_load_profile_absent(session_state_tmp):
    assert not ss.PROFILE_PATH.exists()
    assert ss.load_profile() == ""


def test_load_profile_exists_branch(session_state_tmp):
    ss.PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    ss.PROFILE_PATH.write_text("hello")
    assert ss.load_profile() == "hello"


# (2) load_memory: absent vs present
def test_load_memory_absent(session_state_tmp):
    assert not ss.MEMORY_PATH.exists()
    assert ss.load_memory() == ""


def test_load_memory_present_branch(session_state_tmp):
    ss.MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    ss.MEMORY_PATH.write_text("content here")
    assert ss.load_memory() == "content here"


# (3) append_to_memory: file-created (header seeded) vs append-only (no re-header)
def test_append_to_memory_creates_file_with_header(session_state_tmp, deterministic_locale):
    ss.MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    assert not ss.MEMORY_PATH.exists()
    ss.append_to_memory(SAMPLE_TRANSCRIPT, "- discussed X")
    text = ss.MEMORY_PATH.read_text()
    # Seeded header IS written.
    assert text.startswith("# Memory — what we've discussed\n\n")
    assert "One section per session, append-only." in text
    # Dated section header + entry.
    assert "## April 14, 2026 — 9:05 AM\n" in text
    assert text.endswith("- discussed X\n\n")


def test_append_to_memory_append_only_no_reheader(session_state_tmp, deterministic_locale):
    ss.MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Pre-existing memory file (already has a header of some form).
    ss.MEMORY_PATH.write_text("PREEXISTING\n")
    ss.append_to_memory(SAMPLE_TRANSCRIPT, "  spaced entry  ")
    text = ss.MEMORY_PATH.read_text()
    # The seeded header is NOT re-written; original content preserved, new appended.
    assert text.startswith("PREEXISTING\n")
    assert "# Memory — what we've discussed" not in text
    # summary_text is stripped; header uses memory-date format.
    assert text == "PREEXISTING\n## April 14, 2026 — 9:05 AM\nspaced entry\n\n"


# (4) load_most_recent_transcript_block: missing dir / no eligible files / populated
def test_load_most_recent_missing_dir(session_state_tmp):
    assert not ss.TRANSCRIPTS_DIR.exists()
    assert ss.load_most_recent_transcript_block() is None


def test_load_most_recent_no_eligible_files(session_state_tmp):
    ss.TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    # Only a .usage.json file (excluded by the filter) -> None.
    (ss.TRANSCRIPTS_DIR / "2026-04-14.usage.json").write_text("{}")
    assert ss.load_most_recent_transcript_block() is None


def test_load_most_recent_selects_newest_and_excludes_usage(session_state_tmp, deterministic_locale):
    ss.TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    older = {
        "session_start": "2026-04-10T08:00:00",
        "turns": [{"role": "user", "content": "old"}],
    }
    newer = SAMPLE_TRANSCRIPT
    (ss.TRANSCRIPTS_DIR / "2026-04-10-080000.json").write_text(json.dumps(older))
    (ss.TRANSCRIPTS_DIR / "2026-04-14-090500.json").write_text(json.dumps(newer))
    # A .usage.json that sorts last but must be excluded.
    (ss.TRANSCRIPTS_DIR / "2026-04-20.usage.json").write_text(json.dumps(newer))

    result = ss.load_most_recent_transcript_block()
    expected = (
        "## Session from April 14, 2026 at 9:05am (most recent)\n"
        "  You: Hi\n"
        "  Matt: Hello there"
    )
    assert result == expected


# (5)+(6) noon/midnight boundary of `hour % 12 or 12`
def test_format_memory_date_midnight(deterministic_locale):
    assert ss._format_memory_date("2026-01-01T00:07:00") == "January 1, 2026 — 12:07 AM"


def test_format_memory_date_noon(deterministic_locale):
    assert ss._format_memory_date("2026-12-25T12:30:00") == "December 25, 2026 — 12:30 PM"


def test_format_memory_date_pm(deterministic_locale):
    assert ss._format_memory_date("2026-07-04T15:09:00") == "July 4, 2026 — 3:09 PM"


def test_format_session_time_midnight(deterministic_locale):
    assert ss._format_session_time("2026-01-01T00:07:00") == "January 1, 2026 at 12:07am"


def test_format_session_time_noon(deterministic_locale):
    assert ss._format_session_time("2026-12-25T12:30:00") == "December 25, 2026 at 12:30pm"


def test_format_session_time_pm(deterministic_locale):
    assert ss._format_session_time("2026-07-04T15:09:00") == "July 4, 2026 at 3:09pm"


# Error-path: malformed iso timestamp raises ValueError (current behavior).
def test_format_memory_date_bad_iso_raises():
    with pytest.raises(ValueError):
        ss._format_memory_date("not-a-timestamp")


def test_format_session_time_bad_iso_raises():
    with pytest.raises(ValueError):
        ss._format_session_time("not-a-timestamp")


# append_to_memory / _format_full_transcript_block error path on missing keys.
def test_append_to_memory_missing_session_start_raises(session_state_tmp):
    with pytest.raises(KeyError):
        ss.append_to_memory({"turns": []}, "text")


def test_format_full_transcript_block_missing_turns_raises(deterministic_locale):
    with pytest.raises(KeyError):
        ss._format_full_transcript_block({"session_start": "2026-04-14T09:05:00"})


# ---------------------------------------------------------------------------
# c10: dual-import interface proof over CASES
# ---------------------------------------------------------------------------
_PURE_FS_FREE = {"_format_memory_date", "_format_session_time", "_format_full_transcript_block"}


def _invoke(fn, args):
    return fn(*args)


@pytest.mark.parametrize("helper", _PURE_FS_FREE)
def test_dual_import_pure_helpers(helper, imported_bot, deterministic_locale):
    import session_state
    for args in CASES[helper]:
        bot_fn = getattr(imported_bot, helper)
        ss_fn = getattr(session_state, helper)
        assert bot_fn(*args) == ss_fn(*args)


def test_dual_import_load_profile(imported_bot, session_state_tmp):
    import session_state
    ss.PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    ss.PROFILE_PATH.write_text("profile body")
    # Both import paths observe the patched session_state.PROFILE_PATH.
    assert imported_bot.load_profile() == "profile body"
    assert session_state.load_profile() == "profile body"
    assert imported_bot.load_profile() == session_state.load_profile()


def test_dual_import_load_memory(imported_bot, session_state_tmp):
    import session_state
    ss.MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    ss.MEMORY_PATH.write_text("mem body")
    assert imported_bot.load_memory() == session_state.load_memory() == "mem body"


def test_dual_import_load_most_recent(imported_bot, session_state_tmp, deterministic_locale):
    import session_state
    ss.TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    (ss.TRANSCRIPTS_DIR / "2026-04-14-090500.json").write_text(json.dumps(SAMPLE_TRANSCRIPT))
    assert imported_bot.load_most_recent_transcript_block() == \
        session_state.load_most_recent_transcript_block()


def test_dual_import_append_to_memory(imported_bot, session_state_tmp, deterministic_locale):
    import session_state
    ss.MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Call via bot path.
    imported_bot.append_to_memory(SAMPLE_TRANSCRIPT, "via bot")
    text_after_bot = ss.MEMORY_PATH.read_text()
    # Call via session_state path (append).
    session_state.append_to_memory(SAMPLE_TRANSCRIPT, "via ss")
    text_after_ss = ss.MEMORY_PATH.read_text()
    assert text_after_ss.startswith(text_after_bot)
    assert "via bot" in text_after_ss and "via ss" in text_after_ss


def test_dual_import_patched_const_observed_by_both(imported_bot, session_state_tmp):
    """Both import paths' HELPERS read session_state's patched module globals.

    The moved functions close over session_state's namespace (their __globals__),
    so patching session_state.PROFILE_PATH is observed identically whether the
    helper is reached via bot.load_profile or session_state.load_profile — proving
    the relocation preserved the global-resolution behavior across both paths.
    """
    import session_state

    # The re-exported helper objects are identical (c6).
    assert imported_bot.load_profile is session_state.load_profile
    # And they resolve the PATCHED session_state global, not a stale copy.
    ss.PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    ss.PROFILE_PATH.write_text("patched-observed")
    assert imported_bot.load_profile() == "patched-observed"
    assert session_state.load_profile() == "patched-observed"
    # session_state.PROFILE_PATH is the patched tmp path.
    assert session_state.PROFILE_PATH == session_state_tmp / "profile.md"
