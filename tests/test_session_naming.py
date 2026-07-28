"""Hermetic tests for the pure session_naming module.

session_naming.session_analysis_filename builds the on-disk name for a
session-analysis markdown file — date-first (so the folder sorts chronologically)
with an 8-char shortid of the UUID session_id (so the file still joins to
cost-log.jsonl rows and ~/.voice-tutor/artifacts/<full-uuid>.md). It's Pipecat-free
and derives the date from the session's actual START time, not the write time, so
it is tested here with plain datetimes (no bot.py / pipecat / network / filesystem).
"""

from datetime import datetime

from session_naming import session_analysis_filename


def test_uuid_session_id_yields_date_first_name_with_shortid():
    start = datetime(2026, 7, 22, 16, 14, 28)
    fn = session_analysis_filename(start, "53a8c8db-8afb-44b0-ba7c-4a4b8c57d037")
    assert fn == "session-analysis-2026-07-22-161428-53a8c8db.md"


def test_shortid_is_first_8_chars_of_session_id():
    start = datetime(2026, 7, 27, 16, 34, 41)
    fn = session_analysis_filename(start, "7beee170-47d1-418d-b35d-5cc56babaccc")
    # shortid preserves the join to cost-log rows / artifacts/<full-uuid>.md
    assert fn == "session-analysis-2026-07-27-163441-7beee170.md"
    assert "7beee170" in fn


def test_no_session_id_degrades_to_bare_timestamp():
    # Legacy non-study path: the id is itself the start timestamp, so no shortid.
    start = datetime(2026, 6, 1, 10, 6, 53)
    fn = session_analysis_filename(start, None)
    assert fn == "session-analysis-2026-06-01-100653.md"


def test_empty_session_id_degrades_to_bare_timestamp():
    start = datetime(2026, 6, 1, 10, 6, 53)
    assert session_analysis_filename(start, "") == "session-analysis-2026-06-01-100653.md"


def test_uses_start_time_not_write_time():
    # Two different start times must produce two different names regardless of
    # when the file is actually written.
    a = session_analysis_filename(datetime(2026, 5, 13, 14, 31, 7), "f1dd0ac0-1234")
    b = session_analysis_filename(datetime(2026, 5, 13, 14, 32, 7), "f1dd0ac0-1234")
    assert a == "session-analysis-2026-05-13-143107-f1dd0ac0.md"
    assert b == "session-analysis-2026-05-13-143207-f1dd0ac0.md"
    assert a != b


def test_midnight_zero_padding():
    start = datetime(2026, 1, 2, 0, 0, 0)
    fn = session_analysis_filename(start, "abcd1234-ef")
    assert fn == "session-analysis-2026-01-02-000000-abcd1234.md"
