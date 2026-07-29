"""Hermetic tests for the pure session_naming module.

session_naming.session_analysis_filename builds the on-disk name for a
session-analysis markdown file — date-first (so the folder sorts chronologically)
with an 8-char shortid of the UUID session_id (so the file still joins to
session-log.jsonl rows and ~/.voice-tutor/artifacts/<full-uuid>.md). It's Pipecat-free
and derives the date from the session's actual START time, not the write time, so
it is tested here with plain datetimes (no bot.py / pipecat / network / filesystem).
"""

from datetime import datetime

import session_naming as sn
from session_naming import find_analysis_path, session_analysis_filename


def test_uuid_session_id_yields_date_first_name_with_shortid():
    start = datetime(2026, 7, 22, 16, 14, 28)
    fn = session_analysis_filename(start, "53a8c8db-8afb-44b0-ba7c-4a4b8c57d037")
    assert fn == "session-analysis-2026-07-22-161428-53a8c8db.md"


def test_shortid_is_first_8_chars_of_session_id():
    start = datetime(2026, 7, 27, 16, 34, 41)
    fn = session_analysis_filename(start, "7beee170-47d1-418d-b35d-5cc56babaccc")
    # shortid preserves the join to session-log rows / artifacts/<full-uuid>.md
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


# --- find_analysis_path: the reader must agree with the writer -----------------
def _write(directory, name):
    p = directory / name
    p.write_text("analysis")
    return p


def _user_dir(tmp_path, user_id="matt"):
    d = tmp_path / user_id
    d.mkdir(exist_ok=True)
    return d


def test_round_trip_builder_name_is_found_by_finder(tmp_path):
    """The load-bearing property: a file written under the builder's name is
    located by the finder from the *same* session id. app.py reads by session id;
    bot.py wrote by session id. A finder-only test could pass while the two
    schemes drift — this pins them together."""
    session_id = "7beee170-47d1-418d-b35d-5cc56babaccc"
    start = datetime(2026, 7, 27, 16, 34, 41)
    user_dir = _user_dir(tmp_path)
    written = _write(user_dir, session_analysis_filename(start, session_id))
    assert find_analysis_path(tmp_path, "matt", session_id) == written


def test_round_trip_selects_the_right_session_among_many(tmp_path):
    # Several sessions' files coexist in the flat folder; the finder returns the
    # one whose shortid matches — not a neighbour.
    ids = {
        "7beee170-47d1-418d-b35d-5cc56babaccc": datetime(2026, 7, 27, 16, 34, 41),
        "53a8c8db-8afb-44b0-ba7c-4a4b8c57d037": datetime(2026, 7, 22, 16, 14, 28),
        "f6148c26-af09-491d-b644-1522db9f42c5": datetime(2026, 7, 26, 10, 23, 23),
    }
    user_dir = _user_dir(tmp_path)
    written = {sid: _write(user_dir, session_analysis_filename(st, sid)) for sid, st in ids.items()}
    for sid, path in written.items():
        assert find_analysis_path(tmp_path, "matt", sid) == path


def test_finder_returns_none_when_absent(tmp_path):
    _write(_user_dir(tmp_path), "session-analysis-2026-07-27-163441-7beee170.md")
    assert find_analysis_path(tmp_path, "matt", "deadbeef-0000-0000-0000-000000000000") is None


def test_finder_ignores_legacy_names_without_shortid(tmp_path):
    # Pre-UUID date-only / date+timestamp files carry no shortid and belong to
    # sessions the app never looks up by id — they must not match anything.
    user_dir = _user_dir(tmp_path)
    _write(user_dir, "session-analysis-2026-04-15.md")
    _write(user_dir, "session-analysis-2026-04-17-184001.md")
    assert find_analysis_path(tmp_path, "matt", "2026-04-15") is None
    assert find_analysis_path(tmp_path, "matt", "20260417-xxxx") is None


def test_finder_deterministic_on_shortid_collision(tmp_path):
    # Two ids sharing the first 8 chars (vanishingly unlikely) -> first by sorted
    # name, deterministically, never an exception.
    user_dir = _user_dir(tmp_path)
    a = _write(user_dir, "session-analysis-2026-07-01-090000-7beee170.md")
    _write(user_dir, "session-analysis-2026-07-02-090000-7beee170.md")
    got = find_analysis_path(tmp_path, "matt", "7beee170-aaaa-bbbb-cccc-dddddddddddd")
    assert got == a  # earliest sorted name


def test_finder_guards_against_glob_metacharacters(tmp_path):
    # A non-hex/non-alnum prefix (e.g. glob metachars) must yield None, not an
    # over-broad match against unrelated files.
    _write(_user_dir(tmp_path), "session-analysis-2026-07-27-163441-7beee170.md")
    assert find_analysis_path(tmp_path, "matt", "*") is None
    assert find_analysis_path(tmp_path, "matt", "") is None
    assert find_analysis_path(tmp_path, "matt", "ab*cdef0") is None


def test_find_analysis_path_round_trip_within_user_dir(tmp_path):
    start = datetime(2026, 7, 27, 14, 30, 5)
    sid = "abcd1234-0000-0000-0000-000000000000"
    name = sn.session_analysis_filename(start, sid)      # writer's name
    user_dir = tmp_path / "matt"
    user_dir.mkdir()
    (user_dir / name).write_text("analysis")

    # Reader finds it within matt/; a different user finds nothing (mirror image).
    assert sn.find_analysis_path(tmp_path, "matt", sid) == user_dir / name
    assert sn.find_analysis_path(tmp_path, "sarah", sid) is None


def test_shortid_len_shared_between_writer_and_reader():
    # Guards the drift the 2026-07-27 fix closed: the reader must glob on exactly
    # the shortid the writer embeds.
    start = datetime(2026, 7, 27, 14, 30, 5)
    sid = "abcdef12-9999-0000-0000-000000000000"
    name = sn.session_analysis_filename(start, sid)
    assert f"-{sid[:sn.SHORTID_LEN]}.md" in name
