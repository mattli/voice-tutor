"""The telemetry gate keys on USER utterances only — a session is analyzed/
summarized iff the user spoke at least VOICE_TUTOR_MIN_USER_TURNS times. The
tutor's hidden kickoff and its spoken replies (role='assistant') must NOT count,
so a session where only the opener fired is skipped."""

import session_state as ss


def test_count_user_turns_counts_only_user_role():
    turns = [
        {"role": "assistant", "content": "opener"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "reply"},
    ]
    assert ss.count_user_turns(turns) == 1


def test_kickoff_only_has_zero_user_turns_and_is_skipped():
    # Only the tutor's opener recorded; the user never spoke.
    turns = [{"role": "assistant", "content": "opener"}]
    assert ss.count_user_turns(turns) == 0
    assert ss.has_min_user_turns(turns, 1) is False


def test_user_spoke_once_meets_default_floor():
    # John's session shape: opener → one user utterance → reply.
    turns = [
        {"role": "assistant", "content": "opener"},
        {"role": "user", "content": "can you give me a different overview?"},
        {"role": "assistant", "content": "sure"},
    ]
    assert ss.has_min_user_turns(turns, 1) is True


def test_minimum_is_respected():
    turns = [{"role": "user", "content": "hello"}]  # one user utterance
    assert ss.has_min_user_turns(turns, 1) is True
    assert ss.has_min_user_turns(turns, 2) is False


def test_empty_turns_is_skipped():
    assert ss.has_min_user_turns([], 1) is False


# session_expects_summary: the /telemetry poll's expectation must be computed from
# the SAME gate bot.py writes by, so the frontend never waits for a summary that
# will never be produced (an unsatisfiable poll caps at 60s).

import json


def test_expects_summary_true_when_user_spoke(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"turns": [{"role": "assistant"}, {"role": "user", "content": "hi"}]}))
    assert ss.session_expects_summary(p, 1) is True


def test_expects_summary_false_when_only_kickoff_fired(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"turns": [{"role": "assistant", "content": "opener"}]}))
    assert ss.session_expects_summary(p, 1) is False


def test_expects_summary_false_when_transcript_missing(tmp_path):
    # user never spoke → save_transcript wrote no transcript → nothing is generated
    assert ss.session_expects_summary(tmp_path / "missing.json", 1) is False
