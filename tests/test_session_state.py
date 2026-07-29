"""Sprint 2 — per-user transcript & memory isolation on a SHARED document.

A shared demo doc lives once in documents/_shared/, but ALL per-user state
about it — transcripts and memory included — stays per-user. session_state's
helpers key on ``user_id`` via the storage path
(``TRANSCRIPTS_DIR/<user_id>/`` and ``MEMORY_DIR/<user_id>.md``), so two users
studying the SAME shared doc id never read each other's transcript or memory.

These tests target the PURE session_state helpers directly (no TestClient, no
pipecat, no HTTP). They use conftest's ``session_state_tmp`` fixture, which
monkeypatches session_state's module-level Path constants to a per-test tmp
root and guards the real ~/.voice-tutor is never touched.
"""

import json

import session_state as ss

# One shared document id, studied by BOTH users below. session_state keys its
# transcript/memory storage on user_id, so this shared doc's transcripts and
# memory still land in distinct per-user paths — the collision this proves.
_SHARED_DOC = "shared-demo-doc"


def _write_transcript(root, *, user_id, session_id, session_start, turns):
    """Write one transcript JSON under TRANSCRIPTS_DIR/<user_id>/<session_id>.json.

    Mirrors the on-disk layout ``load_most_recent_transcript_block`` reads:
    per-user subdirectory, ``*.json`` (non ``.usage.json``) files.
    """
    user_dir = root / "transcripts" / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    payload = {"session_start": session_start, "document_id": _SHARED_DOC, "turns": turns}
    (user_dir / f"{session_id}.json").write_text(json.dumps(payload))


def test_shared_doc_transcript_block_is_per_user(session_state_tmp):
    """Two users studied the SAME shared doc id → each reads only their own
    most-recent transcript block, never the other's."""
    root = session_state_tmp

    _write_transcript(
        root, user_id="matt", session_id="sa",
        session_start="2026-07-25T10:00:00",
        turns=[{"role": "assistant", "content": "ALPHA-line for matt"}],
    )
    _write_transcript(
        root, user_id="sarah", session_id="sb",
        session_start="2026-07-26T11:00:00",
        turns=[{"role": "assistant", "content": "BRAVO-line for sarah"}],
    )

    a_block = ss.load_most_recent_transcript_block("matt")
    b_block = ss.load_most_recent_transcript_block("sarah")

    assert a_block is not None and "ALPHA-line for matt" in a_block
    assert "BRAVO-line for sarah" not in a_block

    assert b_block is not None and "BRAVO-line for sarah" in b_block
    assert "ALPHA-line for matt" not in b_block


def test_shared_doc_transcript_none_for_user_without_sessions(session_state_tmp):
    """A third user with no transcripts on the shared doc gets None, even though
    A and B both have transcripts for that same shared doc id."""
    root = session_state_tmp
    _write_transcript(
        root, user_id="matt", session_id="sa",
        session_start="2026-07-25T10:00:00",
        turns=[{"role": "assistant", "content": "ALPHA-line for matt"}],
    )
    _write_transcript(
        root, user_id="sarah", session_id="sb",
        session_start="2026-07-26T11:00:00",
        turns=[{"role": "assistant", "content": "BRAVO-line for sarah"}],
    )

    assert ss.load_most_recent_transcript_block("dev") is None


def test_shared_doc_memory_is_per_user(session_state_tmp):
    """Memory appended while studying the SAME shared doc id stays per-user:
    each user reads only their own memory file, never the other's."""
    transcript_a = {"session_start": "2026-07-25T10:00:00"}
    transcript_b = {"session_start": "2026-07-26T11:00:00"}

    ss.append_to_memory("matt", transcript_a, "ALPHA-memory only matt studied shared-demo-doc")
    ss.append_to_memory("sarah", transcript_b, "BRAVO-memory only sarah studied shared-demo-doc")

    a_mem = ss.load_memory("matt")
    b_mem = ss.load_memory("sarah")

    assert "ALPHA-memory" in a_mem and "BRAVO-memory" not in a_mem
    assert "BRAVO-memory" in b_mem and "ALPHA-memory" not in b_mem

    # A user who never studied the shared doc has no memory at all.
    assert ss.load_memory("dev") == ""


def test_shared_doc_memory_files_are_distinct_paths(session_state_tmp):
    """Path-level proof: the two users' memory files are separate files under
    MEMORY_DIR keyed by user_id, so the shared doc cannot alias them together."""
    transcript = {"session_start": "2026-07-25T10:00:00"}
    ss.append_to_memory("matt", transcript, "ALPHA-memory")
    ss.append_to_memory("sarah", transcript, "BRAVO-memory")

    matt_path = ss.memory_path("matt")
    sarah_path = ss.memory_path("sarah")
    assert matt_path != sarah_path
    assert "ALPHA-memory" in matt_path.read_text()
    assert "ALPHA-memory" not in sarah_path.read_text()
    assert "BRAVO-memory" in sarah_path.read_text()
