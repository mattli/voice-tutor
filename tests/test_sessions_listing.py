"""Hermetic tests for the pure sessions.list_study_sessions() helper.

These tests target the PURE HELPER ONLY. They MUST NOT import app.py, import
pipecat, or construct a TestClient — they exercise sessions.py directly, which
is Pipecat-free (imports only ``documents`` for title resolution).

Fixtures (conftest.py):
  - ``cost_log_tmp`` monkeypatches ``sessions.SESSION_LOG_JSONL_PATH`` to a per-test
    tmp ledger and guards the real vault cost-log is never mutated.
  - ``docs_dir`` monkeypatches ``documents.DOCUMENTS_DIR`` to a per-test tmp dir
    and guards the real documents dir is never mutated. We seed ``<doc_id>.txt``
    directly so ``documents.load_document`` resolves a title.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

import sessions

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _write_ledger(path, rows):
    """Write an iterable of already-serialized JSONL lines (strings) to ``path``."""
    path.write_text("".join(line if line.endswith("\n") else line + "\n" for line in rows))


def _study_row(session_id, session_start, document_id, duration=480, cost=1.39, user_id="matt"):
    return json.dumps(
        {
            "kind": "session",
            "mode": "study",
            "session_id": session_id,
            "session_start": session_start,
            "session_end": session_start,
            "session_duration_sec": duration,
            "cost_total_usd": cost,
            "document_id": document_id,
            "user_id": user_id,
        }
    )


def _seed_doc(docs_dir, doc_id, title, user_id="matt"):
    """Materialize a document so documents.load_document(user_id, doc_id) resolves ``title``.

    Documents are namespaced per-user under ``docs_dir/<user_id>/`` (documents.py
    Task 7), so the seeded file must land in the same user's subdirectory that
    ``list_study_sessions`` will look it up under.
    """
    user_docs_dir = docs_dir / user_id
    user_docs_dir.mkdir(parents=True, exist_ok=True)
    (user_docs_dir / f"{doc_id}.txt").write_text(f"# {title}\nbody text")


def _seed_session(path, *, session_id, document_id, user_id, session_start="2026-02-10T12:00:00"):
    """Append one valid study-session row stamped with ``user_id`` to the ledger at ``path``."""
    row = json.dumps(
        {
            "kind": "session",
            "mode": "study",
            "session_id": session_id,
            "session_start": session_start,
            "session_end": session_start,
            "session_duration_sec": 480,
            "cost_total_usd": 1.39,
            "document_id": document_id,
            "user_id": user_id,
        }
    )
    with path.open("a") as f:
        f.write(row + "\n")


# --------------------------------------------------------------------------- #
# c1 — module is Pipecat-free                                                  #
# --------------------------------------------------------------------------- #


def test_module_surface():
    assert callable(sessions.list_study_sessions)
    assert hasattr(sessions, "SESSION_LOG_JSONL_PATH")
    # Default path is the expanduser-resolved vault session-log.jsonl.
    assert sessions.SESSION_LOG_JSONL_PATH == (
        Path.home()
        / "second-brain"
        / "products"
        / "voice-tutor"
        / "validation"
        / "session-log.jsonl"
    )


def test_import_is_pipecat_free():
    """A fresh `import sessions` in a clean interpreter must NOT pull in
    pipecat/bot/app/fastapi. Checked in a subprocess so the assertion is
    independent of whatever earlier tests loaded into this process's sys.modules.
    ``documents`` (and anthropic/pypdf) ARE allowed and intentionally not checked.
    """
    code = (
        "import sys; import sessions; "
        "bad=[m for m in ('pipecat','bot','app','fastapi') if m in sys.modules]; "
        "assert not bad, bad; print('OK')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


# --------------------------------------------------------------------------- #
# c9 — user_id scope + mirror image                                           #
# --------------------------------------------------------------------------- #


def test_list_scoped_to_user_and_mirror_image(cost_log_tmp, docs_dir):
    # Materialize docs so titles resolve (existing helper pattern in this file).
    _seed_doc(docs_dir, "doc-a", "A", user_id="matt")
    _seed_doc(docs_dir, "doc-b", "B", user_id="sarah")
    _seed_session(cost_log_tmp, session_id="sa", document_id="doc-a", user_id="matt")
    _seed_session(cost_log_tmp, session_id="sb", document_id="doc-b", user_id="sarah")

    matt = sessions.list_study_sessions("matt")
    sarah = sessions.list_study_sessions("sarah")
    dev = sessions.list_study_sessions("dev")

    assert [r["session_id"] for r in matt] == ["sa"]
    assert [r["session_id"] for r in sarah] == ["sb"]
    assert dev == []  # mirror image: a user with no sessions sees nothing


# --------------------------------------------------------------------------- #
# c7 — empty and absent ledger                                                 #
# --------------------------------------------------------------------------- #


def test_absent_ledger_yields_empty(cost_log_tmp):
    # cost_log_tmp points at a path that does not exist yet.
    assert not cost_log_tmp.exists()
    assert sessions.list_study_sessions("matt") == []


def test_empty_ledger_yields_empty(cost_log_tmp):
    cost_log_tmp.write_text("")
    assert sessions.list_study_sessions("matt") == []


# --------------------------------------------------------------------------- #
# c2 — newest-first ordering + exact field set + no reformatting              #
# --------------------------------------------------------------------------- #


def test_newest_first_ordering_and_fields(cost_log_tmp, docs_dir):
    _seed_doc(docs_dir, "doc-a", "Alpha")
    _seed_doc(docs_dir, "doc-b", "Beta")
    _seed_doc(docs_dir, "doc-c", "Gamma")
    # Append order deliberately differs from session_start-descending order.
    _write_ledger(
        cost_log_tmp,
        [
            _study_row("s-mid", "2026-02-10T12:00:00", "doc-b", duration=300, cost=0.50),
            _study_row("s-old", "2026-01-01T09:00:00", "doc-a", duration=480, cost=1.39),
            _study_row("s-new", "2026-03-15T18:30:00", "doc-c", duration=600, cost=2.10),
        ],
    )
    result = sessions.list_study_sessions("matt")
    assert [r["session_id"] for r in result] == ["s-new", "s-mid", "s-old"]

    expected_keys = {
        "session_id",
        "document_title",
        "session_start",
        "session_duration_sec",
        "cost_total_usd",
    }
    for row in result:
        assert set(row.keys()) == expected_keys

    newest = result[0]
    # session_start passed through as the raw ISO string (no humanization).
    assert newest["session_start"] == "2026-03-15T18:30:00"
    # numeric fields stay numeric (no "$1.39" strings, no "8m" durations).
    assert newest["session_duration_sec"] == 600
    assert isinstance(newest["session_duration_sec"], int)
    assert newest["cost_total_usd"] == 2.10
    assert isinstance(newest["cost_total_usd"], float)
    assert newest["document_title"] == "Gamma"


def test_equal_session_start_ties_do_not_raise(cost_log_tmp, docs_dir):
    _seed_doc(docs_dir, "doc-a", "Alpha")
    _seed_doc(docs_dir, "doc-b", "Beta")
    _write_ledger(
        cost_log_tmp,
        [
            _study_row("s-1", "2026-02-10T12:00:00", "doc-a"),
            _study_row("s-2", "2026-02-10T12:00:00", "doc-b"),
        ],
    )
    result = sessions.list_study_sessions("matt")
    assert {r["session_id"] for r in result} == {"s-1", "s-2"}


# --------------------------------------------------------------------------- #
# c3 — study-only filtering                                                    #
# --------------------------------------------------------------------------- #


def test_filtering_excludes_non_study_and_docless_and_artifact(cost_log_tmp, docs_dir):
    _seed_doc(docs_dir, "doc-a", "Alpha")
    rows = [
        # (a) valid study row
        _study_row("s-valid", "2026-02-10T12:00:00", "doc-a"),
        # (b) open-chat row: mode != study
        json.dumps(
            {
                "kind": "session",
                "mode": "open-chat",
                "session_id": "s-openchat",
                "session_start": "2026-02-11T12:00:00",
                "session_duration_sec": 100,
                "cost_total_usd": 0.10,
                "document_id": "doc-a",
            }
        ),
        # (b2) missing mode entirely
        json.dumps(
            {
                "kind": "session",
                "session_id": "s-nomode",
                "session_start": "2026-02-12T12:00:00",
                "session_duration_sec": 100,
                "cost_total_usd": 0.10,
                "document_id": "doc-a",
            }
        ),
        # (c) doc-less study row: null document_id
        json.dumps(
            {
                "kind": "session",
                "mode": "study",
                "session_id": "s-docless-null",
                "session_start": "2026-02-13T12:00:00",
                "session_duration_sec": 100,
                "cost_total_usd": 0.10,
                "document_id": None,
            }
        ),
        # (c2) doc-less study row: missing document_id key
        json.dumps(
            {
                "kind": "session",
                "mode": "study",
                "session_id": "s-docless-missing",
                "session_start": "2026-02-14T12:00:00",
                "session_duration_sec": 100,
                "cost_total_usd": 0.10,
            }
        ),
        # (d) non-session row (kind != session)
        json.dumps({"kind": "turn", "mode": "study", "session_id": "s-turn", "document_id": "doc-a"}),
        # (e) artifact row sharing session_id, valid doc, but lacking output fields
        json.dumps(
            {
                "kind": "artifact",
                "mode": "study",
                "session_id": "s-valid",
                "document_id": "doc-a",
            }
        ),
    ]
    _write_ledger(cost_log_tmp, rows)

    result = sessions.list_study_sessions("matt")
    ids = [r["session_id"] for r in result]
    assert ids == ["s-valid"]
    # doc-less study rows are absent (NOT present with document_title=None here).
    assert "s-docless-null" not in ids
    assert "s-docless-missing" not in ids
    assert "s-openchat" not in ids
    assert "s-nomode" not in ids
    assert "s-turn" not in ids


# --------------------------------------------------------------------------- #
# c4 — document_title resolution mirrors /api/sessions/latest join            #
# --------------------------------------------------------------------------- #


def test_document_title_resolved(cost_log_tmp, docs_dir):
    import documents

    _seed_doc(docs_dir, "doc-x", "The Great Document")
    _write_ledger(cost_log_tmp, [_study_row("s-1", "2026-02-10T12:00:00", "doc-x")])
    result = sessions.list_study_sessions("matt")
    assert len(result) == 1
    assert result[0]["document_title"] == documents.load_document("matt", "doc-x")[0]
    assert result[0]["document_title"] == "The Great Document"


# --------------------------------------------------------------------------- #
# c5 — non-null but unresolvable document_id → row kept, title None            #
# --------------------------------------------------------------------------- #


def test_unresolvable_document_id_kept_with_none_title(cost_log_tmp, docs_dir):
    # docs_dir is empty: doc-missing has no corresponding document.
    docs_dir.mkdir(parents=True, exist_ok=True)
    _write_ledger(cost_log_tmp, [_study_row("s-1", "2026-02-10T12:00:00", "doc-missing")])
    result = sessions.list_study_sessions("matt")
    assert len(result) == 1
    assert result[0]["session_id"] == "s-1"
    assert result[0]["document_title"] is None


# --------------------------------------------------------------------------- #
# c6 — malformed lines skipped                                                 #
# --------------------------------------------------------------------------- #


def test_malformed_lines_skipped(cost_log_tmp, docs_dir):
    _seed_doc(docs_dir, "doc-a", "Alpha")
    cost_log_tmp.write_text(
        "this is not json at all\n"
        "{ broken json ]\n"
        + _study_row("s-good", "2026-02-10T12:00:00", "doc-a")
        + "\n"
    )
    result = sessions.list_study_sessions("matt")
    assert [r["session_id"] for r in result] == ["s-good"]


# --------------------------------------------------------------------------- #
# c8 — path read at call time (monkeypatch takes effect)                       #
# --------------------------------------------------------------------------- #


def test_path_resolved_at_call_time(tmp_path, monkeypatch, docs_dir):
    _seed_doc(docs_dir, "doc-a", "Alpha")
    seeded = tmp_path / "seeded.jsonl"
    _write_ledger(seeded, [_study_row("s-1", "2026-02-10T12:00:00", "doc-a")])

    # Guard the real file is not mutated even though we don't use cost_log_tmp here.
    import sessions as _s

    monkeypatch.setattr(_s, "SESSION_LOG_JSONL_PATH", seeded)
    assert [r["session_id"] for r in _s.list_study_sessions("matt")] == ["s-1"]

    # Re-point at a non-existent path within the same test → call-time read yields [].
    monkeypatch.setattr(_s, "SESSION_LOG_JSONL_PATH", tmp_path / "nope.jsonl")
    assert _s.list_study_sessions("matt") == []


# --------------------------------------------------------------------------- #
# Task 12 — ownership predicate, Matt-only gate, telemetry redaction          #
# --------------------------------------------------------------------------- #


def test_session_belongs_to(cost_log_tmp):
    _seed_session(cost_log_tmp, session_id="sa", document_id="d", user_id="matt")
    assert sessions.session_belongs_to("matt", "sa") is True
    assert sessions.session_belongs_to("sarah", "sa") is False
    assert sessions.session_belongs_to("matt", "missing") is False


def _seed_transcript(user_id, session_id):
    """Write a transcript where bot.py's teardown writes it (first, per-user)."""
    d = sessions.TRANSCRIPTS_DIR / user_id
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{session_id}.json").write_text('{"turns": []}')


def test_ownership_holds_BEFORE_the_ledger_row_is_written(cost_log_tmp):
    # The ledger row is the LAST thing teardown writes — after the coverage
    # judge, the summary, and the analysis. Keying ownership on it made the
    # /telemetry composite 404 for the whole of teardown, so the ended view
    # polled 60s and reported "Recap didn't generate" while the recap had been
    # on disk since ~5s (observed 2026-08-04). The transcript lands first and is
    # itself the ownership fact.
    _seed_transcript("matt", "in-flight")
    assert sessions.session_belongs_to("matt", "in-flight") is True


def test_a_transcript_proves_ownership_only_for_ITS_OWN_user(cost_log_tmp):
    _seed_transcript("matt", "s1")
    assert sessions.session_belongs_to("sarah", "s1") is False


def test_a_crafted_session_id_cannot_probe_another_users_transcript(cost_log_tmp):
    # The probe path is built from the AUTHENTICATED user_id, and the session id
    # is collapsed to one component, so traversal can't answer the question
    # about someone else's file.
    #
    # SARAH'S OWN DIRECTORY MUST EXIST or this test passes vacuously: pathlib +
    # os.stat only traverse ".." through a directory that is really there, so
    # with no transcripts/sarah/ the probe returns False whether or not the guard
    # is present, and the assertion proves nothing (CLAUDE.md, "a traversal test
    # must materialize it"). Verified 2026-08-04: with the guard removed and this
    # seed line deleted the test still passes; with the seed line it fails.
    _seed_transcript("matt", "secret")
    _seed_transcript("sarah", "own-session")
    assert sessions.session_belongs_to("sarah", "../matt/secret") is False


def test_the_ledger_still_answers_when_no_transcript_exists(cost_log_tmp):
    # Fallback intact: a session whose transcript was never written (user never
    # spoke) or has since been archived is still owned per the ledger.
    _seed_session(cost_log_tmp, session_id="no-tx", document_id="d", user_id="matt")
    assert sessions.session_belongs_to("matt", "no-tx") is True
    assert sessions.session_belongs_to("sarah", "no-tx") is False


# --------------------------------------------------------------------------- #
# Sprint 2 — per-user session listing on a SHARED document id                  #
#                                                                              #
# A shared demo doc has one id but per-user state. list_study_sessions must    #
# stay keyed on user_id: two users A and B who both studied the SAME shared    #
# doc id must each see only their own session record, never the other's.       #
# --------------------------------------------------------------------------- #

_SHARED_DOC = "shared-demo-doc"


def test_list_study_sessions_shared_doc_is_per_user(cost_log_tmp, docs_dir):
    # Both users study the SAME shared doc id (the collision the shared
    # namespace makes possible). Seed a per-user doc so a title resolves for
    # each — documents remain per-user regardless of the shared session key.
    _seed_doc(docs_dir, _SHARED_DOC, "Shared Demo", user_id="matt")
    _seed_doc(docs_dir, _SHARED_DOC, "Shared Demo", user_id="sarah")
    _seed_session(cost_log_tmp, session_id="sa", document_id=_SHARED_DOC, user_id="matt")
    _seed_session(cost_log_tmp, session_id="sb", document_id=_SHARED_DOC, user_id="sarah")

    matt = sessions.list_study_sessions("matt")
    sarah = sessions.list_study_sessions("sarah")

    matt_ids = [r["session_id"] for r in matt]
    sarah_ids = [r["session_id"] for r in sarah]

    # matt sees only his session on the shared doc; sarah's is excluded.
    assert matt_ids == ["sa"]
    assert "sb" not in matt_ids
    # Mirror image: sarah sees only hers; matt's is excluded.
    assert sarah_ids == ["sb"]
    assert "sa" not in sarah_ids

    # A user with no sessions on the shared doc sees nothing at all.
    assert sessions.list_study_sessions("dev") == []


def test_history_keeps_the_title_after_the_document_is_archived(cost_log_tmp, docs_dir):
    # Removing a document from the picker does not unhappen the sessions run
    # against it, so their rows must not degrade to an untitled entry.
    import documents

    _seed_doc(docs_dir, "doc-arch", "Graph Engineering", user_id="matt")
    _seed_session(cost_log_tmp, session_id="s1", document_id="doc-arch", user_id="matt")
    assert sessions.list_study_sessions("matt")[0]["document_title"] == "Graph Engineering"

    documents.archive_document("matt", "doc-arch")
    assert sessions.list_study_sessions("matt")[0]["document_title"] == "Graph Engineering"


def test_can_view_machine_artifacts_matt_only():
    # Mirror image: a non-matt user is denied prompt/analysis/cost-log surfaces.
    assert sessions.can_view_machine_artifacts("matt") is True
    assert sessions.can_view_machine_artifacts("sarah") is False


def test_redact_telemetry_strips_matt_only_fields_for_non_matt():
    full = {"recap": "r", "cost": {"x": 1}, "memory_append": "m",
            "analysis": "AAA", "has_prompt": True, "document_title": "T"}
    # Matt sees everything.
    assert sessions.redact_telemetry_for_user(full, "matt") == full
    # Sarah's composite must not carry the analysis or a prompt reference.
    red = sessions.redact_telemetry_for_user(full, "sarah")
    assert red["analysis"] is None and red["has_prompt"] is False
    # Her own learning artifacts survive.
    assert red["recap"] == "r" and red["cost"] == {"x": 1} and red["memory_append"] == "m"
