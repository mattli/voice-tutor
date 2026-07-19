"""Hermetic tests for the pure sessions.list_study_sessions() helper.

These tests target the PURE HELPER ONLY. They MUST NOT import app.py, import
pipecat, or construct a TestClient — they exercise sessions.py directly, which
is Pipecat-free (imports only ``documents`` for title resolution).

Fixtures (conftest.py):
  - ``cost_log_tmp`` monkeypatches ``sessions.COST_LOG_JSONL_PATH`` to a per-test
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


def _study_row(session_id, session_start, document_id, duration=480, cost=1.39):
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
        }
    )


def _seed_doc(docs_dir, doc_id, title):
    """Materialize a document so documents.load_document(doc_id) resolves ``title``."""
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / f"{doc_id}.txt").write_text(f"# {title}\nbody text")


# --------------------------------------------------------------------------- #
# c1 — module is Pipecat-free                                                  #
# --------------------------------------------------------------------------- #


def test_module_surface():
    assert callable(sessions.list_study_sessions)
    assert hasattr(sessions, "COST_LOG_JSONL_PATH")
    # Default path is the expanduser-resolved vault cost-log.jsonl.
    assert sessions.COST_LOG_JSONL_PATH == (
        Path.home()
        / "second-brain"
        / "products"
        / "voice-tutor"
        / "validation"
        / "cost-log.jsonl"
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
# c7 — empty and absent ledger                                                 #
# --------------------------------------------------------------------------- #


def test_absent_ledger_yields_empty(cost_log_tmp):
    # cost_log_tmp points at a path that does not exist yet.
    assert not cost_log_tmp.exists()
    assert sessions.list_study_sessions() == []


def test_empty_ledger_yields_empty(cost_log_tmp):
    cost_log_tmp.write_text("")
    assert sessions.list_study_sessions() == []


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
    result = sessions.list_study_sessions()
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
    result = sessions.list_study_sessions()
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

    result = sessions.list_study_sessions()
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
    result = sessions.list_study_sessions()
    assert len(result) == 1
    assert result[0]["document_title"] == documents.load_document("doc-x")[0]
    assert result[0]["document_title"] == "The Great Document"


# --------------------------------------------------------------------------- #
# c5 — non-null but unresolvable document_id → row kept, title None            #
# --------------------------------------------------------------------------- #


def test_unresolvable_document_id_kept_with_none_title(cost_log_tmp, docs_dir):
    # docs_dir is empty: doc-missing has no corresponding document.
    docs_dir.mkdir(parents=True, exist_ok=True)
    _write_ledger(cost_log_tmp, [_study_row("s-1", "2026-02-10T12:00:00", "doc-missing")])
    result = sessions.list_study_sessions()
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
    result = sessions.list_study_sessions()
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

    monkeypatch.setattr(_s, "COST_LOG_JSONL_PATH", seeded)
    assert [r["session_id"] for r in _s.list_study_sessions()] == ["s-1"]

    # Re-point at a non-existent path within the same test → call-time read yields [].
    monkeypatch.setattr(_s, "COST_LOG_JSONL_PATH", tmp_path / "nope.jsonl")
    assert _s.list_study_sessions() == []
