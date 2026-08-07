"""Hermetic tests for archiving a document out of the picker, and undoing it.

Removal is a MOVE, never a delete (the standing rule), which is what makes a
single tap plus an undo the right affordance instead of a confirm modal. These
tests pin the properties that make that safe: nothing is destroyed, an archived
document stops being studyable, a shared document is refused rather than removed
for everyone, and one user can never reach another's namespace.

Follows CLAUDE.md "test via pure helpers, not TestClient" — the app.py routes are
thin wrappers over these functions.
"""

import json

import pytest

import claims
import coverage_store as cs
import documents


def _seed(docs_dir, user_id, doc_id, *, title="Graph Engineering", original="notes.md"):
    """Write a document exactly as save_upload lays it out, plus its sidecars."""
    d = docs_dir / user_id
    d.mkdir(parents=True, exist_ok=True)
    text = f"# {title}\n\nSome body text.\n"
    (d / f"{doc_id}.txt").write_text(text)
    (d / f"{doc_id}-{original}").write_text(text)
    (d / f"{doc_id}.summary.txt").write_text("A one-line summary.")
    (d / f"{doc_id}.claims.json").write_text(json.dumps({"source_hash": "h", "claims": []}))
    return text


def _listed_ids(docs_dir, user_id):
    return [d["document_id"] for d in documents._scan_documents(docs_dir / user_id)]


# --------------------------------------------------------------------------- #
# Archiving.
# --------------------------------------------------------------------------- #

def test_an_archived_document_leaves_the_picker(docs_dir):
    _seed(docs_dir, "matt", "doc-1")
    assert _listed_ids(docs_dir, "matt") == ["doc-1"]
    documents.archive_document("matt", "doc-1")
    assert _listed_ids(docs_dir, "matt") == []


def test_nothing_is_deleted_only_moved(docs_dir):
    # The standing rule: archive, never delete. The undo toast is the fast path;
    # these files are the durable one.
    _seed(docs_dir, "matt", "doc-1")
    result = documents.archive_document("matt", "doc-1")
    archive = docs_dir / "matt" / documents.ARCHIVE_DIRNAME
    landed = sorted(p.name for p in archive.rglob("*") if p.is_file())
    assert landed == sorted([
        "doc-1-notes.md", "doc-1.claims.json", "doc-1.summary.txt", "doc-1.txt",
    ])
    assert sorted(result["files"]) == landed


def test_the_claim_map_travels_with_its_document(docs_dir):
    _seed(docs_dir, "matt", "doc-1")
    documents.archive_document("matt", "doc-1")
    assert not (docs_dir / "matt" / "doc-1.claims.json").exists()


def test_an_archived_document_cannot_be_loaded_for_study(docs_dir):
    # The one thing archiving must actually prevent.
    _seed(docs_dir, "matt", "doc-1")
    documents.archive_document("matt", "doc-1")
    assert documents.load_document("matt", "doc-1") is None


def test_the_archive_folder_is_invisible_to_the_document_scan(docs_dir):
    # The picker globs *.txt non-recursively, so a subdirectory drops out with no
    # filter to keep in sync. Pin it — a scan change could silently resurrect
    # every archived document.
    _seed(docs_dir, "matt", "doc-1")
    _seed(docs_dir, "matt", "doc-2", title="Second")
    documents.archive_document("matt", "doc-1")
    assert _listed_ids(docs_dir, "matt") == ["doc-2"]


def test_archiving_one_document_never_moves_another(docs_dir):
    # A bare `doc-1*` glob would also match doc-10's files.
    _seed(docs_dir, "matt", "doc-1")
    _seed(docs_dir, "matt", "doc-10", title="Ten")
    documents.archive_document("matt", "doc-1")
    assert _listed_ids(docs_dir, "matt") == ["doc-10"]
    assert (docs_dir / "matt" / "doc-10.claims.json").exists()


def test_a_shared_document_is_REFUSED_not_removed_for_everyone(docs_dir):
    # It belongs to every user and there is no app write path to that directory,
    # so one user removing it would silently remove it for all of them.
    _seed(docs_dir, documents.SHARED_USER_ID, "shared-1")
    with pytest.raises(documents.DocumentActionError) as e:
        documents.archive_document("matt", "shared-1")
    assert e.value.status_code == 409
    assert (docs_dir / documents.SHARED_USER_ID / "shared-1.txt").exists()
    assert documents.load_document("matt", "shared-1") is not None


def test_an_unknown_document_is_a_404(docs_dir):
    with pytest.raises(documents.DocumentActionError) as e:
        documents.archive_document("matt", "nope")
    assert e.value.status_code == 404


def test_a_crafted_doc_id_cannot_archive_another_users_document(docs_dir):
    _seed(docs_dir, "victim", "secret")
    (docs_dir / "matt").mkdir(parents=True, exist_ok=True)
    with pytest.raises(documents.DocumentActionError) as e:
        documents.archive_document("matt", "../victim/secret")
    assert e.value.status_code == 404
    assert (docs_dir / "victim" / "secret.txt").exists()


def test_one_user_archiving_never_touches_another_users_copy(docs_dir):
    _seed(docs_dir, "matt", "doc-1")
    _seed(docs_dir, "sarah", "doc-1")
    documents.archive_document("matt", "doc-1")
    assert _listed_ids(docs_dir, "sarah") == ["doc-1"]


# --------------------------------------------------------------------------- #
# Undo.
# --------------------------------------------------------------------------- #

def test_restore_puts_the_document_back(docs_dir):
    _seed(docs_dir, "matt", "doc-1")
    documents.archive_document("matt", "doc-1")
    documents.restore_document("matt", "doc-1")
    assert _listed_ids(docs_dir, "matt") == ["doc-1"]
    assert documents.load_document("matt", "doc-1") is not None


def test_restore_brings_back_every_sidecar_too(docs_dir):
    _seed(docs_dir, "matt", "doc-1")
    documents.archive_document("matt", "doc-1")
    documents.restore_document("matt", "doc-1")
    for name in ("doc-1.txt", "doc-1-notes.md", "doc-1.summary.txt", "doc-1.claims.json"):
        assert (docs_dir / "matt" / name).exists(), name


def test_restore_with_nothing_archived_is_a_404(docs_dir):
    with pytest.raises(documents.DocumentActionError) as e:
        documents.restore_document("matt", "doc-1")
    assert e.value.status_code == 404


def test_restore_refuses_to_overwrite_a_live_document(docs_dir):
    _seed(docs_dir, "matt", "doc-1")
    documents.archive_document("matt", "doc-1")
    _seed(docs_dir, "matt", "doc-1", title="A different document with the same id")
    with pytest.raises(documents.DocumentActionError) as e:
        documents.restore_document("matt", "doc-1")
    assert e.value.status_code == 409
    assert documents.load_document("matt", "doc-1")[0] == "A different document with the same id"


def test_restore_takes_the_MOST_RECENT_archive(docs_dir):
    _seed(docs_dir, "matt", "doc-1", title="First")
    documents.archive_document("matt", "doc-1", stamp="2026-08-01-100000")
    _seed(docs_dir, "matt", "doc-1", title="Second")
    documents.archive_document("matt", "doc-1", stamp="2026-08-02-100000")
    documents.restore_document("matt", "doc-1")
    assert documents.load_document("matt", "doc-1")[0] == "Second"


def test_a_crafted_doc_id_cannot_restore_out_of_the_users_namespace(docs_dir):
    _seed(docs_dir, "victim", "secret")
    documents.archive_document("victim", "secret")
    (docs_dir / "matt").mkdir(parents=True, exist_ok=True)
    with pytest.raises(documents.DocumentActionError) as e:
        documents.restore_document("matt", "../victim/secret")
    assert e.value.status_code == 404


# --------------------------------------------------------------------------- #
# What archiving must NOT break.
# --------------------------------------------------------------------------- #

def test_history_keeps_the_title_of_an_archived_document(docs_dir):
    # A session that really happened should not render as "Unknown document"
    # because the document was later put away.
    _seed(docs_dir, "matt", "doc-1", title="Graph Engineering")
    documents.archive_document("matt", "doc-1")
    assert documents.resolve_title("matt", "doc-1") == "Graph Engineering"


def test_resolving_a_title_does_NOT_make_the_document_studyable_again(docs_dir):
    _seed(docs_dir, "matt", "doc-1")
    documents.archive_document("matt", "doc-1")
    assert documents.resolve_title("matt", "doc-1") is not None
    assert documents.load_document("matt", "doc-1") is None, "study must still miss"


def test_resolve_title_is_None_for_a_document_that_never_existed(docs_dir):
    assert documents.resolve_title("matt", "doc-1") is None


def test_an_archived_documents_coverage_neither_shows_nor_breaks_the_picker(
    docs_dir, tmp_path, monkeypatch
):
    # Coverage sidecars are records of sessions that really happened, so they
    # stay. The picker payload is built from documents that EXIST, so an archived
    # document contributes no bar — and no error either.
    monkeypatch.setattr(cs, "TRANSCRIPTS_DIR", tmp_path / "transcripts")
    text = _seed(docs_dir, "matt", "doc-1")
    (tmp_path / "transcripts" / "matt").mkdir(parents=True)
    (tmp_path / "transcripts" / "matt" / "s1.coverage.json").write_text(json.dumps({
        "session_id": "s1", "user_id": "matt", "document_id": "doc-1",
        "source_hash": claims.source_hash_of(text), "claims_total": 2,
        "verdicts": [{"claim_id": "c1", "covered": True, "turns": [1]}],
    }))

    def identity_for(doc_id):
        loaded = documents.load_document("matt", doc_id)
        if loaded is None:
            return None
        return {"source_hash": claims.source_hash_of(loaded[1]), "claims_total": 2}

    assert "doc-1" in cs.documents_view("matt", identity_for)
    documents.archive_document("matt", "doc-1")
    assert cs.documents_view("matt", identity_for) == {}
    assert (tmp_path / "transcripts" / "matt" / "s1.coverage.json").exists()
