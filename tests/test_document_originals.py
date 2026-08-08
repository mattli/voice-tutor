"""Hermetic tests for where a document's PRESERVED ORIGINAL lives on disk.

The picker finds documents by globbing ``*.txt`` non-recursively. An original
stored beside the extracted text is therefore invisible to that scan only by
luck of its extension — and a ``.txt`` upload is exactly the case where the luck
runs out, listing one document twice under two different ids.

These tests pin the structural property that makes the class of bug impossible:
originals live in a subdirectory the scan cannot reach, and every reader and the
archiver look for them there. Follows CLAUDE.md "test via pure helpers, not
TestClient".
"""

import json

import pytest

import documents


def _upload(docs_dir, user_id, filename, body="# Title\n\nSome body text.\n"):
    return documents.save_upload(user_id, filename, body.encode("utf-8"))


@pytest.fixture(autouse=True)
def _no_summary_calls(monkeypatch):
    """save_upload calls Haiku for a summary; these tests are offline."""
    monkeypatch.setattr(documents, "_generate_summary", lambda text: None)


# --------------------------------------------------------------------------- #
# The bug: a .txt upload listed twice.
# --------------------------------------------------------------------------- #

def test_a_txt_upload_appears_ONCE_in_the_picker(docs_dir):
    # THE REGRESSION TEST. Before the fix this returned 2 entries: the extracted
    # <id>.txt and the preserved <id>-notes.txt, the latter under a different id
    # (its own stem) and with its own wasted summary call.
    result = _upload(docs_dir, "matt", "notes.txt")
    listed = documents._scan_documents(docs_dir / "matt")
    assert len(listed) == 1, f"expected one document, got {[d['document_id'] for d in listed]}"
    assert listed[0]["document_id"] == result["document_id"]


@pytest.mark.parametrize("filename", ["notes.txt", "notes.md", "notes.markdown"])
def test_every_text_extension_yields_exactly_one_document(docs_dir, filename):
    _upload(docs_dir, "matt", filename)
    assert len(documents._scan_documents(docs_dir / "matt")) == 1


def test_the_original_never_lands_in_the_scanned_directory(docs_dir):
    # The structural guarantee, asserted directly rather than through its
    # symptom — this is what makes the fix hold for extensions not yet allowed
    # (a future .rtf or .html cannot reintroduce the phantom).
    _upload(docs_dir, "matt", "notes.txt")
    d = docs_dir / "matt"
    flat = {p.name for p in d.iterdir() if p.is_file()}
    assert not any(n.startswith("notes") for n in flat), flat
    assert (d / documents.ORIGINALS_DIRNAME).is_dir()
    assert [p.name for p in (d / documents.ORIGINALS_DIRNAME).iterdir()] != []


def test_many_txt_uploads_stay_one_document_each(docs_dir):
    ids = {_upload(docs_dir, "matt", f"doc{i}.txt")["document_id"] for i in range(4)}
    listed = documents._scan_documents(docs_dir / "matt")
    assert len(listed) == 4
    assert {d["document_id"] for d in listed} == ids


# --------------------------------------------------------------------------- #
# The original still does its job: filename and upload time.
# --------------------------------------------------------------------------- #

def test_the_filename_fallback_reads_the_original_from_its_new_home(docs_dir):
    # _derive_title prefers the document's own first line and only falls back to
    # the original's FILENAME when the text offers nothing — so the fallback is
    # seeded directly here rather than through save_upload, which refuses a doc
    # with no extractable text. This is the path that silently degrades to
    # "<id>.txt" as a title if the reader cannot find the original.
    d = docs_dir / "matt" / documents.ORIGINALS_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    (docs_dir / "matt" / "doc-1.txt").write_text("   \n\n   \n")
    (d / "doc-1-quarterly-report.md").write_text("original bytes")
    title, _text = documents._load_from_dir(docs_dir / "matt", "doc-1")
    assert title == "quarterly-report", title


def test_the_title_comes_from_the_document_when_it_has_one(docs_dir):
    r = _upload(docs_dir, "matt", "notes.txt", body="# Quarterly Report\n\nBody.\n")
    listed = documents._scan_documents(docs_dir / "matt")[0]
    assert listed["title"] == "Quarterly Report"
    loaded = documents.load_document("matt", r["document_id"])
    assert loaded is not None and loaded[0] == "Quarterly Report"


def test_uploaded_at_is_read_from_the_original(docs_dir):
    r = _upload(docs_dir, "matt", "notes.txt")
    original = documents._original_for(docs_dir / "matt", r["document_id"])
    assert original is not None and original.parent.name == documents.ORIGINALS_DIRNAME
    listed = documents._scan_documents(docs_dir / "matt")[0]
    assert listed["uploaded_at"]


# --------------------------------------------------------------------------- #
# LEGACY flat layout — pre-2026-08-07 documents are not migrated.
# --------------------------------------------------------------------------- #

def _seed_legacy(docs_dir, user_id, doc_id, original="notes.md", title="Graph Engineering"):
    d = docs_dir / user_id
    d.mkdir(parents=True, exist_ok=True)
    text = f"# {title}\n\nBody.\n"
    (d / f"{doc_id}.txt").write_text(text)
    (d / f"{doc_id}-{original}").write_text(text)   # flat, the old layout
    (d / f"{doc_id}.summary.txt").write_text("summary")
    return text


def test_a_legacy_flat_document_still_lists_correctly(docs_dir):
    _seed_legacy(docs_dir, "matt", "legacy-1")
    listed = documents._scan_documents(docs_dir / "matt")
    assert len(listed) == 1
    assert listed[0]["document_id"] == "legacy-1"
    assert listed[0]["title"] == "Graph Engineering"
    assert listed[0]["uploaded_at"]


def test_legacy_and_new_layouts_coexist(docs_dir):
    _seed_legacy(docs_dir, "matt", "legacy-1")
    new_id = _upload(docs_dir, "matt", "fresh.txt")["document_id"]
    ids = {d["document_id"] for d in documents._scan_documents(docs_dir / "matt")}
    assert ids == {"legacy-1", new_id}


def test_a_legacy_original_is_found_by_the_resolver(docs_dir):
    _seed_legacy(docs_dir, "matt", "legacy-1", original="report.md")
    found = documents._original_for(docs_dir / "matt", "legacy-1")
    assert found is not None and found.name == "legacy-1-report.md"
    assert found.parent.name == "matt", "legacy originals stay flat; they are not migrated"


def test_a_phantoms_own_summary_is_not_mistaken_for_the_original(docs_dir):
    # A document duplicated by the old bug left a real sidecar named
    # <id>-<filename>.summary.txt, which matches the legacy glob.
    d = docs_dir / "matt"
    d.mkdir(parents=True, exist_ok=True)
    (d / "old-1.txt").write_text("# T\n\nBody.\n")
    (d / "old-1-notes.txt").write_text("# T\n\nBody.\n")
    (d / "old-1-notes.summary.txt").write_text("phantom summary")
    found = documents._original_for(d, "old-1")
    assert found is not None and found.name == "old-1-notes.txt"


# --------------------------------------------------------------------------- #
# Archive / restore — the call site where a miss fails SILENTLY.
# --------------------------------------------------------------------------- #

def test_archiving_takes_the_original_out_of_originals(docs_dir):
    # If _doc_files misses the subdirectory this passes visually — the document
    # leaves the picker — while the user's original file is left behind.
    doc_id = _upload(docs_dir, "matt", "notes.txt")["document_id"]
    documents.archive_document("matt", doc_id, stamp="2026-08-07-120000")
    live_originals = docs_dir / "matt" / documents.ORIGINALS_DIRNAME
    left_behind = [p.name for p in live_originals.iterdir()] if live_originals.is_dir() else []
    assert left_behind == [], f"original left in the live directory: {left_behind}"
    archived = docs_dir / "matt" / "_archive" / f"{doc_id}-2026-08-07-120000"
    assert (archived / documents.ORIGINALS_DIRNAME / f"{doc_id}-notes.txt").exists()


def test_archive_restore_round_trip_puts_the_original_back(docs_dir):
    doc_id = _upload(docs_dir, "matt", "notes.txt")["document_id"]
    before = (docs_dir / "matt" / documents.ORIGINALS_DIRNAME / f"{doc_id}-notes.txt").read_text()
    documents.archive_document("matt", doc_id, stamp="2026-08-07-120000")
    assert documents._scan_documents(docs_dir / "matt") == []
    documents.restore_document("matt", doc_id)
    restored = docs_dir / "matt" / documents.ORIGINALS_DIRNAME / f"{doc_id}-notes.txt"
    assert restored.exists() and restored.read_text() == before
    listed = documents._scan_documents(docs_dir / "matt")
    assert len(listed) == 1 and listed[0]["document_id"] == doc_id


def test_a_restored_txt_document_still_appears_only_once(docs_dir):
    # The round trip must not flatten the original back beside the text and
    # quietly reintroduce the phantom.
    doc_id = _upload(docs_dir, "matt", "notes.txt")["document_id"]
    documents.archive_document("matt", doc_id, stamp="2026-08-07-120000")
    documents.restore_document("matt", doc_id)
    assert len(documents._scan_documents(docs_dir / "matt")) == 1


def test_archiving_a_legacy_document_still_moves_its_flat_original(docs_dir):
    _seed_legacy(docs_dir, "matt", "legacy-1", original="report.md")
    documents.archive_document("matt", "legacy-1", stamp="2026-08-07-120000")
    remaining = [p.name for p in (docs_dir / "matt").iterdir() if p.is_file()]
    assert remaining == [], f"legacy files left behind: {remaining}"


def test_an_archived_documents_title_survives(docs_dir):
    # resolve_title reads the ARCHIVED copy — history keeps its names.
    r = _upload(docs_dir, "matt", "notes.txt", body="# Quarterly Report\n\nBody.\n")
    documents.archive_document("matt", r["document_id"], stamp="2026-08-07-120000")
    assert documents.load_document("matt", r["document_id"]) is None
    assert documents.resolve_title("matt", r["document_id"]) == "Quarterly Report"


def test_an_archived_documents_filename_fallback_also_survives(docs_dir):
    # The harder half: with no title in the text, resolve_title must find the
    # original INSIDE the archive folder's own _originals/ subdirectory.
    d = docs_dir / "matt"
    (d / documents.ORIGINALS_DIRNAME).mkdir(parents=True, exist_ok=True)
    (d / "doc-1.txt").write_text("   \n")
    (d / documents.ORIGINALS_DIRNAME / "doc-1-quarterly-report.md").write_text("bytes")
    documents.archive_document("matt", "doc-1", stamp="2026-08-07-120000")
    assert documents.resolve_title("matt", "doc-1") == "quarterly-report"


def test_the_originals_directory_is_not_itself_a_document(docs_dir):
    _upload(docs_dir, "matt", "notes.txt")
    ids = [d["document_id"] for d in documents._scan_documents(docs_dir / "matt")]
    assert documents.ORIGINALS_DIRNAME not in ids
