"""Characterization tests for documents.list_documents / load_document / save_upload.

These use the ``docs_dir`` fixture (conftest.py) which monkeypatches
``documents.DOCUMENTS_DIR`` to a per-test tmp_path and proves the real
production documents dir is untouched.

Enumerated list/load cases (counted toward the c8 floor):
  L1  list shape + deterministic order
  L2  list on empty / missing directory
  L3  load a missing document
Plus supporting positive-path characterization of save_upload + load_document.
"""

import asyncio

import pytest

import documents
from documents import list_documents, load_document, save_upload


@pytest.fixture(autouse=True)
def _stub_summary(monkeypatch):
    # Hermetic pin: list_documents/save_upload now backfill per-document summaries
    # via a best-effort Haiku network call (documents._generate_summary). Stub it
    # to the graceful "no summary" path (returns None) so the suite never touches
    # the network. Summary *content* is model-dependent and intentionally not
    # characterized; the shape (a "summary" key, None when unavailable) is.
    monkeypatch.setattr(documents, "_generate_summary", lambda text: None)


def test_save_upload_redirects_into_tmp_path(docs_dir):
    # Proves the monkeypatch redirection works: a document written through the
    # public API materializes under the tmp_path directory, not the real dir.
    result = save_upload("matt", "a.md", b"# Doc One\nbody")
    doc_id = result["document_id"]
    assert (docs_dir / "matt" / f"{doc_id}.txt").exists()
    assert (docs_dir / "matt" / f"{doc_id}-a.md").exists()
    # Returned metadata shape + values (verbatim current behavior).
    assert sorted(result.keys()) == ["char_count", "document_id", "summary", "title"]
    assert result["title"] == "Doc One"
    assert result["char_count"] == len("# Doc One\nbody")
    assert result["summary"] is None  # stubbed generator -> graceful no-summary path


def test_load_document_returns_title_and_text(docs_dir):
    result = save_upload("matt", "a.md", b"# Doc One\nbody")
    loaded = load_document("matt", result["document_id"])
    assert loaded == ("Doc One", "# Doc One\nbody")


def test_load_document_bare_txt_without_original_sibling(docs_dir):
    # load_document's original_name fallback: when no "<id>-*" sibling exists,
    # the display name defaults to "<id>.txt" (which has no "# " H1 impact here
    # since the text itself carries the H1).
    user_docs_dir = docs_dir / "matt"
    user_docs_dir.mkdir(parents=True, exist_ok=True)
    doc_id = "bare-doc-id"
    (user_docs_dir / f"{doc_id}.txt").write_text("# Bare\nz")
    assert load_document("matt", doc_id) == ("Bare", "# Bare\nz")


def test_l3_load_missing_document_returns_none(docs_dir):
    # L3: missing document -> returns None (not an exception).
    assert load_document("matt", "does-not-exist") is None


def test_l2_list_empty_existing_directory(docs_dir):
    # L2: directory exists but contains no *.txt -> [].
    (docs_dir / "matt").mkdir(parents=True, exist_ok=True)
    assert asyncio.run(list_documents("matt")) == []


def test_l2_list_missing_directory_returns_empty(docs_dir):
    # L2: directory does not exist at all -> [] (early return).
    assert not docs_dir.exists()
    assert asyncio.run(list_documents("matt")) == []


def test_l1_list_shape_and_deterministic_order(docs_dir):
    # L1: list_documents sorts by uploaded_at DESC. Order IS deterministic
    # here because save_upload writes each original file with a distinct mtime
    # (later save -> newer mtime -> appears first).
    r1 = save_upload("matt", "first.md", b"# First\nx")
    r2 = save_upload("matt", "second.md", b"# Second\ny")

    docs = asyncio.run(list_documents("matt"))
    assert len(docs) == 2

    # Exact shape of each entry (keys) — verbatim current behavior.
    for entry in docs:
        assert sorted(entry.keys()) == [
            "char_count",
            "document_id",
            "summary",
            "title",
            "uploaded_at",
        ]

    # Deterministic order: most-recently-saved first.
    assert [d["title"] for d in docs] == ["Second", "First"]
    assert docs[0]["document_id"] == r2["document_id"]
    assert docs[1]["document_id"] == r1["document_id"]
    assert docs[0]["char_count"] == len("# Second\ny")
    assert docs[1]["char_count"] == len("# First\nx")


def test_documents_are_user_scoped(docs_dir):
    a = save_upload("matt", "a.md", b"# A\nbody")
    b = save_upload("sarah", "b.md", b"# B\nbody")

    matt_ids = {d["document_id"] for d in asyncio.run(list_documents("matt"))}
    sarah_ids = {d["document_id"] for d in asyncio.run(list_documents("sarah"))}

    assert a["document_id"] in matt_ids and a["document_id"] not in sarah_ids
    assert b["document_id"] in sarah_ids and b["document_id"] not in matt_ids
    # Mirror image: cross-user load_document returns None.
    assert load_document("sarah", a["document_id"]) is None
    assert load_document("matt", a["document_id"]) is not None
    # A fresh user's picker is empty (demo docs deferred — spec §1).
    assert asyncio.run(list_documents("dev")) == []
