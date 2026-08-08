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
import inspect

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
    # The preserved original lives in _originals/, never beside the extracted
    # text — a flat original that is itself a .txt would be scanned as its own
    # document (see documents.ORIGINALS_DIRNAME).
    assert (docs_dir / "matt" / documents.ORIGINALS_DIRNAME / f"{doc_id}-a.md").exists()
    assert not (docs_dir / "matt" / f"{doc_id}-a.md").exists()
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


# --------------------------------------------------------------------------- #
# Shared-namespace (documents/_shared/) resolution fallback.
#
# Docs seeded under documents/_shared/ are offered to EVERY user and loadable by
# every user, WITHOUT weakening per-user isolation: user namespace resolves
# first, then _shared/ on a miss; a colliding user doc shadows the shared one.
# save_upload NEVER writes to _shared/. These extend the mirror-image pattern
# above and use the same conftest-monkeypatched ``docs_dir`` fixture.
# --------------------------------------------------------------------------- #


def _seed_doc(dir_path, doc_id, text, original_name=None):
    """Write a document (its .txt, and an original sibling) directly into a
    namespace directory — bypassing save_upload, so we can seed the _shared/ dir
    (which save_upload deliberately never writes to)."""
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / f"{doc_id}.txt").write_text(text)
    if original_name is not None:
        (dir_path / f"{doc_id}-{original_name}").write_text(text)


def test_shared_doc_visible_to_every_user_own_uploads_stay_private(docs_dir):
    # A doc in documents/_shared/ appears in the picker for matt, sarah, and a
    # third user; each user's own upload stays invisible to the others.
    _seed_doc(docs_dir / "_shared", "shared-1", "# Shared\nbody", "shared.md")
    a = save_upload("matt", "a.md", b"# Matt A\nbody")
    b = save_upload("sarah", "b.md", b"# Sarah B\nbody")

    for user in ("matt", "sarah", "carol"):
        ids = {d["document_id"] for d in asyncio.run(list_documents(user))}
        assert "shared-1" in ids, f"shared doc missing for {user}"

    matt_ids = {d["document_id"] for d in asyncio.run(list_documents("matt"))}
    sarah_ids = {d["document_id"] for d in asyncio.run(list_documents("sarah"))}
    carol_ids = {d["document_id"] for d in asyncio.run(list_documents("carol"))}
    # Own uploads stay private (mirror-image isolation preserved under the union).
    assert a["document_id"] in matt_ids and a["document_id"] not in sarah_ids
    assert b["document_id"] in sarah_ids and b["document_id"] not in matt_ids
    # A third user with no uploads sees ONLY the shared doc.
    assert carol_ids == {"shared-1"}


def test_shared_doc_absent_shared_dir_list_unchanged(docs_dir):
    # With no _shared/ dir seeded, list behavior is unchanged: fresh user -> [];
    # a user with only their own docs sees exactly those.
    assert not (docs_dir / "_shared").exists()
    assert asyncio.run(list_documents("dev")) == []
    a = save_upload("matt", "a.md", b"# A\nbody")
    ids = {d["document_id"] for d in asyncio.run(list_documents("matt"))}
    assert ids == {a["document_id"]}


def test_list_dedupes_shadow_collision_prefers_user_doc(docs_dir):
    # A user doc and a shared doc sharing a doc_id: the id appears EXACTLY ONCE
    # in the user's picker, as the USER's version (title/metadata), matching load.
    _seed_doc(docs_dir / "_shared", "D", "# Shared Title\nshared body", "shared.md")
    _seed_doc(docs_dir / "matt", "D", "# Matt Title\nmatt body", "matt.md")

    docs = asyncio.run(list_documents("matt"))
    entries = [d for d in docs if d["document_id"] == "D"]
    assert len(entries) == 1
    assert entries[0]["title"] == "Matt Title"  # user's version, not shared


def test_load_document_shared_resolves_for_two_users(docs_dir):
    # A doc only in _shared/ loads successfully for two different users.
    _seed_doc(docs_dir / "_shared", "shared-1", "# Shared\nbody", "shared.md")
    assert load_document("matt", "shared-1") == ("Shared", "# Shared\nbody")
    assert load_document("sarah", "shared-1") == ("Shared", "# Shared\nbody")


def test_load_document_other_users_doc_still_none(docs_dir):
    # A doc that belongs to another user and is NOT shared -> None (isolation).
    s = save_upload("sarah", "s.md", b"# Sarah\nbody")
    assert load_document("matt", s["document_id"]) is None


def test_load_document_shadowing_user_shadows_shared(docs_dir):
    # Deterministic shadowing: matt has his own 'D'; sarah does not.
    _seed_doc(docs_dir / "_shared", "D", "# Shared\nshared body", "shared.md")
    _seed_doc(docs_dir / "matt", "D", "# Matt\nmatt body", "matt.md")
    assert load_document("matt", "D") == ("Matt", "# Matt\nmatt body")
    assert load_document("sarah", "D") == ("Shared", "# Shared\nshared body")


def test_save_upload_never_writes_to_shared(docs_dir):
    # Pre-seed _shared/ with a sentinel; snapshot its bytes. Uploads for two
    # users land under their OWN dirs and leave _shared/ byte-for-byte unchanged.
    import hashlib

    shared = docs_dir / "_shared"
    _seed_doc(shared, "sentinel", "# Sentinel\nx", "sentinel.md")

    def snap(root):
        out = {}
        for p in sorted(root.rglob("*")):
            if p.is_file():
                out[p.relative_to(root).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
        return out

    before = snap(shared)

    m = save_upload("matt", "m.md", b"# M\nbody")
    s = save_upload("sarah", "s.md", b"# S\nbody")

    assert (docs_dir / "matt" / f"{m['document_id']}.txt").exists()
    assert (docs_dir / "sarah" / f"{s['document_id']}.txt").exists()
    assert snap(shared) == before, "save_upload mutated the _shared/ namespace"

    # Aliasing hole closed upstream: no request can drive save_upload with
    # '_shared' as the acting user, because sanitize_user_id rejects it.
    import identity
    assert identity.sanitize_user_id("_shared") is None


def test_document_helper_signatures_unchanged(docs_dir):
    # No signature churn: parameter names/order pinned, and list_documents is a
    # coroutine (async-ness preserved). The shared namespace is an internal
    # fallback only.
    assert list(inspect.signature(list_documents).parameters) == ["user_id"]
    assert list(inspect.signature(load_document).parameters) == ["user_id", "doc_id"]
    assert list(inspect.signature(save_upload).parameters) == ["user_id", "filename", "raw"]
    assert asyncio.iscoroutinefunction(list_documents)
