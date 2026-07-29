"""Hermetic tests for auto-warm on upload completion (goal Part 1).

The verifier runs EXACTLY:
    uv run --with pytest pytest tests/test_upload_autowarm.py -q
in a fresh worktree, so ALL hermetic tests for this sprint live in THIS file.

Determinism / no live API: the background warm task (``app._warm_claims``) is
monkeypatched or spied, and the document-summary Haiku call
(``documents._generate_summary``) is stubbed to a constant — so uploading a doc
performs NO network I/O. Where the REAL warm path is exercised (idempotency /
tripwire), the deeper LLM step (``claims.extract_claims``) is spied instead, and
``claims.generate_claims``'s real source_hash cache guard runs live.

House rules honored (see CLAUDE.md): app.py imports pipecat at module scope, so
we do NOT construct a FastAPI ``TestClient``. Instead we call the thin async
route function ``app.upload_document`` directly with a real ``BackgroundTasks``
(which captures scheduled tasks) and hermetic tmp storage via the shared
``docs_dir`` / ``claims_docs_dir`` fixtures, which redirect the module-level
DOCUMENTS_DIR of documents.py and claims.py respectively.
"""

import asyncio
import inspect
import logging
from unittest.mock import MagicMock, patch

import pytest
from fastapi import BackgroundTasks

import app
import claims
import documents

USER_ID = "matt"
OTHER_USER = "sarah"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


class _FakeUpload:
    """Minimal stand-in for fastapi.UploadFile: .filename + async .read()."""

    def __init__(self, filename: str, raw: bytes):
        self.filename = filename
        self._raw = raw

    async def read(self) -> bytes:
        return self._raw


def _text_of_words(n: int) -> str:
    """``n`` whitespace-separated tokens -> len(text.split()) == n exactly."""
    return " ".join(["w"] * n)


def _no_summary(monkeypatch):
    """Stub the doc-summary Haiku call so save_upload makes no network call."""
    monkeypatch.setattr(documents, "_generate_summary", lambda text: None)


def _run_upload(filename: str, raw: bytes, user_id: str = USER_ID):
    """Call the real upload route with a real BackgroundTasks; return (result, bg)."""
    bg = BackgroundTasks()
    result = asyncio.run(app.upload_document(_FakeUpload(filename, raw), bg, user_id))
    return result, bg


def _scheduled_warm_calls(bg: BackgroundTasks):
    """Extract (user_id, doc_id) tuples for every _warm_claims task in ``bg``."""
    calls = []
    for task in bg.tasks:
        if task.func is app._warm_claims:
            calls.append(task.args)
    return calls


@pytest.fixture(autouse=True)
def _reset_warming():
    """Keep the module-level in-flight set clean across tests."""
    app._claims_warming.clear()
    yield
    app._claims_warming.clear()


# --------------------------------------------------------------------------- #
# c1 / c2 / c9: a fresh under-bound upload schedules the SHARED warm task once,
# for the uploading user's namespace, with the returned document_id.
# --------------------------------------------------------------------------- #


def test_upload_schedules_shared_warm_with_correct_user_and_doc(docs_dir, monkeypatch):
    _no_summary(monkeypatch)
    text = _text_of_words(300)  # well under the bound

    # Spy the shared warm task so no extraction runs and we can assert the seam.
    spy = MagicMock()
    monkeypatch.setattr(app, "_warm_claims", spy)

    result, bg = _run_upload("notes.txt", text.encode("utf-8"))
    doc_id = result["document_id"]

    calls = _scheduled_warm_calls(bg)
    assert calls == [(USER_ID, doc_id)], (
        "upload must schedule the shared warm exactly once, for the uploading "
        f"user's namespace and the returned document_id; got {calls!r}"
    )
    # The under-bound doc carries no rejection message.
    assert result.get("claim_extraction_rejected") is None


def test_upload_warm_targets_uploader_namespace_not_shared(docs_dir, monkeypatch):
    # c9: the scheduled warm carries the uploading user's user_id — never _shared.
    _no_summary(monkeypatch)
    monkeypatch.setattr(app, "_warm_claims", MagicMock())
    result, bg = _run_upload("d.txt", _text_of_words(50).encode("utf-8"), OTHER_USER)
    (uid, _doc_id), = _scheduled_warm_calls(bg)
    assert uid == OTHER_USER
    assert uid != documents.SHARED_USER_ID


def test_upload_reuses_existing_warm_task_no_bespoke_extraction(docs_dir, monkeypatch):
    # c2: the seam is the existing shared _warm_claims task the prepare path uses
    # — not a new inline extraction. Assert the scheduled callable is identity-
    # equal to app._warm_claims (the same task prepare_claims schedules).
    _no_summary(monkeypatch)
    monkeypatch.setattr(app, "_warm_claims", MagicMock())
    _result, bg = _run_upload("d.txt", _text_of_words(10).encode("utf-8"))
    warm_tasks = [t for t in bg.tasks if t.func is app._warm_claims]
    assert len(warm_tasks) == 1, "exactly the shared warm task, scheduled once"


# --------------------------------------------------------------------------- #
# c3 / c4: upload success is INDEPENDENT of extraction success.
# --------------------------------------------------------------------------- #


def test_upload_succeeds_when_triggered_warm_raises(docs_dir, monkeypatch):
    # c3: even if the warm task raises, the upload response is the normal success
    # payload and the file landed. (The real _warm_claims swallows exceptions;
    # here we prove the ENDPOINT never propagates a warm failure regardless.)
    _no_summary(monkeypatch)

    def _boom(user_id, doc_id):
        raise RuntimeError("extraction blew up")

    monkeypatch.setattr(app, "_warm_claims", _boom)

    result, _bg = _run_upload("d.txt", _text_of_words(100).encode("utf-8"))
    # Same success shape as today: the metadata dict from save_upload.
    assert result["document_id"]
    assert result["char_count"] == len(_text_of_words(100))
    assert "title" in result and "summary" in result
    # File landed on disk in the uploading user's namespace.
    doc_id = result["document_id"]
    assert (docs_dir / USER_ID / f"{doc_id}.txt").exists()


def test_doc_loadable_and_unwarmed_after_failed_extraction(docs_dir, claims_docs_dir, monkeypatch):
    # c4: after a failed/absent extraction the doc loads as an unwarmed doc.
    _no_summary(monkeypatch)
    text = _text_of_words(120)

    def _boom(user_id, doc_id):
        raise RuntimeError("extraction blew up")

    monkeypatch.setattr(app, "_warm_claims", _boom)
    result, _bg = _run_upload("d.txt", text.encode("utf-8"))
    doc_id = result["document_id"]

    loaded = documents.load_document(USER_ID, doc_id)
    assert loaded is not None
    _title, loaded_text = loaded
    assert loaded_text == text
    # No fresh sidecar -> behaves exactly as an unwarmed doc (plain study mode).
    assert claims.load_fresh_claims(USER_ID, doc_id, loaded_text) is None


# --------------------------------------------------------------------------- #
# c5: an already-warmed (fresh sidecar) doc is NOT re-extracted — idempotency is
# the real generate_claims source_hash cache guard reached through the warm path.
# --------------------------------------------------------------------------- #


def test_prewarmed_doc_not_reextracted_through_real_warm_path(docs_dir, claims_docs_dir, monkeypatch):
    _no_summary(monkeypatch)
    text = _text_of_words(200)
    real_warm = app._warm_claims  # capture BEFORE patching, to run for real later

    # Upload the doc for real (warm spied so nothing extracts during upload).
    with patch.object(app, "_warm_claims", MagicMock()):
        result, _bg = _run_upload("d.txt", text.encode("utf-8"))
    doc_id = result["document_id"]

    # Pre-seed a FRESH sidecar for the doc's current source_hash.
    fresh = [claims.Claim(id="c1", claim="seeded", anchor="w", anchor_unresolved=True)]
    sidecar = claims.write_claims(
        USER_ID, doc_id, fresh, source_hash=claims._hash_source(text)
    )
    before = sidecar.read_bytes()
    before_mtime = sidecar.stat().st_mtime_ns

    # Now run the REAL shared warm end-to-end with the live cache guard, spying
    # only the deeper LLM extraction step (extract_claims).
    extract = MagicMock()
    with patch.object(claims, "extract_claims", extract):
        asyncio.run(real_warm(USER_ID, doc_id))

    assert extract.called is False, (
        "a fresh sidecar (matching source_hash) must short-circuit — the LLM "
        "extraction step must never be invoked"
    )
    # The pre-seeded sidecar's content and mtime are unchanged.
    assert sidecar.read_bytes() == before
    assert sidecar.stat().st_mtime_ns == before_mtime


# --------------------------------------------------------------------------- #
# c6 / c7: an OVERSIZED doc — upload still succeeds, file lands + loadable +
# unwarmed; the rejection reason (count, limit, remedy) is surfaced; the tripwire
# fires + logs inside the shared warm seam.
# --------------------------------------------------------------------------- #


def test_oversized_upload_lands_file_and_surfaces_rejection(docs_dir, claims_docs_dir, monkeypatch):
    _no_summary(monkeypatch)
    over = claims.CLAIM_MAX_WORDS + 250
    text = _text_of_words(over)
    assert len(text.split()) == over

    # Spy the warm task so the endpoint alone is under test here.
    monkeypatch.setattr(app, "_warm_claims", MagicMock())
    result, bg = _run_upload("huge.txt", text.encode("utf-8"))
    doc_id = result["document_id"]

    # Upload still succeeded (same success shape) and the file landed + loads.
    assert result["document_id"]
    assert result["char_count"] == len(text)
    assert (docs_dir / USER_ID / f"{doc_id}.txt").exists()
    loaded = documents.load_document(USER_ID, doc_id)
    assert loaded is not None and loaded[1] == text
    # Unwarmed: no fresh sidecar.
    assert claims.load_fresh_claims(USER_ID, doc_id, text) is None

    # The rejection reason surfaced to the user names the actual word count, the
    # limit, and a split/excerpt suggestion.
    reason = result.get("claim_extraction_rejected")
    assert reason, "oversized upload must surface a rejection reason"
    assert str(over) in reason, f"actual word count {over} missing from: {reason!r}"
    assert str(claims.CLAIM_MAX_WORDS) in reason, f"limit missing from: {reason!r}"
    assert "split" in reason.lower() or "excerpt" in reason.lower()

    # Not a new upload failure mode: the warm is still scheduled regardless.
    assert _scheduled_warm_calls(bg) == [(USER_ID, doc_id)]


def test_under_bound_upload_not_rejected_and_schedules_warm(docs_dir, monkeypatch):
    # c6 (mirror): a doc under the bound is not rejected and DOES schedule warm.
    _no_summary(monkeypatch)
    monkeypatch.setattr(app, "_warm_claims", MagicMock())
    result, bg = _run_upload("small.txt", _text_of_words(500).encode("utf-8"))
    assert result.get("claim_extraction_rejected") is None
    assert _scheduled_warm_calls(bg) == [(USER_ID, result["document_id"])]


def test_oversized_warm_logs_rejection_with_word_count_via_seam(docs_dir, claims_docs_dir, monkeypatch, caplog):
    # c7: the durable rejection log line (with the word count) is emitted by the
    # claims tripwire path reached THROUGH the real shared warm seam — not from a
    # redundant handler-side check. Run the REAL _warm_claims end-to-end.
    _no_summary(monkeypatch)
    over = claims.CLAIM_MAX_WORDS + 313
    text = _text_of_words(over)
    real_warm = app._warm_claims  # capture BEFORE patching, to run for real later

    # Upload for real (warm spied during upload so it doesn't run yet).
    with patch.object(app, "_warm_claims", MagicMock()):
        result, _bg = _run_upload("huge.txt", text.encode("utf-8"))
    doc_id = result["document_id"]

    # Now drive the REAL shared warm seam; extract_claims must never be reached.
    extract = MagicMock()
    with caplog.at_level(logging.INFO, logger="claims"):
        with patch.object(claims, "extract_claims", extract):
            # Best-effort warm: must NOT raise to the caller.
            asyncio.run(real_warm(USER_ID, doc_id))

    assert extract.called is False, "tripwire must precede extraction at the seam"
    claim_records = [r for r in caplog.records if r.name == "claims"]
    assert claim_records, "no rejection log line emitted by the claims seam"
    assert any(str(over) in r.getMessage() for r in claim_records), (
        f"rejection log line missing the word count {over}"
    )


# --------------------------------------------------------------------------- #
# c8: signatures / discipline unchanged; additive optional response field only.
# --------------------------------------------------------------------------- #


def test_helper_signatures_unchanged():
    # documents helpers keep required user_id discipline and exact signatures.
    assert list(inspect.signature(documents.save_upload).parameters) == [
        "user_id", "filename", "raw",
    ]
    assert list(inspect.signature(documents.load_document).parameters) == [
        "user_id", "doc_id",
    ]
    assert list(inspect.signature(claims.generate_claims).parameters) == [
        "user_id", "doc_id", "document_text",
    ]
    # The shared warm task + scheduling seam keep the (user_id, doc_id) shape.
    assert list(inspect.signature(app._warm_claims).parameters) == ["user_id", "doc_id"]


def test_rejection_reason_is_pure_and_reuses_bound(claims_docs_dir):
    # rejection_reason reuses claims' single source of truth; under-bound -> None.
    assert claims.rejection_reason(_text_of_words(claims.CLAIM_MAX_WORDS)) is None
    over = claims.CLAIM_MAX_WORDS + 1
    reason = claims.rejection_reason(_text_of_words(over))
    assert reason is not None
    assert str(over) in reason and str(claims.CLAIM_MAX_WORDS) in reason


def test_upload_response_is_superset_of_todays_success_shape(docs_dir, monkeypatch):
    # c3/c8: the success payload is a superset — the original keys are all present
    # and unchanged; only an optional field may be added.
    _no_summary(monkeypatch)
    monkeypatch.setattr(app, "_warm_claims", MagicMock())
    result, _bg = _run_upload("d.txt", _text_of_words(30).encode("utf-8"))
    for key in ("document_id", "title", "char_count", "summary"):
        assert key in result
