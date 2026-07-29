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
from fastapi.params import Depends as params_Depends

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


# =========================================================================== #
# SPRINT 2 — idempotency THROUGH the guard; prepare-path / _shared integrity.
#
# These tests exercise the REAL generate_claims / load_fresh_claims source_hash
# cache guard (UNPATCHED) and spy ONLY the deepest live LLM step,
# claims.extract_claims — so there is never a live API call, yet the single
# freshness decision inside the shared cache guard is what governs whether an
# extraction happens. Where an assertion needs a deterministic count, the ONE
# scheduled warm BackgroundTask is captured and driven to completion exactly
# once and the _claims_warming in-flight set is confirmed reset, so the observed
# extract-count is never an artifact of in-flight de-dup.
# =========================================================================== #


def _force_doc_id(monkeypatch, doc_id: str):
    """Pin documents.save_upload's minted uuid so an upload lands at ``doc_id``.

    save_upload derives the document_id via ``uuid.uuid4()`` (module-level
    ``uuid`` in documents.py). Forcing it lets a test pre-seed a sidecar keyed by
    the doc's source_hash BEFORE upload, so idempotency can be proven through the
    real end-to-end upload-completion entry point (not a direct _warm_claims call).
    """

    class _FixedUUID:
        def __str__(self):
            return doc_id

    monkeypatch.setattr(documents.uuid, "uuid4", lambda: _FixedUUID())


def _run_warm_tasks(bg: BackgroundTasks):
    """Execute every scheduled _warm_claims task in ``bg`` to completion, once.

    Returns the list of (user_id, doc_id) tuples that were run. Driving the real
    async task here is what turns 'a task was scheduled' into 'the warm actually
    ran through the live cache guard', so the extract-spy count is meaningful.
    """
    ran = []
    for task in bg.tasks:
        if task.func is app._warm_claims:
            asyncio.run(app._warm_claims(*task.args))
            ran.append(task.args)
    return ran


def _seed_fresh_sidecar(user_id: str, doc_id: str, text: str):
    """Write a valid sidecar stamped with ``text``'s current source_hash."""
    fresh = [claims.Claim(id="c1", claim="seeded", anchor="w", anchor_unresolved=True)]
    return claims.write_claims(
        user_id, doc_id, fresh, source_hash=claims._hash_source(text)
    )


# --------------------------------------------------------------------------- #
# c1: idempotency proven through the full UPLOAD-COMPLETION end-to-end entry.
# --------------------------------------------------------------------------- #


def test_upload_completion_fresh_sidecar_zero_extraction_end_to_end(
    docs_dir, claims_docs_dir, monkeypatch
):
    _no_summary(monkeypatch)
    doc_id = "fixed-fresh-doc"
    text = _text_of_words(300)  # under the bound
    _force_doc_id(monkeypatch, doc_id)

    # Drive the REAL upload-completion path end-to-end. save_upload writes the
    # user's <doc_id>.txt (so the doc resolves to the uploader namespace), then
    # the endpoint schedules the shared warm. The real generate_claims guard runs
    # UNPATCHED; only the deepest LLM step is spied.
    extract = MagicMock()
    with patch.object(claims, "extract_claims", extract):
        result, bg = _run_upload("d.txt", text.encode("utf-8"))
        assert result["document_id"] == doc_id

        # Pre-seed a FRESH sidecar keyed by the doc's current source_hash, in the
        # uploader's namespace, AFTER the .txt exists so it resolves user-first.
        sidecar = _seed_fresh_sidecar(USER_ID, doc_id, text)
        before = sidecar.read_bytes()
        before_mtime = sidecar.stat().st_mtime_ns

        # Capture-and-run the exactly-one scheduled warm task to completion once.
        ran = _run_warm_tasks(bg)

    assert ran == [(USER_ID, doc_id)], (
        "upload completion must schedule exactly the one shared warm task for the "
        f"uploader's namespace and doc; got {ran!r}"
    )
    # Not an in-flight de-dup artifact: the marker is reset before we assert.
    assert app._claims_warming == set()
    assert extract.called is False, (
        "a fresh sidecar (matching source_hash) must short-circuit inside the "
        "real generate_claims guard — the live LLM step must never be invoked"
    )
    # The pre-seeded sidecar is untouched, and the uploaded doc still loads.
    assert sidecar.read_bytes() == before
    assert sidecar.stat().st_mtime_ns == before_mtime
    assert documents.load_document(USER_ID, doc_id) is not None


# --------------------------------------------------------------------------- #
# c2: the trigger routes THROUGH the single source_hash freshness decision —
# fresh -> zero extract, stale/absent -> exactly one.
# --------------------------------------------------------------------------- #


def test_upload_fresh_vs_stale_governed_by_one_freshness_decision(
    docs_dir, claims_docs_dir, monkeypatch
):
    _no_summary(monkeypatch)
    text = _text_of_words(250)

    # Case (a): FRESH sidecar for the current source_hash -> zero extraction.
    _force_doc_id(monkeypatch, "doc-fresh")
    extract_a = MagicMock()
    with patch.object(claims, "extract_claims", extract_a):
        result_a, bg_a = _run_upload("a.txt", text.encode("utf-8"))
        _seed_fresh_sidecar(USER_ID, result_a["document_id"], text)
        ran_a = _run_warm_tasks(bg_a)
    assert ran_a == [(USER_ID, "doc-fresh")]
    assert app._claims_warming == set()
    assert extract_a.call_count == 0, "fresh sidecar must yield zero extract calls"

    # Case (b): STALE sidecar (hash of DIFFERENT text) -> exactly one extraction.
    _force_doc_id(monkeypatch, "doc-stale")
    extract_b = MagicMock(return_value=[])
    with patch.object(claims, "extract_claims", extract_b):
        result_b, bg_b = _run_upload("b.txt", text.encode("utf-8"))
        # Seed a sidecar stamped with a NON-matching source_hash → stale.
        _seed_fresh_sidecar(USER_ID, result_b["document_id"], _text_of_words(999))
        ran_b = _run_warm_tasks(bg_b)
    assert ran_b == [(USER_ID, "doc-stale")]
    assert app._claims_warming == set()
    assert extract_b.call_count == 1, "a stale sidecar must trigger exactly one extract"


# --------------------------------------------------------------------------- #
# c3: an unwarmed doc (no/stale sidecar) driven through upload completion runs
# live extraction exactly once, for the uploader's user_id + doc_id.
# --------------------------------------------------------------------------- #


def test_upload_completion_unwarmed_extracts_once_correct_namespace(
    docs_dir, claims_docs_dir, monkeypatch
):
    _no_summary(monkeypatch)
    doc_id = "unwarmed-doc"
    text = _text_of_words(400)
    _force_doc_id(monkeypatch, doc_id)

    extract = MagicMock(return_value=[])  # no sidecar exists → must be reached once
    with patch.object(claims, "extract_claims", extract):
        result, bg = _run_upload("d.txt", text.encode("utf-8"))
        assert result["document_id"] == doc_id
        # The scheduled task carries the correct user_id and doc_id.
        assert _scheduled_warm_calls(bg) == [(USER_ID, doc_id)]
        ran = _run_warm_tasks(bg)

    assert ran == [(USER_ID, doc_id)]
    assert app._claims_warming == set()
    assert extract.call_count == 1, "an unwarmed doc must trigger exactly one extract"
    # Extraction ran on the uploaded document's own text.
    assert extract.call_args.args[0] == text


# --------------------------------------------------------------------------- #
# c4: the picker-click prepare path stays intact as the retry/fallback warm path
# and goes through the SAME shared cache guard.
# --------------------------------------------------------------------------- #


def _place_doc(base_dir, user_id: str, doc_id: str, text: str):
    """Write a bare <doc_id>.txt under ``base_dir/user_id`` (no sidecar)."""
    d = base_dir / user_id
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{doc_id}.txt").write_text(text)


def _run_prepare(doc_id: str, user_id: str = USER_ID):
    """Call the real prepare route with a real BackgroundTasks; return (status, bg)."""
    bg = BackgroundTasks()
    status = asyncio.run(app.prepare_claims(doc_id, bg, user_id))
    return status, bg


def test_prepare_path_unwarmed_extracts_once_fresh_yields_zero(
    docs_dir, claims_docs_dir, monkeypatch
):
    _no_summary(monkeypatch)
    text = _text_of_words(200)

    # Unwarmed doc via the prepare path -> exactly one extraction.
    _place_doc(docs_dir, USER_ID, "prep-unwarmed", text)
    extract_u = MagicMock(return_value=[])
    with patch.object(claims, "extract_claims", extract_u):
        status, bg = _run_prepare("prep-unwarmed")
        assert status == {"status": "warming"}
        ran = _run_warm_tasks(bg)
    assert ran == [(USER_ID, "prep-unwarmed")]
    assert app._claims_warming == set()
    assert extract_u.call_count == 1

    # Fresh-sidecar doc via the prepare path -> zero extraction (cached short-circuit).
    _place_doc(docs_dir, USER_ID, "prep-fresh", text)
    _seed_fresh_sidecar(USER_ID, "prep-fresh", text)
    extract_f = MagicMock()
    with patch.object(claims, "extract_claims", extract_f):
        status_f, bg_f = _run_prepare("prep-fresh")
        assert status_f == {"status": "cached"}, "fresh sidecar → prepare reports cached"
        ran_f = _run_warm_tasks(bg_f)  # no warm scheduled; drives nothing
    assert ran_f == []
    assert app._claims_warming == set()
    assert extract_f.called is False


# --------------------------------------------------------------------------- #
# c5: _shared prepare-path warming intact; upload-path namespace isolation.
# --------------------------------------------------------------------------- #


def test_prepare_warms_unwarmed_shared_doc_in_shared_namespace(
    docs_dir, claims_docs_dir, monkeypatch
):
    _no_summary(monkeypatch)
    text = _text_of_words(180)
    # Place an unwarmed doc in the _shared namespace only (no per-user copy).
    _place_doc(docs_dir, documents.SHARED_USER_ID, "shared-doc", text)

    extract = MagicMock(return_value=[])
    with patch.object(claims, "extract_claims", extract):
        status, bg = _run_prepare("shared-doc", USER_ID)
        assert status == {"status": "warming"}
        ran = _run_warm_tasks(bg)
    assert ran == [(USER_ID, "shared-doc")]
    assert app._claims_warming == set()
    assert extract.call_count == 1, "unwarmed _shared doc must warm via prepare path"
    # The sidecar was written into the _shared namespace (resolved there), not
    # the per-user one — the single shared sidecar serves every user.
    assert (docs_dir / documents.SHARED_USER_ID / "shared-doc.claims.json").exists()
    assert not (docs_dir / USER_ID / "shared-doc.claims.json").exists()


def test_upload_isolation_shared_sidecar_untouched_on_collision(
    docs_dir, claims_docs_dir, monkeypatch
):
    # Isolation control: a pre-existing _shared sidecar at the SAME doc_id is
    # neither read nor written nor extracted; the uploaded doc warms strictly in
    # the uploader's namespace (save_upload writes into user_dir(user_id), and
    # generate_claims resolves user-first). The collision is forced explicitly.
    _no_summary(monkeypatch)
    doc_id = "collision-doc"
    user_text = _text_of_words(220)
    shared_text = _text_of_words(500)  # different content → different source_hash

    # Seed a _shared doc + a FRESH _shared sidecar at the colliding doc_id.
    _place_doc(docs_dir, documents.SHARED_USER_ID, doc_id, shared_text)
    shared_sidecar = _seed_fresh_sidecar(documents.SHARED_USER_ID, doc_id, shared_text)
    shared_before = shared_sidecar.read_bytes()
    shared_before_mtime = shared_sidecar.stat().st_mtime_ns

    _force_doc_id(monkeypatch, doc_id)
    extract = MagicMock(return_value=[])
    with patch.object(claims, "extract_claims", extract):
        result, bg = _run_upload("mine.txt", user_text.encode("utf-8"))
        assert result["document_id"] == doc_id
        ran = _run_warm_tasks(bg)

    # The executed warm targets the UPLOADER's user_id (never _shared).
    assert ran == [(USER_ID, doc_id)]
    assert app._claims_warming == set()
    # The uploaded per-user doc shadows the shared one, so extraction ran on the
    # UPLOADER's text and wrote the per-user sidecar.
    assert extract.call_count == 1
    assert extract.call_args.args[0] == user_text
    assert (docs_dir / USER_ID / f"{doc_id}.claims.json").exists()
    # The _shared sidecar is byte-for-byte and mtime untouched — never read as the
    # cache for this warm, never rewritten, never the extraction target.
    assert shared_sidecar.read_bytes() == shared_before
    assert shared_sidecar.stat().st_mtime_ns == shared_before_mtime


# --------------------------------------------------------------------------- #
# c6: no route/helper signature changes; required-user_id discipline intact.
# --------------------------------------------------------------------------- #


def _param_spec(func):
    """(ordered param names, {name: has_default}) for ``func``."""
    params = inspect.signature(func).parameters
    names = list(params)
    has_default = {n: p.default is not inspect.Parameter.empty for n, p in params.items()}
    return names, has_default


def _user_id_default(func):
    """The default OBJECT for ``func``'s user_id parameter (or Parameter.empty)."""
    return inspect.signature(func).parameters["user_id"].default


def test_signatures_and_required_user_id_discipline_unchanged():
    # Sprint-start literals: exact ordered param names + required/optional status
    # per symbol. For the two ROUTE functions, user_id is injected via FastAPI's
    # ``Depends(require_user)`` — a required dependency whose "default" is the
    # Depends marker (that IS the required-user_id discipline for routes, and is
    # unchanged this sprint). For every pure helper, user_id is non-defaulted.
    expected = {
        app.upload_document: (
            ["file", "background_tasks", "user_id"],
            {"file": False, "background_tasks": False, "user_id": True},
        ),
        app.prepare_claims: (
            ["doc_id", "background_tasks", "user_id"],
            {"doc_id": False, "background_tasks": False, "user_id": True},
        ),
        app._warm_claims: (["user_id", "doc_id"], {"user_id": False, "doc_id": False}),
        app._schedule_warm: (
            ["background_tasks", "user_id", "doc_id"],
            {"background_tasks": False, "user_id": False, "doc_id": False},
        ),
        claims.generate_claims: (
            ["user_id", "doc_id", "document_text"],
            {"user_id": False, "doc_id": False, "document_text": False},
        ),
        claims.load_fresh_claims: (
            ["user_id", "doc_id", "document_text"],
            {"user_id": False, "doc_id": False, "document_text": False},
        ),
        claims.extract_claims: (["document_text"], {"document_text": False}),
        claims._resolve_doc_namespace: (
            ["user_id", "doc_id"],
            {"user_id": False, "doc_id": False},
        ),
        documents.save_upload: (
            ["user_id", "filename", "raw"],
            {"user_id": False, "filename": False, "raw": False},
        ),
        documents.load_document: (["user_id", "doc_id"], {"user_id": False, "doc_id": False}),
    }
    # The route functions where user_id is a required FastAPI dependency.
    route_funcs = {app.upload_document, app.prepare_claims}
    for func, (exp_names, exp_required) in expected.items():
        names, has_default = _param_spec(func)
        assert names == exp_names, f"{func.__qualname__} param order changed: {names!r}"
        for pname, should_be_defaulted in exp_required.items():
            assert has_default[pname] is should_be_defaulted, (
                f"{func.__qualname__}.{pname} required/optional status changed"
            )
        # Required-user_id discipline: routes inject via Depends(require_user);
        # helpers keep user_id a plain required (non-defaulted) parameter.
        if "user_id" in names:
            if func in route_funcs:
                dep = _user_id_default(func)
                assert isinstance(dep, params_Depends) and dep.dependency is app.require_user, (
                    f"{func.__qualname__}.user_id must stay injected via "
                    "Depends(require_user)"
                )
            else:
                assert has_default["user_id"] is False, (
                    f"{func.__qualname__}.user_id must stay required (non-defaulted)"
                )
