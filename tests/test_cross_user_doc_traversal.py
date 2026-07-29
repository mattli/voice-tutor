"""Regression tests for the cross-user document read via a crafted ``doc_id``.

A logged-in user could read ANOTHER user's document text (and claim sidecar) by
supplying a ``document_id`` shaped like ``../<other_user>/<their_doc_uuid>`` that
string-joins its way out of the caller's ``documents/<user_id>/`` directory. The
fix sanitizes ``doc_id`` to a single path component (``Path(doc_id).name``) at the
shared helper boundaries so every read/write path inherits containment:

  * documents._load_from_dir       (reached from documents.load_document — bot.py,
    app.py, sessions.py all call load_document)
  * claims._claims_path            (read/write of the .claims.json sidecar)
  * claims._resolve_doc_namespace  (user-vs-_shared placement decision; an
    unsanitized id here lets a brand-new attacker's crafted id resolve to the
    SHARED namespace and poison its sidecar — see the _shared-write test below)

These tests pin the containment PROPERTY directly at each boundary: a crafted id
must resolve inside the CALLER's namespace, never another user's dir. The demo-docs
suite deliberately did not pin this (the guard lived upstream at the route), so this
is a genuine coverage gap being closed. See
products/voice-tutor/planning/2026-07-28-cross-user-doc-read-gap.md.

IMPORTANT test precondition: ``..`` traversal via pathlib + os.stat only resolves
when the caller's OWN directory already exists (an intermediate component must
exist to traverse ``..`` out of it). In production the attacker is a provisioned
user whose ``documents/<attacker>/`` dir exists, so every read-leak test here first
materializes the attacker's dir — otherwise the traversal silently ENOENTs and the
test would pass against vulnerable code for the wrong reason.

Hermetic: uses the shared ``docs_dir`` / ``claims_docs_dir`` fixtures (conftest.py),
which redirect the module-level DOCUMENTS_DIR of documents.py and claims.py to the
SAME per-test tmp_path/documents and prove the real ~/.voice-tutor dir is untouched.
No network (the doc-summary Haiku call is stubbed; claims extraction is monkeypatched).
"""

import asyncio

import pytest

import app
import claims
import documents
from claims import Claim

VICTIM = "sarah"
ATTACKER = "matt"


@pytest.fixture(autouse=True)
def _stub_summary(monkeypatch):
    """No network from save_upload's best-effort summary call."""
    monkeypatch.setattr(documents, "_generate_summary", lambda text: None)


def _seed_txt(root, user_id, doc_id, text):
    """Write ``documents/<user_id>/<doc_id>.txt`` directly and return its path."""
    d = root / user_id
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{doc_id}.txt"
    p.write_text(text)
    return p


def _ensure_user_dir(root, user_id):
    """Materialize ``documents/<user_id>/`` so ``..`` traversal out of it resolves
    (the realistic precondition: the attacker is a provisioned user)."""
    (root / user_id).mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# documents.py — load_document / _load_from_dir containment
# --------------------------------------------------------------------------- #


def test_load_document_crafted_id_cannot_read_other_users_doc(docs_dir):
    victim_id = "0000victim-uuid"
    secret = "# Victim Secret\nsarah's private notes"
    _seed_txt(docs_dir, VICTIM, victim_id, secret)
    _ensure_user_dir(docs_dir, ATTACKER)  # attacker exists -> traversal resolves

    crafted = f"../{VICTIM}/{victim_id}"
    loaded = documents.load_document(ATTACKER, crafted)

    assert loaded is None, f"crafted id leaked victim doc: {loaded!r}"


def test_load_document_crafted_id_absolute_path(docs_dir):
    # An absolute-path shaped id must also not escape the caller's namespace
    # (a `dir / "/abs/path"` join otherwise discards the base dir entirely).
    victim_id = "abs-victim-uuid"
    victim_path = _seed_txt(docs_dir, VICTIM, victim_id, "secret")
    _ensure_user_dir(docs_dir, ATTACKER)
    abs_like = str(victim_path.with_suffix(""))  # /.../documents/sarah/<uuid>
    assert documents.load_document(ATTACKER, abs_like) is None


def test_load_from_dir_crafted_id_stays_in_dir(docs_dir):
    # The helper boundary itself: _load_from_dir(matt_dir, "../sarah/<id>") must
    # not read out of matt_dir.
    victim_id = "helper-victim-uuid"
    _seed_txt(docs_dir, VICTIM, victim_id, "secret text")
    matt_dir = docs_dir / ATTACKER
    matt_dir.mkdir(parents=True, exist_ok=True)
    assert documents._load_from_dir(matt_dir, f"../{VICTIM}/{victim_id}") is None


def test_load_document_own_doc_still_resolves(docs_dir):
    # The sanitize must not break the legitimate case: a plain uuid still loads.
    res = documents.save_upload(ATTACKER, "a.md", b"# Mine\nbody")
    loaded = documents.load_document(ATTACKER, res["document_id"])
    assert loaded == ("Mine", "# Mine\nbody")


# --------------------------------------------------------------------------- #
# claims.py — path / namespace / sidecar containment
# --------------------------------------------------------------------------- #


def test_claims_path_crafted_id_stays_in_caller_namespace(claims_docs_dir):
    p = claims._claims_path(ATTACKER, f"../{VICTIM}/xyz")
    # Resolved path must live under documents/matt, never documents/sarah.
    assert p.resolve().parent == (claims_docs_dir / ATTACKER).resolve()
    assert VICTIM not in p.resolve().parts


def test_write_claims_crafted_id_writes_into_caller_namespace(claims_docs_dir):
    path = claims.write_claims(
        ATTACKER, f"../{VICTIM}/pwn", [Claim(id="c1", claim="x", anchor="x")]
    )
    assert path.resolve().parent == (claims_docs_dir / ATTACKER).resolve()
    # Nothing was written into the victim's dir.
    assert not (claims_docs_dir / VICTIM).exists()


def test_load_fresh_claims_crafted_id_cannot_read_other_users_sidecar(claims_docs_dir):
    victim_id = "fresh-victim-uuid"
    victim_text = "victim document text"
    _seed_txt(claims_docs_dir, VICTIM, victim_id, victim_text)
    # Victim has a FRESH sidecar for their own doc.
    claims.write_claims(
        VICTIM,
        victim_id,
        [Claim(id="c1", claim="victim claim", anchor="victim")],
        source_hash=claims._hash_source(victim_text),
    )
    _ensure_user_dir(claims_docs_dir, ATTACKER)  # attacker exists -> traversal resolves

    # Attacker tries to read it via a crafted id, passing the victim's text so the
    # freshness hash WOULD match if the read escaped.
    got = claims.load_fresh_claims(ATTACKER, f"../{VICTIM}/{victim_id}", victim_text)
    assert got is None, f"crafted id leaked victim claims: {got!r}"


def test_resolve_doc_namespace_crafted_id_cannot_alias_shared(claims_docs_dir):
    # A brand-new attacker (no own dir yet) crafts an id that, unsanitized, makes
    # _resolve_doc_namespace return "_shared" (user_txt ENOENTs; the crafted
    # shared_txt = _shared/../<x>/<y>.txt traverses to an existing file). That
    # would route generate_claims' WRITE into the shared namespace — sidecar
    # poisoning. Sanitized, the id collapses to its final component and resolves
    # to the caller's own namespace.
    victim_id = "shared-poison-uuid"
    _seed_txt(claims_docs_dir, VICTIM, victim_id, "victim body")
    (claims_docs_dir / claims.SHARED_USER_ID).mkdir(parents=True, exist_ok=True)
    # NB: attacker (matt) has NO dir here — the scenario that reaches the _shared branch.
    ns = claims._resolve_doc_namespace(ATTACKER, f"../{VICTIM}/{victim_id}")
    assert ns == ATTACKER, f"crafted id resolved to foreign namespace: {ns!r}"


def test_generate_claims_crafted_id_writes_into_caller_namespace(claims_docs_dir, monkeypatch):
    victim_id = "gen-victim-uuid"
    victim_text = "victim doc text"
    _seed_txt(claims_docs_dir, VICTIM, victim_id, victim_text)
    victim_sidecar = claims.write_claims(
        VICTIM,
        victim_id,
        [Claim(id="c1", claim="victim claim", anchor="victim")],
        source_hash=claims._hash_source(victim_text),
    )
    victim_before = victim_sidecar.read_bytes()
    _ensure_user_dir(claims_docs_dir, ATTACKER)

    # No live LLM: extraction is monkeypatched to a fixed attacker claim set.
    monkeypatch.setattr(
        claims, "extract_claims", lambda text: [Claim(id="c1", claim="attacker", anchor="a")]
    )
    out = claims.generate_claims(ATTACKER, f"../{VICTIM}/{victim_id}", "attacker text")
    assert [c.claim for c in out] == ["attacker"]

    # Victim's sidecar is byte-for-byte untouched (not read-as-cache, not overwritten).
    assert victim_sidecar.read_bytes() == victim_before
    # The attacker's write landed in the attacker's own namespace.
    attacker_sidecars = list((claims_docs_dir / ATTACKER).glob("*.claims.json"))
    assert attacker_sidecars, "attacker's sidecar was not written into their namespace"


# --------------------------------------------------------------------------- #
# app.upload_document — the rejection-reason attach must not fail a good upload
# --------------------------------------------------------------------------- #


class _FakeUpload:
    def __init__(self, filename, raw):
        self.filename = filename
        self._raw = raw

    async def read(self):
        return self._raw


def test_upload_survives_load_document_hiccup_during_annotation(
    docs_dir, claims_docs_dir, monkeypatch
):
    # save_upload succeeds and the file lands; then the display-only rejection-
    # reason annotation re-reads the doc. If that re-read raises (disk hiccup),
    # the successful upload must NOT be turned into an error response.
    from fastapi import BackgroundTasks

    calls = {"n": 0}

    def flaky_load(user_id, doc_id):
        # The only SYNCHRONOUS load_document in the route is the annotation re-read
        # (save_upload doesn't call it; the warm task runs later, in the background).
        calls["n"] += 1
        raise OSError("simulated disk read hiccup")

    monkeypatch.setattr(documents, "load_document", flaky_load)

    bg = BackgroundTasks()
    result = asyncio.run(
        app.upload_document(_FakeUpload("a.md", b"# Doc\nbody"), bg, ATTACKER)
    )

    assert calls["n"] >= 1, "annotation path did not attempt the re-read"
    assert result.get("document_id"), "upload result missing document_id"
    assert "claim_extraction_rejected" not in result
    # The file really did land despite the annotation hiccup.
    doc_id = result["document_id"]
    assert (docs_dir / ATTACKER / f"{doc_id}.txt").exists()
