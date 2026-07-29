"""Regression tests for the cross-user WRITE (and persisted-read) via a crafted
``session_id``.

Sibling of the doc_id read-gap fix. The WebRTC offer body carries a client-
controlled ``session_id`` (``bot.py`` ``session_id_override = body.get("session_id")``)
which is stamped into ``study_meta["session_id"]`` (bot.py, the ``session_id_override
or document_id`` construction) and from there becomes the filename stem of every
per-session artifact the bot writes:

    TRANSCRIPTS_DIR/<user_id>/<session_id>.prompt.txt   (system prompt)
    TRANSCRIPTS_DIR/<user_id>/<session_id>.json         (transcript)
    TRANSCRIPTS_DIR/<user_id>/<session_id>.summary.md
    TRANSCRIPTS_DIR/<user_id>/<session_id>.usage.json
    ARTIFACTS_DIR/<user_id>/<session_id>.md             (recap artifact)

A crafted ``session_id`` like ``../<victim>/x`` string-joins OUT of the caller's
own directory, so an attacker (using their OWN valid ``document_id`` to enter study
mode) can overwrite another user's saved system-prompt / transcript / recap files —
a cross-user WRITE / integrity break. This is pre-existing and distinct from the
doc_id READ gap; the ``app.py`` read routes already guard the session id
(``safe_id = Path(session_id).name``), but ``bot.py``'s writer does not.

The same value, once persisted into ``session-log.jsonl``, is read back by
``study_history.previous_session_recap`` which builds
``ARTIFACTS_DIR/<user_id>/<best_sid>.md`` from the stored ``session_id`` — so a
crafted stored id is also a cross-user READ of another user's recap artifact.

Fix: a single shared, Pipecat-free helper ``session_naming.safe_session_id`` (mirrors
the doc_id ``Path(...).name`` guard), applied at the ONE ``study_meta["session_id"]``
construction in bot.py (so every writer inherits it) and at the persisted-log read in
study_history.py (a separate trust boundary — rows written before the fix).

Hermetic: ``session_naming`` is pure; ``study_history`` uses the ``study_history_tmp``
fixture (conftest.py) which redirects its ledger + ARTIFACTS_DIR to per-test tmp and
proves the real dirs are untouched. ``bot.py``'s ``bot()`` coroutine constructs the
full Pipecat/ML pipeline and is deliberately NOT unit-tested (see CLAUDE.md "test via
pure helpers, not TestClient"); the write-side property is pinned at the pure helper
that bot.py's single construction point uses.
"""

import json

import pytest

import session_naming
import study_history

VICTIM = "sarah"
ATTACKER = "matt"


# --------------------------------------------------------------------------- #
# session_naming.safe_session_id — the shared choke point (pure)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "crafted",
    [
        f"../{VICTIM}/pwned",
        f"../../{VICTIM}/pwned",
        "a/../b",
        "foo/bar",
    ],
)
def test_safe_session_id_collapses_to_single_component(crafted):
    safe = session_naming.safe_session_id(crafted)
    assert "/" not in safe
    assert VICTIM not in safe or safe == VICTIM  # never a directory segment
    # It is exactly the final path component.
    from pathlib import Path

    assert safe == Path(crafted).name


def test_safe_session_id_absolute_path_stays_a_bare_name():
    assert "/" not in session_naming.safe_session_id("/etc/passwd")


def test_safe_session_id_noop_on_legit_ids():
    # A real UUID and a legacy timestamp stem must pass through unchanged.
    for legit in ("2f6a1c4e-9b0d-4a11-8c2e-1234567890ab", "2026-07-29-140355"):
        assert session_naming.safe_session_id(legit) == legit


def test_safe_session_id_none_passes_through():
    # None is the explicit "no id" signal (callers on the write/read paths never
    # pass it); it is returned unchanged, not turned into a placeholder.
    assert session_naming.safe_session_id(None) is None


@pytest.mark.parametrize("degenerate", ["", ".", "..", "foo/..", "a/."])
def test_safe_session_id_degenerate_returns_safe_bare_component(degenerate):
    # These sanitize (via Path(...).name) to "", ".", or ".." — each of which, as a
    # BARE path component, escapes or aliases the directory. The helper must instead
    # return a safe, non-empty single component so it is safe even when joined
    # WITHOUT a suffix (the nit: suffix-appending was a convention, not a guarantee).
    from pathlib import Path

    out = session_naming.safe_session_id(degenerate)
    assert out not in ("", ".", ".."), out
    assert "/" not in out
    # A bare join stays strictly inside the directory (no escape/alias).
    base = Path("/var/tmp/vt-userdir")
    assert (base / out).resolve().parent == base.resolve()


# --------------------------------------------------------------------------- #
# Write-side property: the transcript/artifact stem stays in the caller's dir.
# Mirrors bot.py's construction (study_meta["session_id"]) + join sites.
# --------------------------------------------------------------------------- #


def test_transcript_write_stem_stays_in_caller_dir(tmp_path):
    transcripts = tmp_path / "transcripts"
    attacker_dir = transcripts / ATTACKER
    attacker_dir.mkdir(parents=True, exist_ok=True)

    crafted = f"../{VICTIM}/pwned"
    stem = session_naming.safe_session_id(crafted)  # what bot.py stamps into study_meta
    for suffix in (".prompt.txt", ".json", ".summary.md", ".usage.json"):
        dest = attacker_dir / f"{stem}{suffix}"
        assert dest.resolve().parent == attacker_dir.resolve(), suffix
        assert VICTIM not in dest.resolve().parts


# --------------------------------------------------------------------------- #
# Persisted-read: a crafted STORED session_id must not read a cross-user recap.
# --------------------------------------------------------------------------- #


def _write_session_row(ledger, *, user_id, document_id, session_id, start):
    row = {
        "kind": "session",
        "mode": "study",
        "user_id": user_id,
        "document_id": document_id,
        "session_id": session_id,
        "session_start": start,
    }
    with ledger.open("a") as f:
        f.write(json.dumps(row) + "\n")


def test_previous_session_recap_crafted_stored_session_id_cannot_read_cross_user(
    study_history_tmp,
):
    ledger, artifacts = study_history_tmp
    doc_id = "shared-doc"

    # Victim owns a real recap artifact.
    (artifacts / VICTIM).mkdir(parents=True, exist_ok=True)
    (artifacts / VICTIM / "secret.md").write_text(
        "## What we covered\n- victim's private recap\n"
    )
    # Attacker's own artifacts dir exists (so the crafted `..` can traverse).
    (artifacts / ATTACKER).mkdir(parents=True, exist_ok=True)

    # Attacker's session-log row stores a crafted session_id pointing at the victim.
    crafted = f"../{VICTIM}/secret"
    _write_session_row(
        ledger,
        user_id=ATTACKER,
        document_id=doc_id,
        session_id=crafted,
        start="2026-07-29T10:00:00",
    )

    recap = study_history.previous_session_recap(
        ATTACKER, doc_id, exclude_session_id="current-session"
    )
    assert recap is None, f"crafted stored session_id leaked victim recap: {recap!r}"


def test_previous_session_recap_legit_session_still_resolves(study_history_tmp):
    # Guard: the sanitize must not break the normal same-user recap read.
    ledger, artifacts = study_history_tmp
    doc_id = "doc-1"
    sid = "2f6a1c4e-9b0d-4a11-8c2e-1234567890ab"
    (artifacts / ATTACKER).mkdir(parents=True, exist_ok=True)
    (artifacts / ATTACKER / f"{sid}.md").write_text(
        "## What we covered\n- vectors\n## Open threads\n- eigenvalues\n"
    )
    _write_session_row(
        ledger, user_id=ATTACKER, document_id=doc_id, session_id=sid, start="2026-07-29T09:00:00"
    )
    recap = study_history.previous_session_recap(ATTACKER, doc_id, exclude_session_id="other")
    assert recap == {"covered": ["vectors"], "open_threads": ["eigenvalues"]}
