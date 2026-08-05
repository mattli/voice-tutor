"""Pure, Pipecat-free study-session listing helper.

Single purpose: read the append-only cost-log session ledger and surface the
completed *study* sessions so the /study/ UI can browse past recaps without
already knowing a session's UUID.

This module deliberately imports nothing from FastAPI / pipecat / bot — only
``documents`` (to resolve a document_id → title, exactly as the reference
``/api/sessions/latest`` join in app.py does). The FastAPI route in app.py is a
thin wrapper around ``list_study_sessions()``.

``SESSION_LOG_JSONL_PATH`` is a module-level constant read at CALL time (not bound
into a local at import time) so a test can ``monkeypatch.setattr`` it to a
per-test tmp_path ledger — mirroring documents.DOCUMENTS_DIR / grounding.WIKI_DIR.
"""

import json
from pathlib import Path

import documents

SESSION_LOG_JSONL_PATH = (
    Path.home() / "second-brain" / "products" / "voice-tutor" / "validation" / "session-log.jsonl"
)

# Per-user transcript root, mirroring session_state.TRANSCRIPTS_DIR. Defined
# LOCALLY (not imported) so this module's import closure stays stdlib+documents,
# and read at CALL time so a test can monkeypatch it.
TRANSCRIPTS_DIR = Path.home() / ".voice-tutor" / "transcripts"


def list_study_sessions(user_id: str) -> list[dict]:
    """Return completed study sessions for ``user_id`` only, newest first.

    ``user_id`` is REQUIRED — this is a scoped listing, not a global one. Rows
    whose ``user_id`` does not match the argument are excluded entirely
    (structural filter), so one user can never see another user's session
    history. There is no unscoped overload.

    Each row is a mapping with exactly:
      - ``session_id``
      - ``document_title`` (resolved via ``documents.load_document(user_id, document_id)``;
        ``None`` if the document no longer resolves)
      - ``session_start`` (raw ISO string from the ledger, unmodified)
      - ``session_duration_sec``
      - ``cost_total_usd``

    A row qualifies iff ``kind == "session"``, ``mode == "study"``, its
    ``user_id`` matches the argument, and it carries a non-null
    ``document_id``. Open-chat / doc-less / non-session (e.g. artifact) /
    other-user rows are excluded. Malformed / non-JSON lines are skipped,
    never fatal. An empty or absent ledger yields an empty list.
    """
    # Read the path from the module namespace at call time so monkeypatch works.
    path = SESSION_LOG_JSONL_PATH
    if not path.exists():
        return []

    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            try:
                entry = json.loads(line)
            except Exception:
                # Malformed / non-JSON line — skip, never fatal.
                continue
            if not isinstance(entry, dict):
                continue
            # Filter BEFORE extracting output fields so mis-tagged rows lacking
            # session_start/duration/cost never raise.
            if entry.get("kind") != "session":
                continue
            if entry.get("mode") != "study":
                continue
            if entry.get("user_id") != user_id:
                continue
            doc_id = entry.get("document_id")
            if doc_id is None:
                continue
            loaded = documents.load_document(user_id, doc_id)
            rows.append(
                {
                    "session_id": entry.get("session_id"),
                    "document_title": loaded[0] if loaded else None,
                    "session_start": entry.get("session_start"),
                    "session_duration_sec": entry.get("session_duration_sec"),
                    "cost_total_usd": entry.get("cost_total_usd"),
                }
            )

    # Newest first by session_start (ISO-8601 lexical) descending. Sort is stable,
    # so equal-session_start ties keep their relative order and never raise.
    rows.sort(key=lambda r: r.get("session_start") or "", reverse=True)
    return rows


MATT_ONLY_USER = "matt"


def session_belongs_to(user_id: str, session_id: str) -> bool:
    """True iff ``session_id`` is ``user_id``'s session.

    TRANSCRIPT FIRST, ledger second. The transcript is written at the very START
    of teardown and lives in the user's OWN namespace
    (``TRANSCRIPTS_DIR/<user_id>/<session_id>.json``), so its presence there is
    itself the ownership fact — and the path is built from the AUTHENTICATED
    ``user_id``, so it can only ever prove membership of that user's directory.

    The ledger row, by contrast, is written at the very END of teardown, after
    the coverage judge, the summary, and the analysis. Keying ownership on it
    made this check — and therefore the whole `/telemetry` composite the ended
    view polls — return "not found" for the entire duration of teardown. The
    frontend polls for 60s and then reports "Recap didn't generate", so a slow
    teardown presented as a BROKEN RECAP even though every artifact had landed.
    Observed 2026-08-04: with the coverage judge in the chain the ledger row
    arrived at ~66s, so all 30 polls 404'd while the recap had been on disk
    since 5s.

    The ledger scan is kept as the fallback for sessions whose transcript is
    absent (never written because the user never spoke, or since archived), so
    this only ever ADDS true results — it cannot make a previously-authorized
    session unauthorized.

    Both ids are collapsed to a single path component before the probe, so a
    crafted ``session_id`` cannot escape the user's directory and answer this
    question about someone else's file.
    """
    safe_user = Path(str(user_id)).name
    safe_session = Path(str(session_id)).name
    if safe_user and safe_session:
        if (TRANSCRIPTS_DIR / safe_user / f"{safe_session}.json").exists():
            return True

    path = SESSION_LOG_JSONL_PATH
    if not path.exists():
        return False
    with path.open() as f:
        for line in f:
            try:
                e = json.loads(line)
            except Exception:
                continue
            if isinstance(e, dict) and e.get("kind") == "session" and e.get("session_id") == session_id:
                return e.get("user_id") == user_id
    return False


def can_view_machine_artifacts(user_id: str) -> bool:
    """Prompt + analysis + global cost-log are about the MACHINE, not the tester's
    learning. Only Matt may view them. (The prompt embeds the private claim map,
    which reading would spoil the steering the validation gate measures.)"""
    return user_id == MATT_ONLY_USER


def redact_telemetry_for_user(telemetry: dict, user_id: str) -> dict:
    """Strip Matt-only fields from the telemetry composite for non-Matt users so
    the single endpoint can't leak the analysis/prompt through a side door. The
    tester keeps their own learning artifacts (recap, cost, memory_append)."""
    if can_view_machine_artifacts(user_id):
        return telemetry
    redacted = dict(telemetry)
    redacted["analysis"] = None
    redacted["has_prompt"] = False
    return redacted
