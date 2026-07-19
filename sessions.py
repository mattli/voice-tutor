"""Pure, Pipecat-free study-session listing helper.

Single purpose: read the append-only cost-log session ledger and surface the
completed *study* sessions so the /study/ UI can browse past recaps without
already knowing a session's UUID.

This module deliberately imports nothing from FastAPI / pipecat / bot — only
``documents`` (to resolve a document_id → title, exactly as the reference
``/api/sessions/latest`` join in app.py does). The FastAPI route in app.py is a
thin wrapper around ``list_study_sessions()``.

``COST_LOG_JSONL_PATH`` is a module-level constant read at CALL time (not bound
into a local at import time) so a test can ``monkeypatch.setattr`` it to a
per-test tmp_path ledger — mirroring documents.DOCUMENTS_DIR / grounding.WIKI_DIR.
"""

import json
from pathlib import Path

import documents

COST_LOG_JSONL_PATH = (
    Path.home() / "second-brain" / "products" / "voice-tutor" / "validation" / "cost-log.jsonl"
)


def list_study_sessions() -> list[dict]:
    """Return completed study sessions, newest first.

    Each row is a mapping with exactly:
      - ``session_id``
      - ``document_title`` (resolved via ``documents.load_document(document_id)``;
        ``None`` if the document no longer resolves)
      - ``session_start`` (raw ISO string from the ledger, unmodified)
      - ``session_duration_sec``
      - ``cost_total_usd``

    A row qualifies iff ``kind == "session"``, ``mode == "study"``, and it carries
    a non-null ``document_id``. Open-chat / doc-less / non-session (e.g. artifact)
    rows are excluded. Malformed / non-JSON lines are skipped, never fatal. An
    empty or absent ledger yields an empty list.
    """
    # Read the path from the module namespace at call time so monkeypatch works.
    path = COST_LOG_JSONL_PATH
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
            doc_id = entry.get("document_id")
            if doc_id is None:
                continue
            loaded = documents.load_document(doc_id)
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
