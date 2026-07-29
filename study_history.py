"""Pure, Pipecat-free helper: the newest prior study session's recap for a
document, parsed into a compact shape for the session-opening prompt.

Module-level path constants are read at CALL time (not bound at import) so tests
can monkeypatch them to per-test tmp paths — mirroring documents.DOCUMENTS_DIR /
sessions.SESSION_LOG_JSONL_PATH.
"""

import json
from pathlib import Path

from session_naming import safe_session_id

SESSION_LOG_JSONL_PATH = (
    Path.home() / "second-brain" / "products" / "voice-tutor" / "validation" / "session-log.jsonl"
)
ARTIFACTS_DIR = Path.home() / ".voice-tutor" / "artifacts"

_FALLBACK_MAX_CHARS = 1000


def _section_bullets(text: str, header: str) -> list[str]:
    """Bullet lines under a `## <header>` section, up to the next `## ` header."""
    lines = text.splitlines()
    out: list[str] = []
    in_section = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## "):
            in_section = stripped[3:].strip().lower() == header.lower()
            continue
        if in_section and (stripped.startswith("- ") or stripped.startswith("* ")):
            out.append(stripped[2:].strip())
    return out


def parse_recap_sections(text: str) -> dict:
    """Parsed shape {"covered", "open_threads"} if a non-empty 'What we covered'
    section is found; else the fallback shape {"fallback_text": text[:1000]}."""
    covered = _section_bullets(text, "What we covered")
    if not covered:
        return {"fallback_text": text[:_FALLBACK_MAX_CHARS]}
    open_threads = _section_bullets(text, "Open threads")
    return {"covered": covered, "open_threads": open_threads}


def previous_session_recap(user_id, document_id, exclude_session_id):
    """The newest prior study session's parsed recap for ``document_id``, or None.

    Scoped to ``user_id``: ledger rows belonging to another user (or missing
    user_id) are excluded, and the recap artifact is resolved under the
    user's own subdirectory. This is the isolation boundary that prevents one
    user's "where you left off" recap from leaking into another user's
    session opener.

    Newest-only, no walk-back: if the single newest qualifying session has no
    recap artifact, return None rather than an older session's recap. A stale
    "last time we covered X" is worse than no recap.
    """
    path = SESSION_LOG_JSONL_PATH
    if not path.exists():
        return None

    best_start = None
    best_sid = None
    with path.open() as f:
        for line in f:
            try:
                entry = json.loads(line)
            except Exception:
                continue
            if not isinstance(entry, dict):
                continue
            if entry.get("kind") != "session" or entry.get("mode") != "study":
                continue
            if entry.get("user_id") != user_id:
                continue
            if entry.get("document_id") != document_id:
                continue
            sid = entry.get("session_id")
            if sid is None or sid == exclude_session_id:
                continue
            start = entry.get("session_start") or ""
            if best_start is None or start > best_start:
                best_start, best_sid = start, sid

    if best_sid is None:
        return None
    # best_sid comes from a persisted session-log row whose session_id was
    # client-controlled when written. Sanitize to a single path component before
    # building the artifact path so a crafted stored id (e.g. from a row written
    # before the bot-side guard existed) can't read another user's recap artifact.
    artifact = ARTIFACTS_DIR / Path(user_id).name / f"{safe_session_id(best_sid)}.md"
    if not artifact.exists():
        return None  # newest-only: do not walk back to an older recap
    try:
        text = artifact.read_text()
    except Exception:
        return None  # unreadable artifact (permission, encoding, delete race): degrade
    if not text.strip():
        return None  # empty/whitespace recap is not a recap
    return parse_recap_sections(text)
