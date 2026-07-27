"""Pure, Pipecat-free helper: the newest prior study session's recap for a
document, parsed into a compact shape for the session-opening prompt.

Module-level path constants are read at CALL time (not bound at import) so tests
can monkeypatch them to per-test tmp paths — mirroring documents.DOCUMENTS_DIR /
sessions.COST_LOG_JSONL_PATH.
"""

import json
from pathlib import Path

COST_LOG_JSONL_PATH = (
    Path.home() / "second-brain" / "products" / "voice-tutor" / "validation" / "cost-log.jsonl"
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
