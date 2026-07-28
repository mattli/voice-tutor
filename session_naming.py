"""Pure, Pipecat-free helpers for session-analysis filenames — the writer's
name builder and the reader's finder, kept together so they can't drift.

Kept in its own tiny module (like ``usage_ledger.py``) so the scheme can be
unit-tested without importing ``bot.py``'s Pipecat/anthropic/ML stack. ``bot.py``
re-imports ``session_analysis_filename`` to write the file; ``app.py`` re-imports
``find_analysis_path`` to look it up by session id. Because the on-disk name is
date-first (``session-analysis-<YYYY-MM-DD-HHMMSS>-<shortid>.md``) the reader
can't reconstruct the exact name from a session id alone — it lacks the start
time — so it matches on the shortid the writer embeds. The round-trip
(build → find) is the property the tests pin.
"""

from datetime import datetime
from pathlib import Path

# Length of the session-id prefix embedded in the filename. Shared by the writer
# and the reader so a change here can't silently desync them.
SHORTID_LEN = 8


def session_analysis_filename(session_start: datetime, session_id: str | None) -> str:
    """Return the session-analysis markdown filename (no directory).

    Date-first so the ``session-analyses/`` folder sorts chronologically; the
    shortid preserves the join to ``session-log.jsonl`` rows and
    ``~/.voice-tutor/artifacts/<full-uuid>.md``.

    ``session_start`` is the session's actual start time, not the write time.
    When a UUID ``session_id`` is present the name is
    ``session-analysis-<YYYY-MM-DD-HHMMSS>-<shortid>.md``. Without one (the legacy
    non-study path, where the id is itself the start timestamp) it degrades to
    ``session-analysis-<YYYY-MM-DD-HHMMSS>.md`` — matching the pre-UUID
    date+timestamp files, which carry no shortid.
    """
    ts = session_start.strftime("%Y-%m-%d-%H%M%S")
    if session_id:
        return f"session-analysis-{ts}-{session_id[:SHORTID_LEN]}.md"
    return f"session-analysis-{ts}.md"


def find_analysis_path(directory: Path, session_id: str) -> Path | None:
    """Locate the session-analysis file for ``session_id`` under ``directory``.

    Files are named by :func:`session_analysis_filename` — date-first with a
    shortid suffix — so the exact name isn't reconstructable from the id alone.
    Match on the shortid (the first ``SHORTID_LEN`` chars of the id, embedded
    verbatim by the writer) via a glob. Returns the matching path, or ``None`` if
    none exists. On the astronomically unlikely shortid collision, returns the
    first match by sorted name (deterministic).

    The shortid must be alphanumeric (real UUID prefixes are hex); a non-alnum
    prefix — e.g. a caller passing glob metacharacters — yields ``None`` rather
    than an over-broad glob.
    """
    shortid = session_id[:SHORTID_LEN]
    if not shortid or not shortid.isalnum():
        return None
    matches = sorted(directory.glob(f"session-analysis-*-{shortid}.md"))
    return matches[0] if matches else None
