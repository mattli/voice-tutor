"""Pure, Pipecat-free helper for the session-analysis output filename.

Kept in its own tiny module (like ``usage_ledger.py``) so the filename scheme
can be unit-tested without importing ``bot.py``'s Pipecat/anthropic/ML stack.
``bot.py`` re-imports ``session_analysis_filename`` and uses it as the sole
source of the on-disk name.
"""

from datetime import datetime


def session_analysis_filename(session_start: datetime, session_id: str | None) -> str:
    """Return the session-analysis markdown filename (no directory).

    Date-first so the ``session-analyses/`` folder sorts chronologically; the
    8-char shortid preserves the join to ``session-log.jsonl`` rows and
    ``~/.voice-tutor/artifacts/<full-uuid>.md``.

    ``session_start`` is the session's actual start time, not the write time.
    When a UUID ``session_id`` is present the name is
    ``session-analysis-<YYYY-MM-DD-HHMMSS>-<first 8 of session_id>.md``. Without
    one (the legacy non-study path, where the id is itself the start timestamp)
    it degrades to ``session-analysis-<YYYY-MM-DD-HHMMSS>.md`` — matching the
    pre-UUID date+timestamp files, which carry no shortid.
    """
    ts = session_start.strftime("%Y-%m-%d-%H%M%S")
    if session_id:
        return f"session-analysis-{ts}-{session_id[:8]}.md"
    return f"session-analysis-{ts}.md"
