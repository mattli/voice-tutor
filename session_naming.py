"""Pure, Pipecat-free helpers for session-analysis filenames — the writer's
name builder and the reader's finder, kept together so they can't drift.

Kept in its own tiny module (like ``usage_ledger.py``) so the scheme can be
unit-tested without importing ``bot.py``'s Pipecat/anthropic/ML stack. ``bot.py``
re-imports ``session_analysis_filename`` to write the file; ``app.py`` re-imports
``find_analysis_path`` to look it up by session id. Because the on-disk name is
date-first (``session-analysis-<YYYY-MM-DD-HHMMSS>-<shortid>.md``) the reader
can't reconstruct the exact name from a session id alone — it lacks the start
time — so it matches on the shortid the writer embeds. Analyses live under a
per-user ``<user_id>/`` subdirectory (``bot.py`` writes there, ``app.py`` passes
``user_id`` to the finder) so the round-trip (build → find within
``directory/<user_id>/``) is the property the tests pin.
"""

import uuid
from datetime import datetime
from pathlib import Path

# Length of the session-id prefix embedded in the filename. Shared by the writer
# and the reader so a change here can't silently desync them.
SHORTID_LEN = 8

# Sanitized results that are UNSAFE as a bare path component: joined onto a
# directory they select the directory itself ("") , the directory (".") , or its
# PARENT ("..") — i.e. they still escape/alias rather than naming a file. When
# ``Path(id).name`` lands on one of these the helper substitutes a placeholder.
_UNSAFE_COMPONENTS = frozenset({"", ".", ".."})


def safe_session_id(session_id):
    """Collapse a ``session_id`` to a single, safe path component so it can't traverse.

    The shared containment choke point for the session-id half of every on-disk
    path, mirroring the doc_id ``Path(...).name`` guard. The ``session_id`` is
    client-controllable (it arrives on the WebRTC offer body as ``session_id`` and
    is stamped into ``study_meta["session_id"]``), and it becomes the filename stem
    of the per-session transcript / prompt / usage / recap artifacts the bot writes
    under ``<user_id>/``. Without this collapse, a crafted id like
    ``../<other_user>/x`` string-joins OUT of the caller's own directory and writes
    into (or, via a persisted row, reads from) another user's namespace.

    Applied at the SINGLE point the client value enters ``study_meta`` (so every
    writer inherits it) and at the persisted-log read in ``study_history`` (a
    separate trust boundary — rows written before this guard existed). A legitimate
    id (a UUID, or a ``%Y-%m-%d-%H%M%S`` timestamp stem) contains no separators, so
    this is a no-op for it.

    Guaranteed safe even when the result is joined onto a directory WITHOUT a
    suffix: if ``Path(id).name`` collapses to a directory selector (``""`` from an
    empty id or ``"."``, or ``".."`` from ``"foo/.."``), a fresh filename-safe
    placeholder id (a hex UUID) is returned instead of the dangerous residue — so
    the function's safety is a property of the function, not a convention every
    call site must uphold by appending ``.txt``/``.md``/etc. ``None`` (the explicit
    "no id" signal) is passed through unchanged; callers that reach the write/read
    paths never pass ``None``.
    """
    if session_id is None:
        return None
    name = Path(str(session_id)).name
    if name in _UNSAFE_COMPONENTS:
        return uuid.uuid4().hex  # unusable id -> safe, unique, alphanumeric stem
    return name


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


def find_analysis_path(directory: Path, user_id: str, session_id: str) -> Path | None:
    """Locate the session-analysis file for ``session_id`` under ``directory/<user_id>/``.

    Files are named by :func:`session_analysis_filename` — date-first with a
    shortid suffix — so the exact name isn't reconstructable from the id alone.
    Match on the shortid (the first ``SHORTID_LEN`` chars of the id, embedded
    verbatim by the writer) via a glob scoped to the user's subdirectory (so one
    user's session id can never resolve into another user's analyses). Returns
    the matching path, or ``None`` if none exists. On the astronomically
    unlikely shortid collision, returns the first match by sorted name
    (deterministic).

    The shortid must be alphanumeric (real UUID prefixes are hex); a non-alnum
    prefix — e.g. a caller passing glob metacharacters — yields ``None`` rather
    than an over-broad glob. ``user_id`` is sanitized via ``Path(user_id).name``
    so it can't escape ``directory`` via path traversal.
    """
    shortid = session_id[:SHORTID_LEN]
    if not shortid or not shortid.isalnum():
        return None
    user_dir = directory / Path(user_id).name
    matches = sorted(user_dir.glob(f"session-analysis-*-{shortid}.md"))
    return matches[0] if matches else None
