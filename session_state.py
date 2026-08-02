"""Pure, Pipecat-free session-state helpers extracted verbatim from bot.py.

These functions and the Path constants they depend on were relocated here with
zero logic changes so they can be characterized in isolation and imported
without pulling in the STT/TTS/Pipecat/LLM stack. bot.py re-imports every name
defined here.
"""

import json
from datetime import datetime
from pathlib import Path

VOICE_TUTOR_DIR = Path.home() / ".voice-tutor"
TRANSCRIPTS_DIR = VOICE_TUTOR_DIR / "transcripts"
PROFILES_DIR = VOICE_TUTOR_DIR / "profiles"
MEMORY_DIR = VOICE_TUTOR_DIR / "memory"


def profile_path(user_id: str) -> Path:
    return PROFILES_DIR / f"{Path(user_id).name}.md"


def memory_path(user_id: str) -> Path:
    return MEMORY_DIR / f"{Path(user_id).name}.md"


def load_profile(user_id: str) -> str:
    p = profile_path(user_id)
    return p.read_text() if p.exists() else ""


def _format_memory_date(iso_ts: str) -> str:
    dt = datetime.fromisoformat(iso_ts)
    hour12 = dt.hour % 12 or 12
    ampm = "AM" if dt.hour < 12 else "PM"
    return f"{dt.strftime('%B')} {dt.day}, {dt.year} — {hour12}:{dt.minute:02d} {ampm}"


def append_to_memory(user_id: str, transcript: dict, summary_text: str):
    p = memory_path(user_id)
    header = f"## {_format_memory_date(transcript['session_start'])}\n"
    entry = header + summary_text.strip() + "\n\n"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            "# Memory — what we've discussed\n\n"
            "One section per session, append-only. Summaries are lifted from "
            "the `.summary.md` sidecar written alongside each transcript.\n\n"
        )
    with p.open("a") as f:
        f.write(entry)


def load_memory(user_id: str) -> str:
    p = memory_path(user_id)
    return p.read_text() if p.exists() else ""


def _format_session_time(iso_ts: str) -> str:
    dt = datetime.fromisoformat(iso_ts)
    # %-d / %-I strip leading zeros on macOS/Linux; avoid cross-platform flags.
    month_day = dt.strftime("%B ") + str(dt.day)
    hour12 = dt.hour % 12 or 12
    ampm = "am" if dt.hour < 12 else "pm"
    return f"{month_day}, {dt.year} at {hour12}:{dt.minute:02d}{ampm}"


def _format_full_transcript_block(transcript: dict, header_suffix: str = "") -> str:
    header = f"## Session from {_format_session_time(transcript['session_start'])}{header_suffix}\n"
    lines = []
    for turn in transcript["turns"]:
        role = "You" if turn["role"] == "assistant" else "Matt"
        lines.append(f"  {role}: {turn['content']}")
    return header + "\n".join(lines)


def load_most_recent_transcript_block(user_id: str) -> str | None:
    """Return the most recent full-transcript block, or None if no transcripts exist.

    Older sessions are no longer loaded here — they accumulate in memory.md instead.
    """
    user_dir = TRANSCRIPTS_DIR / Path(user_id).name
    if not user_dir.exists():
        return None
    files = sorted(
        (f for f in user_dir.glob("*.json") if not f.name.endswith(".usage.json")),
        reverse=True,
    )
    if not files:
        return None
    transcript = json.loads(files[0].read_text())
    return _format_full_transcript_block(transcript, header_suffix=" (most recent)")


def count_user_turns(turns: list[dict]) -> int:
    """Number of USER utterances in a transcript's turn list.

    Counts only turns with role ``'user'`` — the tutor's hidden kickoff and its
    spoken replies (role ``'assistant'``) are excluded. This is deliberately NOT
    a total-turn count.
    """
    return sum(1 for t in turns if t.get("role") == "user")


def has_min_user_turns(turns: list[dict], minimum: int) -> bool:
    """Whether the USER spoke at least ``minimum`` times this session.

    The telemetry gate: a session's summary + analysis are generated iff this is
    True. Keys on USER utterances only (see ``count_user_turns``) — a session
    where only the tutor's opener fired and the user never spoke returns False
    and is skipped. Deliberately NOT a total-turn threshold, so nobody mistakes
    it for one: the assistant kickoff must never make an otherwise-silent session
    look answered.
    """
    return count_user_turns(turns) >= minimum


def session_expects_summary(transcript_path: Path, min_user_turns: int) -> bool:
    """Whether the session at ``transcript_path`` WILL produce a summary/analysis.

    Applies the SAME user-turn gate (``has_min_user_turns``) that bot.py uses at
    write time, so the /telemetry poll's done-condition can only expect what the
    writer actually produces — an expectation the writer can never satisfy would
    make the poll spin to its cap. A missing or unreadable transcript (the user
    never spoke, so no transcript was written) → False.
    """
    try:
        turns = json.loads(transcript_path.read_text()).get("turns", [])
    except (OSError, json.JSONDecodeError):
        return False
    return has_min_user_turns(turns, min_user_turns)
