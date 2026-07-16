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
PROFILE_PATH = VOICE_TUTOR_DIR / "profile.md"
MEMORY_PATH = VOICE_TUTOR_DIR / "memory.md"


def load_profile() -> str:
    if PROFILE_PATH.exists():
        return PROFILE_PATH.read_text()
    return ""


def _format_memory_date(iso_ts: str) -> str:
    dt = datetime.fromisoformat(iso_ts)
    hour12 = dt.hour % 12 or 12
    ampm = "AM" if dt.hour < 12 else "PM"
    return f"{dt.strftime('%B')} {dt.day}, {dt.year} — {hour12}:{dt.minute:02d} {ampm}"


def append_to_memory(transcript: dict, summary_text: str):
    header = f"## {_format_memory_date(transcript['session_start'])}\n"
    entry = header + summary_text.strip() + "\n\n"
    if not MEMORY_PATH.exists():
        MEMORY_PATH.write_text(
            "# Memory — what we've discussed\n\n"
            "One section per session, append-only. Summaries are lifted from "
            "the `.summary.md` sidecar written alongside each transcript.\n\n"
        )
    with MEMORY_PATH.open("a") as f:
        f.write(entry)


def load_memory() -> str:
    if not MEMORY_PATH.exists():
        return ""
    return MEMORY_PATH.read_text()


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


def load_most_recent_transcript_block() -> str | None:
    """Return the most recent full-transcript block, or None if no transcripts exist.

    Older sessions are no longer loaded here — they accumulate in memory.md instead.
    """
    if not TRANSCRIPTS_DIR.exists():
        return None
    files = sorted(
        (f for f in TRANSCRIPTS_DIR.glob("*.json") if not f.name.endswith(".usage.json")),
        reverse=True,
    )
    if not files:
        return None
    transcript = json.loads(files[0].read_text())
    return _format_full_transcript_block(transcript, header_suffix=" (most recent)")
