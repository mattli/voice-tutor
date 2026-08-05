import asyncio
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import anthropic

import claims
import coverage_store
import documents
import identity
import study_history
import wiki
from usage_ledger import UsageLedger
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    InputAudioRawFrame,
    LLMRunFrame,
    MetricsFrame,
    TTSAudioRawFrame,
)
from pipecat.metrics.metrics import LLMUsageMetricsData, TTSUsageMetricsData
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.cartesia.tts import CartesiaTTSService, GenerationConfig
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

# Pricing constants were relocated verbatim into the pure, Pipecat-free
# cost_audit module (so the cost-log auditor can recompute costs without
# importing bot). bot.py re-imports every name here; values are unchanged and
# the logger's math below is untouched.
# Prices last verified 2026-04-15 against official pricing pages and
# cross-checked with the 2026-04-14 session's provider dashboards.
# Sources: claude.com/pricing, deepgram.com/pricing, cartesia.ai/pricing.
# Haiku 4.5 powers the post-session summary + analysis calls. Cartesia bills 1
# credit per character submitted to the TTS WebSocket; $5 / 100_000 credits on
# Pro plan = $0.00005 per character. Character count comes from pipecat's
# TTSUsageMetricsData (exact len(text) sent to Cartesia), not an estimate.
from cost_audit import (
    PRICE_ANTHROPIC_CACHE_READ_PER_MTOK,
    PRICE_ANTHROPIC_CACHE_WRITE_PER_MTOK,
    PRICE_ANTHROPIC_HAIKU_INPUT_PER_MTOK,
    PRICE_ANTHROPIC_HAIKU_OUTPUT_PER_MTOK,
    PRICE_ANTHROPIC_INPUT_PER_MTOK,
    PRICE_ANTHROPIC_OUTPUT_PER_MTOK,
    PRICE_CARTESIA_PER_CHAR,
    PRICE_DEEPGRAM_NOVA3_PER_MIN,
)

# NOTE: session-log.jsonl starts from the first session after this refactor
# (2026-04-15+). Sessions logged before this (e.g. 2026-04-14) only exist as
# rows in cost-log.md — there's no raw usage data to backfill for them.

# Diagnostic: when VOICE_TUTOR_USAGE_TRACE is truthy, UsageAccumulator logs one
# stderr line per on_push_frame observation of a usage-bearing frame (MetricsFrame /
# InputAudioRawFrame / TTSAudioRawFrame). Off by default (one os.getenv + one `if`
# on the hot path). Used to confirm the per-hop multi-count: each unique frame.id
# should appear once per processor hop the frame traverses. See CLAUDE.md
# "Pipecat observers fire per processor hop".
USAGE_TRACE = os.getenv("VOICE_TUTOR_USAGE_TRACE", "").strip().lower() not in ("", "0", "false", "no")

# Per-hop dedup for usage accounting. ON by default (the fix): UsageLedger counts
# each frame's usage once, not once per processor hop it traverses. To restore the
# legacy multi-count for a fast, no-rebuild revert, set VOICE_TUTOR_USAGE_DEDUP to
# any of these (case-insensitive): 0, false, no, off, disable, disabled. ANY OTHER
# value — including empty/unset — leaves dedup ON. See usage_ledger.py and CLAUDE.md
# "Pipecat observers fire per processor hop". Note the unset default differs from
# USAGE_TRACE: dedup defaults ON, the trace defaults OFF.
_DEDUP_DISABLE_VALUES = ("0", "false", "no", "off", "disable", "disabled")
USAGE_DEDUP = os.getenv("VOICE_TUTOR_USAGE_DEDUP", "").strip().lower() not in _DEDUP_DISABLE_VALUES

# Session-aware opening. ON by default: the study tutor opens with a plan
# (orient + propose on a first session; recap + offer the choice on a return).
# To revert to the legacy blank-slate greeting with NO rebuild, set
# VOICE_TUTOR_SESSION_OPENING to any of these (case-insensitive): 0, false, no,
# off, disable, disabled. ANY OTHER value — including empty/unset — leaves it ON.
_SESSION_OPENING_DISABLE_VALUES = ("0", "false", "no", "off", "disable", "disabled")
SESSION_OPENING = (
    os.getenv("VOICE_TUTOR_SESSION_OPENING", "").strip().lower()
    not in _SESSION_OPENING_DISABLE_VALUES
)

# Post-session coverage judging. ON by default: at teardown, a study session's
# transcript is judged against the document's claim map and a per-session
# coverage sidecar is written (see coverage_store). It runs OFF the event loop
# and AFTER the conversation has ended, so it can never touch the voice path.
# To disable with NO rebuild, set VOICE_TUTOR_COVERAGE_JUDGE to any of these
# (case-insensitive): 0, false, no, off, disable, disabled. ANY OTHER value —
# including empty/unset — leaves it ON.
_COVERAGE_JUDGE_DISABLE_VALUES = ("0", "false", "no", "off", "disable", "disabled")
COVERAGE_JUDGE = (
    os.getenv("VOICE_TUTOR_COVERAGE_JUDGE", "").strip().lower()
    not in _COVERAGE_JUDGE_DISABLE_VALUES
)


class UsageAccumulator(BaseObserver):
    def __init__(self):
        super().__init__()
        # All usage counting + the cost summary live in the pure, Pipecat-free
        # UsageLedger. This observer is a thin adapter: it extracts (frame.id,
        # plain values) off pipecat frames and delegates. The ledger dedups by
        # frame.id so per-hop multi-counting can't inflate usage (see its module
        # docstring). Instantiated once per session, so the ledger's seen-id set
        # is per-session by construction.
        self.ledger = UsageLedger(dedup=USAGE_DEDUP)
        self.tool_calls: list[dict] = []
        # Set by the tool handler immediately after it runs; the next
        # TTSAudioRawFrame closes the measurement.
        self._pending_tool: dict | None = None

    def mark_tool_call(self, page: str):
        entry = {"page": page, "start_monotonic": time.monotonic(), "timestamp": datetime.now().isoformat()}
        self._pending_tool = entry

    def _trace(self, frame, data: "FramePushed", kind: str, counted: bool, extra: str = ""):
        # One line per observation. `counted` reflects whether this observation
        # actually incremented a counter (False once dedup lands and the frame.id
        # was already seen). frame.id is the pipecat-unique per-instance id — the
        # dedup key. Tally observations-per-id from these lines to see the hop
        # multiple (pre-fix: N>1 all counted; post-fix: N observed, 1 counted).
        src = type(data.source).__name__ if data.source is not None else "None"
        dst = type(data.destination).__name__ if data.destination is not None else "None"
        print(
            f"[usage-trace] kind={kind} fid={frame.id} name={frame.name} "
            f"src={src} dst={dst} dir={getattr(data.direction, 'name', data.direction)} "
            f"counted={counted}{(' ' + extra) if extra else ''}",
            file=sys.stderr, flush=True,
        )

    async def on_push_frame(self, data: FramePushed):
        frame = data.frame
        # TTSAudioRawFrame check must precede InputAudioRawFrame because
        # both inherit from AudioRawFrame — TTS is output, STT is input.
        if isinstance(frame, TTSAudioRawFrame):
            counted = self.ledger.should_count(frame.id)
            if USAGE_TRACE:
                self._trace(frame, data, "tts_audio", counted, extra=f"bytes={len(frame.audio)}")
            if counted:
                denom = frame.sample_rate * max(frame.num_channels, 1) * 2
                if denom:
                    self.ledger.add_tts_audio(len(frame.audio) / denom)
            # Tool-latency measurement is independent of usage dedup — it fires on
            # the first post-tool audio observation and self-guards via _pending_tool.
            if self._pending_tool is not None:
                latency = time.monotonic() - self._pending_tool["start_monotonic"]
                self.tool_calls.append({
                    "page": self._pending_tool["page"],
                    "timestamp": self._pending_tool["timestamp"],
                    "latency_to_first_audio_sec": round(latency, 3),
                })
                print(
                    f"[wiki-tool] {self._pending_tool['timestamp']} "
                    f"page={self._pending_tool['page']} "
                    f"latency_to_first_audio={latency:.2f}s",
                    file=sys.stderr, flush=True,
                )
                self._pending_tool = None
            return
        if isinstance(frame, InputAudioRawFrame):
            counted = self.ledger.should_count(frame.id)
            if USAGE_TRACE:
                self._trace(frame, data, "stt_audio", counted, extra=f"bytes={len(frame.audio)}")
            if counted:
                denom = frame.sample_rate * max(frame.num_channels, 1) * 2
                if denom:
                    self.ledger.add_stt_audio(len(frame.audio) / denom)
            return
        if isinstance(frame, MetricsFrame):
            # One MetricsFrame can carry BOTH LLM and TTS usage, so the dedup
            # decision is taken once per frame and applies to all its metrics.
            counted = self.ledger.should_count(frame.id)
            if USAGE_TRACE:
                mtypes = []
                for m in frame.data:
                    if isinstance(m, LLMUsageMetricsData):
                        mtypes.append("LLM")
                    elif isinstance(m, TTSUsageMetricsData):
                        mtypes.append(f"TTS({m.value})")
                self._trace(frame, data, "metrics", counted, extra=f"mtypes={','.join(mtypes) or 'none'}")
            if counted:
                for m in frame.data:
                    if isinstance(m, LLMUsageMetricsData):
                        u = m.value
                        # Anthropic's prompt_tokens already excludes cache reads/writes.
                        self.ledger.add_llm_usage(
                            prompt_tokens=u.prompt_tokens,
                            cache_read=u.cache_read_input_tokens,
                            cache_write=u.cache_creation_input_tokens,
                            completion=u.completion_tokens,
                        )
                    elif isinstance(m, TTSUsageMetricsData):
                        self.ledger.add_tts_chars(m.value)

    def summary(self, session_duration_sec: float) -> dict:
        return self.ledger.summary(session_duration_sec)

from session_state import (
    TRANSCRIPTS_DIR,
    VOICE_TUTOR_DIR,
    _format_full_transcript_block,
    _format_memory_date,
    _format_session_time,
    append_to_memory,
    has_min_user_turns,
    load_memory,
    load_most_recent_transcript_block,
    load_profile,
)
from session_naming import safe_session_id, session_analysis_filename
ARTIFACTS_DIR = VOICE_TUTOR_DIR / "artifacts"
# Accumulating memory: one dated section per session, append-only.
# Future: when this file exceeds ~2K tokens, compact older entries by summarizing
# everything before a cutoff date into a single "before April X" block. Not today's
# problem — revisit when the memory block starts dominating the system prompt.
COST_LOG_PATH = Path.home() / "second-brain" / "products" / "voice-tutor" / "validation" / "cost-log.md"
# The append-only per-session ledger. Named session-log.jsonl (cost is one field
# per row, not the whole story); the human-facing report keeps the cost-log.md name.
SESSION_LOG_JSONL_PATH = COST_LOG_PATH.with_name("session-log.jsonl")
SESSION_ANALYSIS_DIR = Path.home() / "second-brain" / "products" / "voice-tutor" / "session-analyses"
# Gate the post-session summary + analysis on whether the USER actually spoke,
# not on wall-clock duration. A session is summarized + analyzed iff the user
# took at least MIN_USER_TURNS turns; the tutor's hidden kickoff and its replies
# don't count (see session_state.has_min_user_turns). Duration was a poor proxy —
# a real 103s/3-turn session was silently dropped at the old 120s floor. Default
# 1 (any session where the user spoke at all); env-tunable so the floor can be
# raised without a rebuild.
MIN_USER_TURNS = int(os.getenv("VOICE_TUTOR_MIN_USER_TURNS", "1"))

# Cartesia Sonic-3 speed multiplier. Valid range [0.6, 1.5]; 1.0 is default.
# Unset → omit the override entirely so behavior matches the pre-flag baseline.
# Set VOICE_TUTOR_TTS_SPEED=0.85 in .env.local for a noticeably slower cadence.
_tts_speed_override = os.getenv("VOICE_TUTOR_TTS_SPEED")
TTS_SPEED: float | None = float(_tts_speed_override) if _tts_speed_override else None

ANALYSIS_PROMPT = """\
Analyze this voice conversation session transcript. Produce a structured markdown \
document with the following sections. Be concise and specific — no filler.

## Session overview
A markdown table with: Duration, Turns, Total cost, Cost/min, LLM cost, STT cost, TTS cost.

## On-demand tool calls
If any tool calls occurred, a table with: Timestamp (HH:MM:SS), Page, Latency to first audio. \
Note any patterns (back-to-back lookups, failed lookups, filler speech before lookup). \
If no tool calls, say "None this session."

## Topics covered
Numbered list of the main topics/threads discussed, with a one-sentence summary each.

## Knowledge sources
Estimate how much of the conversation drew from each pre-loaded context source vs the \
LLM's general knowledge vs {user_name}'s own ideas. Use a simple table with columns: Source, \
Approx turns, Notes. Sources are:
- "On-demand wiki pages" — content fetched via read_wiki_page tool calls during the session
- "Pre-loaded wiki INDEX" — the wiki table of contents in the system prompt (titles and one-line descriptions only, not full page content)
- "Prior-session memory" — the "What we've discussed" block summarizing past sessions
- "Most-recent transcript" — the verbatim block from the previous session
- "General LLM knowledge" — things the model knows independent of any loaded context
- "{user_name}'s own knowledge/ideas" — claims, framing, or context {user_name} introduced themselves

Be specific in Notes about which facts came from which source — don't lump everything \
pre-loaded into one bucket. Note the key finding: which source(s) carried the conversation, \
and was the wiki itself central or peripheral?

## Interaction quality notes
Bullet points on: pacing issues (did {user_name} ask to slow down?), STT errors (misheard words), \
interruptions, response length compliance, and anything else notable about the interaction dynamics.

Here is the session data:

### Usage summary
{usage_json}

### Tool calls
{tool_calls_json}

### Transcript
{transcript_json}
"""

STUDY_BASE_INSTRUCTION = (
    "You are a study companion helping the user understand a specific document "
    "they have loaded. Help them engage actively — ask what they want to focus "
    "on, explain concepts when asked, surface connections, push back when their "
    "understanding is shaky, and let them lead the direction.\n\n"
    "You also have a private claim map (below, after the document): the document "
    "broken into discrete claims. It is YOURS — never read claims aloud verbatim, "
    "never mention claim numbers or the map's existence. Use it to know what's "
    "been covered and what hasn't.\n\n"
    "When the user is actively leading, follow. When they stall, drift, or ask "
    "where to go next, steer toward uncovered claims with a natural bridge (\"that "
    "actually connects to something we haven't touched...\"). You can't ask "
    "someone to explain what they haven't encountered — so introduce an uncovered "
    "claim first, then later invite them to articulate it back in their own words. "
    "Mentioning isn't understanding: \"that makes sense\" isn't articulation — "
    "circle back to those gently. Prefer application, prediction, or contrast "
    "questions over \"repeat that back.\"\n\n"
    "Reference the document directly. Quote short passages when useful. Don't "
    "summarize the whole thing unprompted.\n\n"
    "Keep responses tight. One thought at a time. This is voice — long monologues "
    "don't work."
)

# The passive opener the session-aware behavior replaces. Copied VERBATIM from
# STUDY_BASE_INSTRUCTION above — if that string is reworded, update this too or
# the module-level assert below will fire at import.
_PASSIVE_OPENER_LINE = (
    "Help them engage actively — ask what they want to focus on, explain "
    "concepts when asked, surface connections, push back when their "
    "understanding is shaky, and let them lead the direction."
)

# Replacement: the same engagement sentence minus the passive opener, then the
# new "## Opening the session" section (first wording; tune by ear post-ship).
_OPENING_SECTION = (
    "Help them engage actively — explain concepts when asked, surface "
    "connections, push back when their understanding is shaky, and let them "
    "lead the direction once the session is underway.\n\n"
    "## Opening the session\n"
    "Do NOT open with a generic greeting or an open-ended \"what do you want "
    "to focus on?\". Open with orientation, in one short spoken turn:\n"
    "- If there is NO \"Where you left off\" section below, this is a first "
    "session. Give a one-breath, high-level lay-of-the-land of what this "
    "document covers (two to four beats, synthesized from the document and "
    "your private claim map — never recite the map), then propose starting "
    "with the foundations and building up, then invite them to redirect "
    "(\"…sound good, or is there something specific you want to start "
    "with?\").\n"
    "- If the user declines your proposed starting point, offer two or three "
    "concrete alternative areas drawn from the claim map — the map is your "
    "menu — rather than asking an open-ended question.\n"
    "- If there IS a \"Where you left off\" section below, this is a returning "
    "session. Briefly recap what was covered last time and what was left "
    "open, then ask whether they want to pick up where they left off or "
    "revisit something first — and let them choose. Do not push a next step.\n"
    "Keep the opening to a few sentences — this is voice — then follow their "
    "lead."
)

STUDY_BASE_INSTRUCTION_WITH_OPENING = STUDY_BASE_INSTRUCTION.replace(
    _PASSIVE_OPENER_LINE, _OPENING_SECTION
)
assert STUDY_BASE_INSTRUCTION_WITH_OPENING != STUDY_BASE_INSTRUCTION, (
    "_PASSIVE_OPENER_LINE did not match STUDY_BASE_INSTRUCTION — the opening "
    "section was not injected. Re-copy the line verbatim from the base."
)

# Hidden first-turn trigger. The default produces a generic greeting; the study
# variant (flag ON) triggers the "Opening the session" behavior in the study base.
DEFAULT_KICKOFF_MESSAGE = "Say hello and introduce yourself briefly."
STUDY_KICKOFF_MESSAGE = (
    "Begin the study session now. Open by orienting the user per your "
    '"Opening the session" instructions — do not just say a generic hello.'
)


def kickoff_message(study: bool) -> str:
    """The hidden opening turn. Study + flag ON → the plan-triggering message;
    otherwise the legacy greeting (regular mode, or study with the flag off)."""
    if study and SESSION_OPENING:
        return STUDY_KICKOFF_MESSAGE
    return DEFAULT_KICKOFF_MESSAGE


ARTIFACT_PROMPT = """\
You are writing a markdown recap of a voice-mode study session about a specific \
document. Output ONLY markdown — no preamble, no trailing prose.

SCOPE — read carefully:
- The recap covers ONLY what was actually discussed in the transcript below.
- The document is provided as REFERENCE — use it to quote passages the user \
pointed at, to disambiguate vague references, and to get terms/names right. \
Do NOT summarize sections of the document that did not come up in conversation.
- If the conversation was short or covered only one topic, the recap is short \
and covers only that topic. Do not pad. Do not invent topics.
- If a topic was named but not actually explored, it belongs in "Open threads", \
not "Key points".

Use this structure exactly:

# Study session — {doc_title}
Duration: {duration_mmss}

## What we covered
- short bullets, one per topic ACTUALLY discussed (not topics merely mentioned)

## Key points
### <topic>
Substantive notes on what was said in the conversation about this topic — \
paraphrase the user's reasoning and the tutor's responses, capture concrete \
claims that were made, quote the document only where it sharpens a point that \
came up. Two to four short paragraphs per topic. Omit this section entirely \
if nothing was discussed in enough depth to warrant it.

## Open threads
Things raised but not resolved — questions to come back to. One bullet each. \
Skip this section if there are none.

### Transcript
{transcript_json}

### Document (reference only — do not summarize)
{doc_text}
"""

BASE_INSTRUCTION = (
    "You are a friendly, curious conversational partner and tutor. "
    "Be concise. Say one thought at a time, then let {user_name} respond. "
    "One to two sentences per turn. Never monologue. "
    "Be warm but not sycophantic. Never repeat yourself. "
    "You know {user_name} from prior conversations. "
    "Reference past topics naturally when relevant, but don't force it."
)

# Appended to BASE_INSTRUCTION only when the wiki module is active. Lives in
# bot.py rather than wiki.py because it's about persona framing, not the tool
# itself — the wiki module owns the actual section block and usage rules.
WIKI_TAGLINE = (
    " You have access to {user_name}'s personal knowledge wiki — use it to teach, "
    "connect ideas, and reference things they've been reading and learning about."
)

# Restated at the very end of the system prompt so it stays close to the model's
# next-token decision after a long doc / wiki / memory block. Recency matters.
BREVITY_REMINDER = (
    "\n\n# Reminder\n\n"
    "Voice mode. One thought per turn. One to two sentences. "
    "Then stop and let the user respond. Never monologue. "
    "Speak deliberately — use commas and brief pauses; don't rush."
)

# Appended after BREVITY_REMINDER in study mode. memory.md is ~2400 tokens of
# open-chat session summaries; without recency-priming, that volume drowns out
# the thin STUDY_BASE_INSTRUCTION at the top and the model drifts toward general
# conversation. This reminder pulls the persona back at the last moment.
STUDY_REMINDER = (
    "\n\n# Study mode\n\n"
    "You're a study companion for the document above. The memory section is "
    "background — reference past topics only when they directly illuminate "
    "the document. Stay focused on what's in front of you. "
    "Track coverage against the private claim map; steer to gaps when the user "
    "isn't actively leading."
)


WIKI_ENABLED = os.getenv("WIKI_ENABLED", "true").lower() in ("1", "true", "yes")


SUMMARY_PROMPT = """\
Summarize this voice tutoring conversation in 3-5 short bullet points. Cover what \
was discussed, any decisions {user_name} made, and any open questions or next steps. Be \
terse — this is loaded as context into a future voice session so the tutor can \
pick up continuity. Output only the bullets (one per line, starting with "- "). \
No preamble, no trailing prose.

### Transcript
{transcript_json}
"""


def _summary_prompt(user_name: str, transcript: dict) -> str:
    return SUMMARY_PROMPT.format(
        user_name=user_name,
        transcript_json=json.dumps(transcript, indent=2),
    )


def _analysis_prompt(
    user_name: str, summary: dict, tool_calls: list[dict], transcript: dict
) -> str:
    return ANALYSIS_PROMPT.format(
        user_name=user_name,
        usage_json=json.dumps(summary, indent=2),
        tool_calls_json=json.dumps(tool_calls, indent=2) if tool_calls else "[]",
        transcript_json=json.dumps(transcript, indent=2),
    )


def generate_session_summary(user_id: str, stem: str, transcript: dict) -> dict | None:
    prompt = _summary_prompt(identity.display_name(user_id), transcript)
    try:
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
    except Exception as e:
        print(f"[session-summary] failed: {e}", file=sys.stderr, flush=True)
        return None
    out_path = TRANSCRIPTS_DIR / user_id / f"{stem}.summary.md"
    out_path.write_text(text + "\n")
    print(f"[session-summary] wrote {out_path}", file=sys.stderr, flush=True)
    return {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}


def generate_session_analysis(
    user_id: str,
    stem: str,
    transcript: dict,
    summary: dict,
    tool_calls: list[dict],
    session_start: datetime,
    session_id: str | None,
) -> dict | None:
    prompt = _analysis_prompt(identity.display_name(user_id), summary, tool_calls, transcript)
    try:
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        analysis = resp.content[0].text
    except Exception as e:
        print(f"[session-analysis] failed: {e}", file=sys.stderr, flush=True)
        return None
    header = f"# Session Analysis — {stem}\n\n"
    analysis_dir = SESSION_ANALYSIS_DIR / user_id
    analysis_dir.mkdir(parents=True, exist_ok=True)
    out_path = analysis_dir / session_analysis_filename(session_start, session_id)
    out_path.write_text(header + analysis)
    print(f"[session-analysis] wrote {out_path}", file=sys.stderr, flush=True)
    return {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}


async def generate_artifact(user_id: str, session_id: str, study_meta: dict, transcript: dict, duration_sec: float):
    """Fire-and-forget Haiku call writing ~/.voice-tutor/artifacts/<user_id>/<session_id>.md.

    Writes a separate row to session-log.jsonl with kind="artifact" so the cost is
    auditable without retroactively patching the synchronous session row.
    """
    duration_mmss = f"{int(duration_sec // 60)}:{int(duration_sec % 60):02d}"
    prompt = ARTIFACT_PROMPT.format(
        doc_title=study_meta["doc_title"],
        doc_text=study_meta["doc_text"],
        duration_mmss=duration_mmss,
        transcript_json=json.dumps(transcript, indent=2),
    )
    try:
        client = anthropic.Anthropic()
        resp = await asyncio.to_thread(
            client.messages.create,
            model="claude-haiku-4-5-20251001",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        markdown = resp.content[0].text
    except Exception as e:
        print(f"[artifact] failed for session_id={session_id}: {e}", file=sys.stderr, flush=True)
        return

    user_artifacts_dir = ARTIFACTS_DIR / user_id
    user_artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_path = user_artifacts_dir / f"{session_id}.md"
    out_path.write_text(markdown)
    print(f"[artifact] wrote {out_path}", file=sys.stderr, flush=True)

    in_tok = resp.usage.input_tokens
    out_tok = resp.usage.output_tokens
    cost = (
        in_tok / 1_000_000 * PRICE_ANTHROPIC_HAIKU_INPUT_PER_MTOK
        + out_tok / 1_000_000 * PRICE_ANTHROPIC_HAIKU_OUTPUT_PER_MTOK
    )
    row = {
        "kind": "artifact",
        "session_id": session_id,
        "user_id": user_id,
        "document_id": study_meta["document_id"],
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "cost_usd": round(cost, 4),
    }
    SESSION_LOG_JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SESSION_LOG_JSONL_PATH.open("a") as f:
        f.write(json.dumps(row) + "\n")


def _claim_map_block(claim_texts: list[str]) -> str:
    """Render the private claim-map section: header + numbered claim texts only.

    Claim TEXT only — no anchors, offsets, or ids: the tutor steers on the
    articulable claims, and the extra fields only bloat the prompt. Positioned
    (by the caller) immediately after the document and before the reminders, so
    the map sits with the material it decomposes.
    """
    numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(claim_texts, 1))
    return f"\n## Claim map (private — never reveal)\n\n{numbered}"


def _previously_block(previously: dict) -> str:
    """Render the prior-session recap for the returning-session opener. Accepts
    either shape from study_history.parse_recap_sections: the parsed
    {"covered", "open_threads"} or the {"fallback_text"} fallback."""
    header = "\n# Where you left off on this document\n"
    guide = (
        "\n(This is a returning session. Use this to recap briefly and offer to "
        "continue or revisit — see \"Opening the session\". Never read it verbatim.)"
    )
    if "fallback_text" in previously:
        return f"{header}\nRecap of the previous session:\n\n{previously['fallback_text']}\n{guide}"

    lines = [header, "\nIn the previous session you covered:"]
    for item in previously.get("covered", []):
        lines.append(f"- {item}")
    open_threads = previously.get("open_threads", [])
    if open_threads:
        lines.append("\nLeft open:")
        for item in open_threads:
            lines.append(f"- {item}")
    lines.append(guide)
    return "\n".join(lines)


def build_system_instruction(user_id: str, study: dict | None = None) -> str:
    """Assemble the system prompt.

    Regular mode: base + profile + wiki INDEX + memory + most-recent transcript.
    Study mode (study={doc_title, doc_text, claims?}): base + profile + memory +
    the doc + (optional claim map). Study mode skips the most-recent transcript,
    the wiki INDEX, and the wiki tagline — the doc replaces those as the session's
    focus. ``study["claims"]`` (a list of claim-text strings) is injected as the
    private claim map after the document; when absent/empty (claims not yet
    extracted at session start) the map is simply omitted — the session degrades
    to plain study mode rather than blocking on extraction.

    ``user_id`` is required and scopes every per-user read (profile, memory,
    most-recent transcript) — the isolation boundary that prevents one user's
    context from leaking into another's session.
    """
    profile = load_profile(user_id)
    name = identity.display_name(user_id)

    if study is not None:
        base = STUDY_BASE_INSTRUCTION_WITH_OPENING if SESSION_OPENING else STUDY_BASE_INSTRUCTION
        parts = [base]
        if profile:
            parts.append(f"\n## About the person you're talking to\n\n{profile}")
        memory = load_memory(user_id)
        if memory:
            parts.append(
                f"\n# Background — {name}'s prior topics (reference only if directly relevant to the document)\n\n"
                + memory
            )
        previously = study.get("previously") if SESSION_OPENING else None
        if previously:
            parts.append(_previously_block(previously))
        parts.append(f"\n## Document: {study['doc_title']}\n\n{study['doc_text']}")
        claim_texts = study.get("claims")
        if claim_texts:
            parts.append(_claim_map_block(claim_texts))
        parts.append(BREVITY_REMINDER)
        parts.append(STUDY_REMINDER)
        return "\n".join(parts)

    base = (BASE_INSTRUCTION + (WIKI_TAGLINE if WIKI_ENABLED else "")).replace("{user_name}", name)
    parts = [base]

    if profile:
        parts.append(f"\n## About the person you're talking to\n\n{profile}")

    if WIKI_ENABLED:
        wiki_block = wiki.system_prompt_block()
        if wiki_block:
            parts.append(wiki_block)

    memory = load_memory(user_id)
    if memory:
        parts.append(f"\n# What we've discussed\n\n{memory}")

    most_recent = load_most_recent_transcript_block(user_id)
    if most_recent:
        parts.append(f"\n# Most recent session\n\n{most_recent}")

    parts.append(BREVITY_REMINDER)
    return "\n".join(parts)


def static_prompt_hash(study: bool) -> str:
    """Content hash of the STATIC prompt scaffolding — the base instruction plus
    reminders for the given mode, and nothing dynamic.

    Deliberately excludes profile, memory, the document, and the claim map: those
    vary per person / per session / per doc. Hashing only the fixed scaffolding
    lets a ledger row attribute a session to the exact PROMPT VERSION that ran it,
    stable across documents — mirroring the ``source_hash`` pattern claims.py uses
    to key its cache. Bump-visible: any wording edit to the base/reminder strings
    changes the hash, so a prompt change is traceable across sessions.
    """
    if study:
        if SESSION_OPENING:
            static = (
                STUDY_BASE_INSTRUCTION_WITH_OPENING
                + BREVITY_REMINDER
                + STUDY_REMINDER
                + STUDY_KICKOFF_MESSAGE
            )
        else:
            # Byte-identical to the pre-change input — preserves the historical
            # hash for flag-off sessions. Do NOT add the kickoff here.
            static = STUDY_BASE_INSTRUCTION + BREVITY_REMINDER + STUDY_REMINDER
    else:
        static = BASE_INSTRUCTION + (WIKI_TAGLINE if WIKI_ENABLED else "") + BREVITY_REMINDER
    return hashlib.sha256(static.encode("utf-8")).hexdigest()


async def bot(runner_args):
    transport = SmallWebRTCTransport(
        webrtc_connection=runner_args.webrtc_connection,
        params=TransportParams(audio_out_enabled=True, audio_in_enabled=True),
    )

    body = getattr(runner_args, "body", None) or {}
    user_id = body.get("user_id")  # server-stamped by app.offer()
    if not user_id:
        print("[bot] no user_id on session body; refusing (fail closed)", file=sys.stderr, flush=True)
        return  # fail closed — no session
    document_id = body.get("document_id")
    session_id_override = body.get("session_id")

    study_meta: dict | None = None
    if document_id:
        loaded = documents.load_document(user_id, document_id)
        if loaded is None:
            print(f"[bot] document_id={document_id} not found; falling back to regular mode", file=sys.stderr, flush=True)
        else:
            doc_title, doc_text = loaded
            study_meta = {
                "user_id": user_id,
                "document_id": document_id,
                "doc_title": doc_title,
                "doc_text": doc_text,
                # Sanitize the client-controlled session id to a single path
                # component HERE — the one point it enters study_meta — so every
                # downstream writer that uses it as a filename stem (the .prompt.txt
                # / .json / .summary.md / .usage.json / recap-artifact paths below)
                # inherits containment and can't be steered into another user's dir.
                "session_id": safe_session_id(session_id_override or document_id),
            }

    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        settings=DeepgramSTTService.Settings(model="nova-3", language="en"),
    )

    # Claim-map steering (study mode only). Read the claim set from cache ONLY —
    # a hash-verified, non-blocking read. Extraction is warmed when the doc is
    # selected (POST /api/documents/{id}/claims/prepare); if the session starts
    # before it finishes, load_fresh_claims returns None and we degrade to plain
    # study mode rather than blocking the pipeline on a 30-60s live extraction.
    study_arg = None
    # The full claim objects (id + text) are kept for the teardown coverage judge,
    # which needs the ids; the prompt gets text only. None when the map wasn't
    # ready at session start — coverage is then skipped, exactly as steering is.
    study_claims = None
    if study_meta:
        cached = claims.load_fresh_claims(user_id, study_meta["document_id"], study_meta["doc_text"])
        study_claims = cached
        claim_texts = [c.claim for c in cached] if cached else None
        if claim_texts:
            print(f"[bot] claim map ready: {len(claim_texts)} claims", flush=True)
        else:
            print("[bot] claim map not ready; study session runs without steering", flush=True)
        previously = None
        if SESSION_OPENING:
            try:
                previously = study_history.previous_session_recap(
                    user_id, study_meta["document_id"], study_meta["session_id"]
                )
            except Exception:
                print(
                    "[bot] previous_session_recap failed; degrading to first-session opener",
                    file=sys.stderr,
                    flush=True,
                )
                previously = None
        study_arg = {
            "doc_title": study_meta["doc_title"],
            "doc_text": study_meta["doc_text"],
            "claims": claim_texts,
            "previously": previously,
        }

    system_instruction = build_system_instruction(user_id, study=study_arg)
    prompt_hash = static_prompt_hash(study=study_meta is not None)

    user_tx = TRANSCRIPTS_DIR / user_id
    if study_meta:
        user_tx.mkdir(parents=True, exist_ok=True)
        (user_tx / f"{study_meta['session_id']}.prompt.txt").write_text(system_instruction)

    llm = AnthropicLLMService(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        settings=AnthropicLLMService.Settings(
            model="claude-sonnet-4-5-20250929",
            system_instruction=system_instruction,
            enable_prompt_caching=True,
            max_tokens=1024,
            temperature=0.7,
        ),
    )

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        settings=CartesiaTTSService.Settings(
            model="sonic-3",
            voice="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
            generation_config=GenerationConfig(speed=TTS_SPEED) if TTS_SPEED is not None else None,
        ),
    )

    tools = [] if study_meta else ([wiki.tool_schema()] if WIKI_ENABLED else [])
    context = LLMContext(tools=ToolsSchema(standard_tools=tools))
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    usage = UsageAccumulator()

    if WIKI_ENABLED and not study_meta:
        llm.register_function("read_wiki_page", wiki.make_tool_handler(usage.mark_tool_call))

    pipeline = Pipeline([
        transport.input(),
        stt,
        user_aggregator,
        llm,
        tts,
        transport.output(),
        assistant_aggregator,
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        observers=[usage],
    )

    # Transcript accumulation
    session_start = datetime.now()
    turns: list[dict] = []

    @user_aggregator.event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(aggregator, strategy, message):
        turns.append({
            "role": "user",
            "content": message.content,
            "timestamp": message.timestamp,
        })

    @assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(aggregator, message):
        turns.append({
            "role": "assistant",
            "content": message.content,
            "timestamp": message.timestamp,
        })

    async def run_coverage_judge(transcript: dict):
        """Judge this session's coverage and write the sidecar + its ledger row.

        Runs at teardown, in the same slot as summary/analysis/recap, but placed
        FIRST — immediately after the transcript is on disk — because everything
        it needs exists at that moment and every later teardown step is another
        chance for the process to die and take coverage with it (the known
        hard-stop fragility that already loses transcripts and recaps).

        The judge call itself is a blocking 10-40s Haiku request, so it is run in
        a worker thread: the event loop stays free for any OTHER live session
        instead of stalling for the length of the call. Nothing here touches the
        pipeline — the conversation is already over by the time it runs.

        FAILURE CONTRACT: every failure degrades to no coverage data for this
        session, logged. It must never break the transcript, recap, cost log, or
        ledger that follow it — hence the blanket except around each half.
        """
        if not study_claims:
            print("[coverage] no claim map for this session; skipping", file=sys.stderr, flush=True)
            return
        try:
            sidecar, cost_row = await asyncio.to_thread(
                coverage_store.judge_session,
                user_id=user_id,
                session_id=study_meta["session_id"],
                document_id=study_meta["document_id"],
                source_hash=claims.source_hash_of(study_meta["doc_text"]),
                claim_objs=study_claims,
                transcript=transcript,
            )
        except Exception as e:  # noqa: BLE001 - coverage never breaks teardown
            print(f"[coverage] judge raised, skipping coverage: {e}", file=sys.stderr, flush=True)
            return

        # LEDGER FIRST, in its OWN try. The spend is already incurred by this
        # point and is the one fact that cannot be reconstructed later; the
        # sidecar can (re-judge with --force). Writing it second, inside the same
        # try as the sidecar, meant a failing sidecar write also swallowed the
        # cost row — losing the record precisely in the failure case the
        # write-either-way rule exists for.
        try:
            # Tokens are omitted from cost_row when NO call reported them (the
            # partial-measurement rule), so only price what was actually
            # observed: a missing count must not become a confident $0.0000 in
            # the ledger. usage_complete on the row says which case this is.
            if "input_tokens" in cost_row or "output_tokens" in cost_row:
                cost = (
                    cost_row.get("input_tokens", 0) / 1_000_000 * PRICE_ANTHROPIC_HAIKU_INPUT_PER_MTOK
                    + cost_row.get("output_tokens", 0) / 1_000_000 * PRICE_ANTHROPIC_HAIKU_OUTPUT_PER_MTOK
                )
                cost_row["cost_usd"] = round(cost, 4)
            SESSION_LOG_JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
            with SESSION_LOG_JSONL_PATH.open("a") as f:
                f.write(json.dumps(cost_row) + "\n")
        except Exception as e:  # noqa: BLE001 - coverage never breaks teardown
            print(f"[coverage] ledger write failed: {e}", file=sys.stderr, flush=True)

        try:
            if sidecar is not None:
                # No overwrite= here, deliberately: coverage is append-only, so a
                # reused/crafted session_id cannot clobber an earlier session's
                # record. write_sidecar returns None when it declines.
                path = coverage_store.write_sidecar(
                    user_id, study_meta["session_id"], sidecar
                )
                if path is None:
                    print(
                        f"[coverage] a sidecar already exists for session "
                        f"{study_meta['session_id']}; not overwriting",
                        file=sys.stderr, flush=True,
                    )
                else:
                    print(
                        f"[coverage] wrote {path} — "
                        f"{sidecar['covered_count']}/{sidecar['claims_total']} claims covered",
                        file=sys.stderr, flush=True,
                    )
            else:
                print(
                    f"[coverage] judge failed: {cost_row.get('error')}",
                    file=sys.stderr, flush=True,
                )
        except Exception as e:  # noqa: BLE001 - coverage never breaks teardown
            print(f"[coverage] sidecar write failed: {e}", file=sys.stderr, flush=True)

    async def save_transcript():
        if not turns:
            return
        user_tx.mkdir(parents=True, exist_ok=True)
        stem = study_meta["session_id"] if study_meta else session_start.strftime("%Y-%m-%d-%H%M%S")
        session_end = datetime.now()
        transcript = {
            "session_start": session_start.isoformat(),
            "session_end": session_end.isoformat(),
            "turn_count": len(turns),
            "turns": turns,
        }
        (user_tx / f"{stem}.json").write_text(json.dumps(transcript, indent=2))

        duration_sec = (session_end - session_start).total_seconds()

        # THE TWO LONG MODEL CALLS START HERE, CONCURRENTLY, AND NOTHING WAITS ON
        # THEM. Both the recap and the coverage judge are minutes-scale work that
        # nothing downstream depends on, so running either as a blocking STEP
        # makes every artifact behind it late. The user is watching: the client
        # polls for 60s (30 x 2s in static/study.html) and then gives up, showing
        # "Recap didn't generate" and an unfilled diagnostics panel.
        #
        # Measured on real sessions (2026-08-04), both regressions came from
        # ordering alone, not from the work itself:
        #   * judge awaited BEFORE the recap  -> recap at 74s, past the cap;
        #   * judge awaited before summary/analysis -> those at 69s, also past it,
        #     so cost + analysis never rendered.
        # Started as tasks, the recap lands ~12s and summary/analysis/cost ~20s,
        # while the 50s judge runs alongside instead of in front.
        #
        # Coverage still STARTS as early as it can be correct (the transcript is
        # on disk, the claim map is in hand), which is what the hard-stop
        # durability argument actually required — being early, not being blocking.
        background: list[asyncio.Task] = []
        if study_meta:
            background.append(asyncio.create_task(generate_artifact(
                user_id=user_id,
                session_id=study_meta["session_id"],
                study_meta=study_meta,
                transcript=transcript,
                duration_sec=duration_sec,
            )))
        # Gated on the same user-turn floor as summary/analysis: a session the
        # user never spoke in has nothing to judge and should not be billed.
        if study_meta and COVERAGE_JUDGE and has_min_user_turns(turns, MIN_USER_TURNS):
            background.append(asyncio.create_task(run_coverage_judge(transcript)))
        if background:
            # Yield ONCE so both tasks actually reach their first await — each
            # hands its model call to a worker thread — BEFORE the summary and
            # analysis calls below monopolise the event loop. Those two are
            # synchronous and block the loop for ~15s; without this yield the
            # tasks would not start until the loop is next free, re-serialising
            # exactly what this restructure exists to parallelise.
            await asyncio.sleep(0)

        # TRY/FINALLY, not a plain sequence. Everything below is fast, local
        # work — but if any of it raises (a full disk on .usage.json or
        # cost-log.md, an OSError in append_to_memory), the await loop at the
        # end is skipped, and the two background tasks are then referenced
        # ONLY by this dead frame's local list — the exact weak-reference GC
        # hazard noted below, now with in-flight model calls to destroy. It
        # would also skip the caller's `await task.cancel()`, leaving the
        # pipeline running. The finally makes awaiting them unconditional.
        try:
            summary = usage.summary(duration_sec)

            # Run post-session Haiku calls before finalizing the cost log so their
            # tokens roll into the row. They were previously off-the-books — small
            # (~$0.025/session) but unaccounted for vs the Anthropic dashboard.
            post_input = 0
            post_output = 0
            if has_min_user_turns(turns, MIN_USER_TURNS):
                u = generate_session_summary(user_id, stem, transcript)
                if u:
                    post_input += u["input_tokens"]
                    post_output += u["output_tokens"]
                summary_path = user_tx / f"{stem}.summary.md"
                if summary_path.exists():
                    append_to_memory(user_id, transcript, summary_path.read_text())

            if has_min_user_turns(turns, MIN_USER_TURNS):
                u = generate_session_analysis(
                    user_id,
                    stem,
                    transcript,
                    summary,
                    usage.tool_calls,
                    session_start,
                    study_meta["session_id"] if study_meta else None,
                )
                if u:
                    post_input += u["input_tokens"]
                    post_output += u["output_tokens"]

            post_cost = (
                post_input / 1_000_000 * PRICE_ANTHROPIC_HAIKU_INPUT_PER_MTOK
                + post_output / 1_000_000 * PRICE_ANTHROPIC_HAIKU_OUTPUT_PER_MTOK
            )
            summary["post_session"] = {
                "input_tokens": post_input,
                "output_tokens": post_output,
                "cost_usd": round(post_cost, 4),
            }
            summary["total_cost_usd"] = round(summary["total_cost_usd"] + post_cost, 4)

            (user_tx / f"{stem}.usage.json").write_text(json.dumps(summary, indent=2))
            mins = summary["session_duration_sec"] / 60
            line = (
                f"Session: {mins:.1f}min · {len(turns)} turns · "
                f"${summary['total_cost_usd']:.3f} "
                f"(llm ${summary['llm']['cost_usd']:.3f} · "
                f"stt ${summary['stt']['cost_usd']:.3f} · "
                f"tts ${summary['tts']['cost_usd']:.3f} · "
                f"post ${summary['post_session']['cost_usd']:.3f})"
            )
            print(line, flush=True)

            # The "LLM" column in cost-log.md now means total LLM spend (live Sonnet
            # + post-session Haiku) to match what the Anthropic dashboard reports.
            # JSONL keeps the breakdown.
            llm_total = summary["llm"]["cost_usd"] + summary["post_session"]["cost_usd"]

            COST_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            if not COST_LOG_PATH.exists():
                COST_LOG_PATH.write_text(
                    "# Voice Tutor — Cost Log\n\n"
                    "One row per session. Costs are computed from ground-truth usage\n"
                    "(TTS audio bytes, LLM token counts, Deepgram streamed minutes).\n"
                    "Rates last verified 2026-04-15 against provider pricing pages.\n"
                    "The LLM column includes both live Sonnet and post-session Haiku\n"
                    "(matching the Anthropic dashboard); see `session-log.jsonl` for the\n"
                    "live-vs-post-session breakdown.\n"
                    "Per-session raw usage is logged to `session-log.jsonl` for auditing\n"
                    "(starting 2026-04-15 — earlier sessions have no raw-usage sidecar).\n\n"
                    "| Session start | Duration | Turns | Total | LLM | STT | TTS |\n"
                    "|---|---|---|---|---|---|---|\n"
                )
            row = (
                f"| {session_start.strftime('%Y-%m-%d %H:%M')} "
                f"| {mins:.1f} min "
                f"| {len(turns)} "
                f"| ${summary['total_cost_usd']:.3f} "
                f"| ${llm_total:.3f} "
                f"| ${summary['stt']['cost_usd']:.3f} "
                f"| ${summary['tts']['cost_usd']:.3f} |\n"
            )
            with COST_LOG_PATH.open("a") as f:
                f.write(row)

            jsonl_entry = {
                "kind": "session",
                "user_id": user_id,
                "session_id": session_start.strftime("%Y-%m-%dT%H%M%S"),
                "session_start": session_start.isoformat(),
                "session_end": session_end.isoformat(),
                "session_duration_sec": summary["session_duration_sec"],
                "turns": len(turns),
                "tts_chars": summary["tts"]["chars"],
                "tts_audio_sec_observed": summary["tts"]["audio_sec_observed"],
                "stt_audio_sec_observed": summary["stt"]["audio_sec_observed"],
                "stt_minutes_billed": summary["stt"]["minutes"],
                "llm_uncached_input_tokens": summary["llm"]["uncached_input_tokens"],
                "llm_cache_read_tokens": summary["llm"]["cache_read_tokens"],
                "llm_cache_write_tokens": summary["llm"]["cache_write_tokens"],
                "llm_output_tokens": summary["llm"]["output_tokens"],
                "post_session_input_tokens": post_input,
                "post_session_output_tokens": post_output,
                "cost_llm_usd": summary["llm"]["cost_usd"],
                "cost_stt_usd": summary["stt"]["cost_usd"],
                "cost_tts_usd": summary["tts"]["cost_usd"],
                "cost_post_session_usd": summary["post_session"]["cost_usd"],
                "cost_total_usd": summary["total_cost_usd"],
                "prompt_hash": prompt_hash,
                "tool_calls": usage.tool_calls,
            }
            if study_meta:
                jsonl_entry["session_id"] = study_meta["session_id"]
                jsonl_entry["mode"] = "study"
                jsonl_entry["document_id"] = study_meta["document_id"]
            with SESSION_LOG_JSONL_PATH.open("a") as f:
                f.write(json.dumps(jsonl_entry) + "\n")

        finally:
            # Now — and only now, with every fast artifact already on disk — wait for
            # the two background model calls started at the top. Awaiting them HERE
            # rather than not at all is what keeps them from being orphaned: the
            # caller cancels the pipeline task immediately after this function
            # returns, and a pending bare task would be torn down with the process
            # (asyncio also only holds a WEAK reference to a running task, so an
            # un-awaited one can be garbage-collected mid-flight).
            #
            # Each task already contains its own failure handling and must never
            # break teardown, so a raise here is caught and logged, not propagated.
            # On the normal path every artifact above is already on disk; on the
            # exceptional path (the finally) this still runs, so in-flight work is
            # awaited to completion instead of being dropped mid-call.
            for task_ in background:
                try:
                    await task_
                except Exception as e:  # noqa: BLE001 - background work never breaks teardown
                    print(f"[teardown] background task failed: {e}", file=sys.stderr, flush=True)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        context.add_message({
            "role": "user",
            "content": kickoff_message(study=study_meta is not None),
        })
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        # save_transcript owns its own failure handling and awaits its background
        # work in a finally — but an unexpected raise must still not cost the
        # pipeline its shutdown. Cancelling is the one step that has to happen
        # whatever teardown did, so it goes in a finally of its own.
        try:
            await save_transcript()
        except Exception as e:  # noqa: BLE001 - teardown never blocks the shutdown
            print(f"[teardown] save_transcript failed: {e}", file=sys.stderr, flush=True)
        finally:
            await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)


if __name__ == "__main__":
    from pipecat.runner.run import main
    main()
