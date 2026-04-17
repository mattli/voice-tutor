import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import anthropic

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    InputAudioRawFrame,
    LLMRunFrame,
    MetricsFrame,
    TTSAudioRawFrame,
)
from pipecat.metrics.metrics import LLMUsageMetricsData
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
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

# Prices last verified 2026-04-15 against official pricing pages and
# cross-checked with the 2026-04-14 session's provider dashboards.
# Sources: claude.com/pricing, deepgram.com/pricing, cartesia.ai/pricing.
PRICE_ANTHROPIC_INPUT_PER_MTOK = 3.00
PRICE_ANTHROPIC_OUTPUT_PER_MTOK = 15.00
PRICE_ANTHROPIC_CACHE_WRITE_PER_MTOK = 3.75
PRICE_ANTHROPIC_CACHE_READ_PER_MTOK = 0.30
PRICE_DEEPGRAM_NOVA3_PER_MIN = 0.0077
# Cartesia: 15 credits/sec of audio; $5 / 100_000 credits on Pro plan.
# Actual audio seconds now derived from TTSAudioRawFrame bytes (ground truth),
# not estimated from char count.
CARTESIA_CREDITS_PER_SEC = 15
PRICE_CARTESIA_PER_CREDIT = 5.00 / 100_000

# NOTE: cost-log.jsonl starts from the first session after this refactor
# (2026-04-15+). Sessions logged before this (e.g. 2026-04-14) only exist as
# rows in cost-log.md — there's no raw usage data to backfill for them.


class UsageAccumulator(BaseObserver):
    def __init__(self):
        super().__init__()
        self.uncached_input_tokens = 0
        self.cache_read_tokens = 0
        self.cache_write_tokens = 0
        self.output_tokens = 0
        self.tts_audio_sec = 0.0
        # Cross-check against session_duration_sec; divergence would signal a
        # transport behavior change (e.g. VAD-gated audio_in).
        self.stt_audio_sec = 0.0
        self.tool_calls: list[dict] = []
        # Set by the tool handler immediately after it runs; the next
        # TTSAudioRawFrame closes the measurement.
        self._pending_tool: dict | None = None

    def mark_tool_call(self, page: str):
        entry = {"page": page, "start_monotonic": time.monotonic(), "timestamp": datetime.now().isoformat()}
        self._pending_tool = entry

    async def on_push_frame(self, data: FramePushed):
        frame = data.frame
        # TTSAudioRawFrame check must precede InputAudioRawFrame because
        # both inherit from AudioRawFrame — TTS is output, STT is input.
        if isinstance(frame, TTSAudioRawFrame):
            denom = frame.sample_rate * max(frame.num_channels, 1) * 2
            if denom:
                self.tts_audio_sec += len(frame.audio) / denom
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
            denom = frame.sample_rate * max(frame.num_channels, 1) * 2
            if denom:
                self.stt_audio_sec += len(frame.audio) / denom
            return
        if isinstance(frame, MetricsFrame):
            for m in frame.data:
                if isinstance(m, LLMUsageMetricsData):
                    u = m.value
                    self.cache_read_tokens += u.cache_read_input_tokens or 0
                    self.cache_write_tokens += u.cache_creation_input_tokens or 0
                    # Anthropic's prompt_tokens already excludes cache reads/writes.
                    self.uncached_input_tokens += u.prompt_tokens
                    self.output_tokens += u.completion_tokens

    def summary(self, session_duration_sec: float) -> dict:
        llm_input_cost = self.uncached_input_tokens / 1_000_000 * PRICE_ANTHROPIC_INPUT_PER_MTOK
        llm_cache_read_cost = self.cache_read_tokens / 1_000_000 * PRICE_ANTHROPIC_CACHE_READ_PER_MTOK
        llm_cache_write_cost = self.cache_write_tokens / 1_000_000 * PRICE_ANTHROPIC_CACHE_WRITE_PER_MTOK
        llm_output_cost = self.output_tokens / 1_000_000 * PRICE_ANTHROPIC_OUTPUT_PER_MTOK
        llm_cost = llm_input_cost + llm_cache_read_cost + llm_cache_write_cost + llm_output_cost

        # Deepgram bills per minute of audio streamed over the open connection.
        # SmallWebRTCTransport streams continuously, so session_duration_sec is
        # ground-truth for billing. stt_audio_sec is an observed cross-check —
        # should match within rounding; divergence signals a transport change.
        stt_minutes = session_duration_sec / 60
        stt_cost = stt_minutes * PRICE_DEEPGRAM_NOVA3_PER_MIN

        tts_credits = self.tts_audio_sec * CARTESIA_CREDITS_PER_SEC
        tts_cost = tts_credits * PRICE_CARTESIA_PER_CREDIT

        total = llm_cost + stt_cost + tts_cost
        return {
            "session_duration_sec": round(session_duration_sec, 1),
            "llm": {
                "uncached_input_tokens": self.uncached_input_tokens,
                "cache_read_tokens": self.cache_read_tokens,
                "cache_write_tokens": self.cache_write_tokens,
                "output_tokens": self.output_tokens,
                "cost_usd": round(llm_cost, 4),
            },
            "stt": {
                "minutes": round(stt_minutes, 2),
                "audio_sec_observed": round(self.stt_audio_sec, 1),
                "cost_usd": round(stt_cost, 4),
            },
            "tts": {
                "audio_sec": round(self.tts_audio_sec, 1),
                "credits": round(tts_credits, 0),
                "cost_usd": round(tts_cost, 4),
            },
            "total_cost_usd": round(total, 4),
        }

VOICE_TUTOR_DIR = Path.home() / ".voice-tutor"
TRANSCRIPTS_DIR = VOICE_TUTOR_DIR / "transcripts"
PROFILE_PATH = VOICE_TUTOR_DIR / "profile.md"
# Accumulating memory: one dated section per session, append-only.
# Future: when this file exceeds ~2K tokens, compact older entries by summarizing
# everything before a cutoff date into a single "before April X" block. Not today's
# problem — revisit when the memory block starts dominating the system prompt.
MEMORY_PATH = VOICE_TUTOR_DIR / "memory.md"
WIKI_DIR = Path.home() / "second-brain" / "resources" / "wiki"
COST_LOG_PATH = Path.home() / "second-brain" / "products" / "voice-tutor" / "cost-log.md"
COST_LOG_JSONL_PATH = COST_LOG_PATH.with_suffix(".jsonl")
SESSION_ANALYSIS_DIR = Path.home() / "second-brain" / "products" / "voice-tutor"
MIN_ANALYSIS_DURATION_SEC = 300
MIN_SUMMARY_DURATION_SEC = 120

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

## Wiki usage vs general knowledge
Estimate how much of the conversation drew from wiki content vs the LLM's general knowledge \
vs Matt's own ideas. Use a simple table with columns: Source, Approx turns, Notes. \
Sources are: "On-demand wiki pages", "Pre-loaded wiki", "General LLM knowledge", \
"Matt's own knowledge/ideas". Note the key finding — was the wiki central or peripheral?

## Interaction quality notes
Bullet points on: pacing issues (did Matt ask to slow down?), STT errors (misheard words), \
interruptions, response length compliance, and anything else notable about the interaction dynamics.

Here is the session data:

### Usage summary
{usage_json}

### Tool calls
{tool_calls_json}

### Transcript
{transcript_json}
"""

BASE_INSTRUCTION = (
    "You are a friendly, curious conversational partner and tutor. "
    "Be concise. Say one thought at a time, then let Matt respond. "
    "One to two sentences per turn. Never monologue. "
    "Be warm but not sycophantic. Never repeat yourself. "
    "You know Matt from prior conversations. "
    "Reference past topics naturally when relevant, but don't force it. "
    "You have access to Matt's personal knowledge wiki — use it to teach, "
    "connect ideas, and reference things he's been reading and learning about."
)


def load_profile() -> str:
    if PROFILE_PATH.exists():
        return PROFILE_PATH.read_text()
    return ""


# The wiki is fully on-demand: only INDEX.md is preloaded into the prompt.
# Every other page is fetched via the read_wiki_page tool when a topic
# becomes central to the conversation.
WIKI_USAGE_INSTRUCTIONS = (
    "### How to use the wiki\n\n"
    "- When you need a wiki page, say one short sentence first (e.g. "
    "\"let me pull that up\"), then call `read_wiki_page(path)`. This prevents "
    "silence during the lookup.\n"
    "- Don't open pages speculatively. Open a page only when a specific topic "
    "is central to Matt's current question. One page per question is typical "
    "— don't chain-open multiple pages unless Matt explicitly asks about "
    "multiple topics.\n"
    "- Pass the path relative to the wiki root exactly as shown in the index, "
    "e.g. 'concepts/llm-knowledge-bases.md' or 'landscape/yc-ai-thesis.md'."
)


def load_wiki_index() -> str:
    index_path = WIKI_DIR / "INDEX.md"
    if not index_path.exists():
        return ""
    return index_path.read_text()


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


SUMMARY_PROMPT = """\
Summarize this voice tutoring conversation in 3-5 short bullet points. Cover what \
was discussed, any decisions Matt made, and any open questions or next steps. Be \
terse — this is loaded as context into a future voice session so the tutor can \
pick up continuity. Output only the bullets (one per line, starting with "- "). \
No preamble, no trailing prose.

### Transcript
{transcript_json}
"""


def generate_session_summary(stem: str, transcript: dict):
    prompt = SUMMARY_PROMPT.format(transcript_json=json.dumps(transcript, indent=2))
    try:
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
    except Exception as e:
        print(f"[session-summary] failed: {e}", file=sys.stderr, flush=True)
        return
    out_path = TRANSCRIPTS_DIR / f"{stem}.summary.md"
    out_path.write_text(text + "\n")
    print(f"[session-summary] wrote {out_path}", file=sys.stderr, flush=True)


def generate_session_analysis(stem: str, transcript: dict, summary: dict, tool_calls: list[dict]):
    prompt = ANALYSIS_PROMPT.format(
        usage_json=json.dumps(summary, indent=2),
        tool_calls_json=json.dumps(tool_calls, indent=2) if tool_calls else "[]",
        transcript_json=json.dumps(transcript, indent=2),
    )
    try:
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        analysis = resp.content[0].text
    except Exception as e:
        print(f"[session-analysis] failed: {e}", file=sys.stderr, flush=True)
        return
    header = f"# Session Analysis — {stem}\n\n"
    out_path = SESSION_ANALYSIS_DIR / f"session-analysis-{stem}.md"
    out_path.write_text(header + analysis)
    print(f"[session-analysis] wrote {out_path}", file=sys.stderr, flush=True)


def build_system_instruction() -> str:
    parts = [BASE_INSTRUCTION]

    profile = load_profile()
    if profile:
        parts.append(f"\n## About the person you're talking to\n\n{profile}")

    wiki_index = load_wiki_index()
    if wiki_index:
        parts.append(
            f"\n## Matt's knowledge wiki\n\n{wiki_index}\n\n{WIKI_USAGE_INSTRUCTIONS}"
        )

    memory = load_memory()
    if memory:
        parts.append(f"\n# What we've discussed\n\n{memory}")

    most_recent = load_most_recent_transcript_block()
    if most_recent:
        parts.append(f"\n# Most recent session\n\n{most_recent}")

    return "\n".join(parts)


async def bot(runner_args):
    transport = SmallWebRTCTransport(
        webrtc_connection=runner_args.webrtc_connection,
        params=TransportParams(audio_out_enabled=True, audio_in_enabled=True),
    )

    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        settings=DeepgramSTTService.Settings(model="nova-3", language="en"),
    )

    system_instruction = build_system_instruction()

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
            voice="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
        ),
    )

    read_wiki_schema = FunctionSchema(
        name="read_wiki_page",
        description=(
            "Open a page from Matt's knowledge wiki. Pass the path relative to "
            "the wiki root exactly as shown in the index in the system prompt, "
            "e.g. 'concepts/llm-knowledge-bases.md' or 'landscape/yc-ai-thesis.md'."
        ),
        properties={
            "path": {
                "type": "string",
                "description": "Path relative to wiki root, e.g. 'concepts/llm-knowledge-bases.md'.",
            },
        },
        required=["path"],
    )
    context = LLMContext(tools=ToolsSchema(standard_tools=[read_wiki_schema]))
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    usage = UsageAccumulator()

    async def handle_read_wiki_page(params):
        path = params.arguments.get("path", "")
        requested = (WIKI_DIR / path).resolve()
        wiki_root = WIKI_DIR.resolve()
        try:
            requested.relative_to(wiki_root)
        except ValueError:
            await params.result_callback({"error": "path must be inside the wiki"})
            return
        if not requested.exists():
            await params.result_callback({"error": f"page not found: {path}"})
            return
        usage.mark_tool_call(path)
        print(f"[wiki-tool] opening {path}", file=sys.stderr, flush=True)
        await params.result_callback({"content": requested.read_text()})

    llm.register_function("read_wiki_page", handle_read_wiki_page)

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

    def save_transcript():
        if not turns:
            return
        TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
        stem = session_start.strftime("%Y-%m-%d-%H%M%S")
        session_end = datetime.now()
        transcript = {
            "session_start": session_start.isoformat(),
            "session_end": session_end.isoformat(),
            "turn_count": len(turns),
            "turns": turns,
        }
        (TRANSCRIPTS_DIR / f"{stem}.json").write_text(json.dumps(transcript, indent=2))

        summary = usage.summary((session_end - session_start).total_seconds())
        (TRANSCRIPTS_DIR / f"{stem}.usage.json").write_text(json.dumps(summary, indent=2))
        mins = summary["session_duration_sec"] / 60
        line = (
            f"Session: {mins:.1f}min · {len(turns)} turns · "
            f"${summary['total_cost_usd']:.3f} "
            f"(llm ${summary['llm']['cost_usd']:.3f} · "
            f"stt ${summary['stt']['cost_usd']:.3f} · "
            f"tts ${summary['tts']['cost_usd']:.3f})"
        )
        print(line, flush=True)

        COST_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not COST_LOG_PATH.exists():
            COST_LOG_PATH.write_text(
                "# Voice Tutor — Cost Log\n\n"
                "One row per session. Costs are computed from ground-truth usage\n"
                "(TTS audio bytes, LLM token counts, Deepgram streamed minutes).\n"
                "Rates last verified 2026-04-15 against provider pricing pages.\n"
                "Per-session raw usage is logged to `cost-log.jsonl` for auditing\n"
                "(starting 2026-04-15 — earlier sessions have no raw-usage sidecar).\n\n"
                "| Session start | Duration | Turns | Total | LLM | STT | TTS |\n"
                "|---|---|---|---|---|---|---|\n"
            )
        row = (
            f"| {session_start.strftime('%Y-%m-%d %H:%M')} "
            f"| {mins:.1f} min "
            f"| {len(turns)} "
            f"| ${summary['total_cost_usd']:.3f} "
            f"| ${summary['llm']['cost_usd']:.3f} "
            f"| ${summary['stt']['cost_usd']:.3f} "
            f"| ${summary['tts']['cost_usd']:.3f} |\n"
        )
        with COST_LOG_PATH.open("a") as f:
            f.write(row)

        jsonl_entry = {
            "session_id": session_start.strftime("%Y-%m-%dT%H%M%S"),
            "session_start": session_start.isoformat(),
            "session_end": session_end.isoformat(),
            "session_duration_sec": summary["session_duration_sec"],
            "turns": len(turns),
            "tts_audio_sec": summary["tts"]["audio_sec"],
            "tts_credits": summary["tts"]["credits"],
            "stt_audio_sec_observed": summary["stt"]["audio_sec_observed"],
            "stt_minutes_billed": summary["stt"]["minutes"],
            "llm_uncached_input_tokens": summary["llm"]["uncached_input_tokens"],
            "llm_cache_read_tokens": summary["llm"]["cache_read_tokens"],
            "llm_cache_write_tokens": summary["llm"]["cache_write_tokens"],
            "llm_output_tokens": summary["llm"]["output_tokens"],
            "cost_llm_usd": summary["llm"]["cost_usd"],
            "cost_stt_usd": summary["stt"]["cost_usd"],
            "cost_tts_usd": summary["tts"]["cost_usd"],
            "cost_total_usd": summary["total_cost_usd"],
            "tool_calls": usage.tool_calls,
        }
        with COST_LOG_JSONL_PATH.open("a") as f:
            f.write(json.dumps(jsonl_entry) + "\n")

        if summary["session_duration_sec"] >= MIN_SUMMARY_DURATION_SEC:
            generate_session_summary(stem, transcript)
            summary_path = TRANSCRIPTS_DIR / f"{stem}.summary.md"
            if summary_path.exists():
                append_to_memory(transcript, summary_path.read_text())

        if summary["session_duration_sec"] >= MIN_ANALYSIS_DURATION_SEC:
            generate_session_analysis(stem, transcript, summary, usage.tool_calls)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        context.add_message({"role": "user", "content": "Say hello and introduce yourself briefly."})
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        save_transcript()
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)


if __name__ == "__main__":
    from pipecat.runner.run import main
    main()
