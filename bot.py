import json
import os
from datetime import datetime
from pathlib import Path

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

    async def on_push_frame(self, data: FramePushed):
        frame = data.frame
        # TTSAudioRawFrame check must precede InputAudioRawFrame because
        # both inherit from AudioRawFrame — TTS is output, STT is input.
        if isinstance(frame, TTSAudioRawFrame):
            denom = frame.sample_rate * max(frame.num_channels, 1) * 2
            if denom:
                self.tts_audio_sec += len(frame.audio) / denom
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
WIKI_DIR = Path.home() / "second-brain" / "resources" / "wiki"
COST_LOG_PATH = Path.home() / "second-brain" / "products" / "voice-tutor" / "cost-log.md"
COST_LOG_JSONL_PATH = COST_LOG_PATH.with_suffix(".jsonl")
RECENT_TRANSCRIPT_COUNT = 3

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


def load_wiki() -> str:
    if not WIKI_DIR.exists():
        return ""
    pages = []
    for f in sorted(WIKI_DIR.rglob("*.md")):
        rel_path = f.relative_to(WIKI_DIR)
        pages.append(f"### {rel_path}\n\n{f.read_text()}")
    return "\n\n---\n\n".join(pages)


def load_recent_transcripts() -> list[dict]:
    if not TRANSCRIPTS_DIR.exists():
        return []
    files = sorted(TRANSCRIPTS_DIR.glob("*.json"), reverse=True)[:RECENT_TRANSCRIPT_COUNT]
    transcripts = []
    for f in files:
        transcripts.append(json.loads(f.read_text()))
    return list(reversed(transcripts))  # chronological order


def format_transcript_summary(transcript: dict) -> str:
    date = transcript["session_start"][:10]
    lines = []
    for turn in transcript["turns"]:
        role = "You" if turn["role"] == "assistant" else "Matt"
        lines.append(f"  {role}: {turn['content']}")
    return f"Conversation on {date}:\n" + "\n".join(lines)


def build_system_instruction() -> str:
    parts = [BASE_INSTRUCTION]

    profile = load_profile()
    if profile:
        parts.append(f"\n## About the person you're talking to\n\n{profile}")

    wiki = load_wiki()
    if wiki:
        parts.append(f"\n## Matt's knowledge wiki\n\n{wiki}")

    transcripts = load_recent_transcripts()
    if transcripts:
        parts.append("\n## Recent conversations\n")
        for t in transcripts:
            parts.append(format_transcript_summary(t))

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

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline([
        transport.input(),
        stt,
        user_aggregator,
        llm,
        tts,
        transport.output(),
        assistant_aggregator,
    ])

    usage = UsageAccumulator()
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
        }
        with COST_LOG_JSONL_PATH.open("a") as f:
            f.write(json.dumps(jsonl_entry) + "\n")

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
