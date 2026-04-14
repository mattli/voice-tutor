import json
import os
from datetime import datetime
from pathlib import Path

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
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

VOICE_TUTOR_DIR = Path.home() / ".voice-tutor"
TRANSCRIPTS_DIR = VOICE_TUTOR_DIR / "transcripts"
PROFILE_PATH = VOICE_TUTOR_DIR / "profile.md"
WIKI_DIR = Path.home() / "second-brain" / "resources" / "wiki"
RECENT_TRANSCRIPT_COUNT = 3

BASE_INSTRUCTION = (
    "You are a friendly, curious conversational partner and tutor. "
    "Keep responses concise and natural for voice — "
    "one to three sentences at most unless asked for detail. "
    "Be warm but not sycophantic. Never repeat yourself within a response. "
    "You know the person you're talking to from prior conversations. "
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

    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
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
        filename = session_start.strftime("%Y-%m-%d-%H%M%S") + ".json"
        transcript = {
            "session_start": session_start.isoformat(),
            "session_end": datetime.now().isoformat(),
            "turn_count": len(turns),
            "turns": turns,
        }
        (TRANSCRIPTS_DIR / filename).write_text(json.dumps(transcript, indent=2))

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
