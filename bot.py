import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import anthropic

import documents
import wiki
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
        # Exact character count submitted to Cartesia (the billing unit).
        self.tts_chars = 0
        # Observed audio length from TTSAudioRawFrame bytes — kept as a
        # cross-check / observability metric, not load-bearing for cost.
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
                elif isinstance(m, TTSUsageMetricsData):
                    self.tts_chars += m.value

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

        tts_cost = self.tts_chars * PRICE_CARTESIA_PER_CHAR

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
                "chars": self.tts_chars,
                "audio_sec_observed": round(self.tts_audio_sec, 1),
                "cost_usd": round(tts_cost, 4),
            },
            "total_cost_usd": round(total, 4),
        }

from session_state import (
    MEMORY_PATH,
    PROFILE_PATH,
    TRANSCRIPTS_DIR,
    VOICE_TUTOR_DIR,
    _format_full_transcript_block,
    _format_memory_date,
    _format_session_time,
    append_to_memory,
    load_memory,
    load_most_recent_transcript_block,
    load_profile,
)
ARTIFACTS_DIR = VOICE_TUTOR_DIR / "artifacts"
# Accumulating memory: one dated section per session, append-only.
# Future: when this file exceeds ~2K tokens, compact older entries by summarizing
# everything before a cutoff date into a single "before April X" block. Not today's
# problem — revisit when the memory block starts dominating the system prompt.
COST_LOG_PATH = Path.home() / "second-brain" / "products" / "voice-tutor" / "validation" / "cost-log.md"
COST_LOG_JSONL_PATH = COST_LOG_PATH.with_suffix(".jsonl")
SESSION_ANALYSIS_DIR = Path.home() / "second-brain" / "products" / "voice-tutor" / "session-analyses"
# Lower bound (in seconds) before we spend Haiku tokens on a session summary or
# analysis. 120 is the production default — shorter sessions tend to produce
# thin summaries that pollute memory.md. Set VOICE_TUTOR_MIN_TELEMETRY_SEC in
# .env to override both thresholds together (useful for demos where you want
# the full diagnostics panel to populate from a ~1-min session).
_min_telemetry_override = os.getenv("VOICE_TUTOR_MIN_TELEMETRY_SEC")
MIN_ANALYSIS_DURATION_SEC = int(_min_telemetry_override) if _min_telemetry_override else 120
MIN_SUMMARY_DURATION_SEC = int(_min_telemetry_override) if _min_telemetry_override else 120

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
LLM's general knowledge vs Matt's own ideas. Use a simple table with columns: Source, \
Approx turns, Notes. Sources are:
- "On-demand wiki pages" — content fetched via read_wiki_page tool calls during the session
- "Pre-loaded wiki INDEX" — the wiki table of contents in the system prompt (titles and one-line descriptions only, not full page content)
- "Prior-session memory" — the "What we've discussed" block summarizing past sessions
- "Most-recent transcript" — the verbatim block from the previous session
- "General LLM knowledge" — things the model knows independent of any loaded context
- "Matt's own knowledge/ideas" — claims, framing, or context Matt introduced himself

Be specific in Notes about which facts came from which source — don't lump everything \
pre-loaded into one bucket. Note the key finding: which source(s) carried the conversation, \
and was the wiki itself central or peripheral?

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

STUDY_BASE_INSTRUCTION = (
    "You are a study companion helping the user understand a specific document "
    "they have loaded. Help them engage actively — ask what they want to focus "
    "on, explain concepts when asked, surface connections, push back when their "
    "understanding is shaky, and let them lead the direction.\n\n"
    "Reference the document directly. Quote short passages when useful. Don't "
    "summarize the whole thing unprompted — wait for the user to point at what "
    "they want to dig into.\n\n"
    "Keep responses tight. One thought at a time. This is voice — long monologues "
    "don't work."
)

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
    "Be concise. Say one thought at a time, then let Matt respond. "
    "One to two sentences per turn. Never monologue. "
    "Be warm but not sycophantic. Never repeat yourself. "
    "You know Matt from prior conversations. "
    "Reference past topics naturally when relevant, but don't force it."
)

# Appended to BASE_INSTRUCTION only when the wiki module is active. Lives in
# bot.py rather than wiki.py because it's about persona framing, not the tool
# itself — the wiki module owns the actual section block and usage rules.
WIKI_TAGLINE = (
    " You have access to Matt's personal knowledge wiki — use it to teach, "
    "connect ideas, and reference things he's been reading and learning about."
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
    "the document. Stay focused on what's in front of you."
)


WIKI_ENABLED = os.getenv("WIKI_ENABLED", "true").lower() in ("1", "true", "yes")


SUMMARY_PROMPT = """\
Summarize this voice tutoring conversation in 3-5 short bullet points. Cover what \
was discussed, any decisions Matt made, and any open questions or next steps. Be \
terse — this is loaded as context into a future voice session so the tutor can \
pick up continuity. Output only the bullets (one per line, starting with "- "). \
No preamble, no trailing prose.

### Transcript
{transcript_json}
"""


def generate_session_summary(stem: str, transcript: dict) -> dict | None:
    prompt = SUMMARY_PROMPT.format(transcript_json=json.dumps(transcript, indent=2))
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
    out_path = TRANSCRIPTS_DIR / f"{stem}.summary.md"
    out_path.write_text(text + "\n")
    print(f"[session-summary] wrote {out_path}", file=sys.stderr, flush=True)
    return {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}


def generate_session_analysis(stem: str, transcript: dict, summary: dict, tool_calls: list[dict]) -> dict | None:
    prompt = ANALYSIS_PROMPT.format(
        usage_json=json.dumps(summary, indent=2),
        tool_calls_json=json.dumps(tool_calls, indent=2) if tool_calls else "[]",
        transcript_json=json.dumps(transcript, indent=2),
    )
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
    out_path = SESSION_ANALYSIS_DIR / f"session-analysis-{stem}.md"
    out_path.write_text(header + analysis)
    print(f"[session-analysis] wrote {out_path}", file=sys.stderr, flush=True)
    return {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}


async def generate_artifact(session_id: str, study_meta: dict, transcript: dict, duration_sec: float):
    """Fire-and-forget Haiku call writing ~/.voice-tutor/artifacts/<session_id>.md.

    Writes a separate row to cost-log.jsonl with kind="artifact" so the cost is
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

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ARTIFACTS_DIR / f"{session_id}.md"
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
        "document_id": study_meta["document_id"],
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "cost_usd": round(cost, 4),
    }
    COST_LOG_JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with COST_LOG_JSONL_PATH.open("a") as f:
        f.write(json.dumps(row) + "\n")


def build_system_instruction(study: dict | None = None) -> str:
    """Assemble the system prompt.

    Regular mode: base + profile + wiki INDEX + memory + most-recent transcript.
    Study mode (study={doc_title, doc_text}): base + profile + memory + the doc.
    Study mode skips the most-recent transcript, the wiki INDEX, and the wiki
    tagline — the doc replaces those as the session's focus.
    """
    profile = load_profile()

    if study is not None:
        parts = [STUDY_BASE_INSTRUCTION]
        if profile:
            parts.append(f"\n## About the person you're talking to\n\n{profile}")
        memory = load_memory()
        if memory:
            parts.append(
                "\n# Background — Matt's prior topics (reference only if directly relevant to the document)\n\n"
                + memory
            )
        parts.append(f"\n## Document: {study['doc_title']}\n\n{study['doc_text']}")
        parts.append(BREVITY_REMINDER)
        parts.append(STUDY_REMINDER)
        return "\n".join(parts)

    base = BASE_INSTRUCTION + (WIKI_TAGLINE if WIKI_ENABLED else "")
    parts = [base]

    if profile:
        parts.append(f"\n## About the person you're talking to\n\n{profile}")

    if WIKI_ENABLED:
        wiki_block = wiki.system_prompt_block()
        if wiki_block:
            parts.append(wiki_block)

    memory = load_memory()
    if memory:
        parts.append(f"\n# What we've discussed\n\n{memory}")

    most_recent = load_most_recent_transcript_block()
    if most_recent:
        parts.append(f"\n# Most recent session\n\n{most_recent}")

    parts.append(BREVITY_REMINDER)
    return "\n".join(parts)


async def bot(runner_args):
    transport = SmallWebRTCTransport(
        webrtc_connection=runner_args.webrtc_connection,
        params=TransportParams(audio_out_enabled=True, audio_in_enabled=True),
    )

    body = getattr(runner_args, "body", None) or {}
    document_id = body.get("document_id")
    session_id_override = body.get("session_id")

    study_meta: dict | None = None
    if document_id:
        loaded = documents.load_document(document_id)
        if loaded is None:
            print(f"[bot] document_id={document_id} not found; falling back to regular mode", file=sys.stderr, flush=True)
        else:
            doc_title, doc_text = loaded
            study_meta = {
                "document_id": document_id,
                "doc_title": doc_title,
                "doc_text": doc_text,
                "session_id": session_id_override or document_id,
            }

    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        settings=DeepgramSTTService.Settings(model="nova-3", language="en"),
    )

    system_instruction = build_system_instruction(
        study={"doc_title": study_meta["doc_title"], "doc_text": study_meta["doc_text"]}
        if study_meta else None
    )

    if study_meta:
        TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
        (TRANSCRIPTS_DIR / f"{study_meta['session_id']}.prompt.txt").write_text(system_instruction)

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

    def save_transcript():
        if not turns:
            return
        TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
        stem = study_meta["session_id"] if study_meta else session_start.strftime("%Y-%m-%d-%H%M%S")
        session_end = datetime.now()
        transcript = {
            "session_start": session_start.isoformat(),
            "session_end": session_end.isoformat(),
            "turn_count": len(turns),
            "turns": turns,
        }
        (TRANSCRIPTS_DIR / f"{stem}.json").write_text(json.dumps(transcript, indent=2))

        summary = usage.summary((session_end - session_start).total_seconds())

        # Run post-session Haiku calls before finalizing the cost log so their
        # tokens roll into the row. They were previously off-the-books — small
        # (~$0.025/session) but unaccounted for vs the Anthropic dashboard.
        post_input = 0
        post_output = 0
        if summary["session_duration_sec"] >= MIN_SUMMARY_DURATION_SEC:
            u = generate_session_summary(stem, transcript)
            if u:
                post_input += u["input_tokens"]
                post_output += u["output_tokens"]
            summary_path = TRANSCRIPTS_DIR / f"{stem}.summary.md"
            if summary_path.exists():
                append_to_memory(transcript, summary_path.read_text())

        if summary["session_duration_sec"] >= MIN_ANALYSIS_DURATION_SEC:
            u = generate_session_analysis(stem, transcript, summary, usage.tool_calls)
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

        (TRANSCRIPTS_DIR / f"{stem}.usage.json").write_text(json.dumps(summary, indent=2))
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
                "(matching the Anthropic dashboard); see `cost-log.jsonl` for the\n"
                "live-vs-post-session breakdown.\n"
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
            f"| ${llm_total:.3f} "
            f"| ${summary['stt']['cost_usd']:.3f} "
            f"| ${summary['tts']['cost_usd']:.3f} |\n"
        )
        with COST_LOG_PATH.open("a") as f:
            f.write(row)

        jsonl_entry = {
            "kind": "session",
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
            "tool_calls": usage.tool_calls,
        }
        if study_meta:
            jsonl_entry["session_id"] = study_meta["session_id"]
            jsonl_entry["mode"] = "study"
            jsonl_entry["document_id"] = study_meta["document_id"]
        with COST_LOG_JSONL_PATH.open("a") as f:
            f.write(json.dumps(jsonl_entry) + "\n")

        if study_meta:
            asyncio.create_task(generate_artifact(
                session_id=study_meta["session_id"],
                study_meta=study_meta,
                transcript=transcript,
                duration_sec=summary["session_duration_sec"],
            ))

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
