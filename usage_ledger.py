"""Pure, Pipecat-free usage accounting for a single tutoring session.

Extracted from bot.py's UsageAccumulator so the dedup logic and the cost
`summary()` math are testable without importing pipecat / the ML stack.
bot.py's UsageAccumulator(BaseObserver) is a thin adapter that pulls
``(frame.id, plain usage values)`` off pipecat frames and calls the methods
here.

Per-hop multi-count fix
-----------------------
Pipecat invokes ``on_push_frame`` once per processor-to-processor hop, and one
observer registered on the PipelineTask sees every hop pipeline-wide. The legacy
accumulator did ``self.<counter> += ...`` on every observation, so each frame's
usage was counted once PER HOP it traversed. Runtime evidence (2026-07-22, a
12-turn session) measured the multiple as id-stable integers: LLM tokens 5.00x,
STT audio 8.00x, TTS audio ~2.63x — the multiple equals the number of downstream
hops from the emitting processor to the sink.

This ledger dedups by ``frame.id`` (pipecat's globally-unique per-frame id) so a
frame's usage is counted exactly once regardless of how many hops it makes.

Dedup is decided ONCE PER FRAME via ``should_count(frame_id)`` (a single
MetricsFrame can carry both LLM and TTS usage, so the count decision must span
all of that frame's metrics, not be re-taken per metric).

Seen-id lifecycle
-----------------
``_seen`` is a per-instance, FIFO-bounded ``OrderedDict``. Two layers of bound:
1. The adapter (UsageAccumulator) is instantiated once per session, so this set
   is instance-scoped and is freed at session end — "clear per session" by
   construction. There is no module- or class-level usage state.
2. It is additionally capped at ``seen_id_cap`` so even a marathon single session
   stays constant-memory. The cap is safe against under-dedup because a frame's
   hops cluster tightly in time: the measured max hop-span for a real session was
   295 observations, far below the default cap of 4096, so a frame's later hops
   are never evicted before they are counted.
"""

from collections import OrderedDict

from cost_audit import (
    PRICE_ANTHROPIC_CACHE_READ_PER_MTOK,
    PRICE_ANTHROPIC_CACHE_WRITE_PER_MTOK,
    PRICE_ANTHROPIC_INPUT_PER_MTOK,
    PRICE_ANTHROPIC_OUTPUT_PER_MTOK,
    PRICE_CARTESIA_PER_CHAR,
    PRICE_DEEPGRAM_NOVA3_PER_MIN,
)

# Measured max hop-span was 295 observations (2026-07-22); 4096 is ~14x headroom
# so a frame's later hops are never evicted before they are counted.
DEFAULT_SEEN_ID_CAP = 4096


class UsageLedger:
    def __init__(self, dedup: bool = True, seen_id_cap: int = DEFAULT_SEEN_ID_CAP):
        self.dedup = dedup
        self._seen_id_cap = seen_id_cap
        # frame_id -> True; insertion-ordered so we can FIFO-evict the oldest.
        self._seen: "OrderedDict[int, bool]" = OrderedDict()

        self.uncached_input_tokens = 0
        self.cache_read_tokens = 0
        self.cache_write_tokens = 0
        self.output_tokens = 0
        # Exact character count submitted to Cartesia (the billing unit).
        self.tts_chars = 0
        # Observed audio length from TTSAudioRawFrame bytes — cross-check /
        # observability metric, not load-bearing for cost.
        self.tts_audio_sec = 0.0
        # Cross-check against session_duration_sec.
        self.stt_audio_sec = 0.0

    def should_count(self, frame_id) -> bool:
        """Return True if this frame's usage should be counted now.

        With dedup off, always True (legacy per-hop multi-count behavior — the
        ``VOICE_TUTOR_USAGE_DEDUP=0`` escape hatch). With dedup on, True only the
        first time ``frame_id`` is seen; the seen-set is FIFO-bounded at
        ``seen_id_cap``.
        """
        if not self.dedup:
            return True
        if frame_id in self._seen:
            return False
        self._seen[frame_id] = True
        if len(self._seen) > self._seen_id_cap:
            self._seen.popitem(last=False)  # evict oldest (FIFO)
        return True

    def add_llm_usage(self, *, prompt_tokens, cache_read, cache_write, completion) -> None:
        """Apply one LLMUsageMetricsData payload. Caller must have gated this on
        ``should_count(frame.id)`` — this method does NOT re-check dedup."""
        self.cache_read_tokens += cache_read or 0
        self.cache_write_tokens += cache_write or 0
        # Anthropic's prompt_tokens already excludes cache reads/writes.
        self.uncached_input_tokens += prompt_tokens
        self.output_tokens += completion

    def add_tts_chars(self, chars: int) -> None:
        self.tts_chars += chars

    def add_stt_audio(self, seconds: float) -> None:
        self.stt_audio_sec += seconds

    def add_tts_audio(self, seconds: float) -> None:
        self.tts_audio_sec += seconds

    def summary(self, session_duration_sec: float) -> dict:
        llm_input_cost = self.uncached_input_tokens / 1_000_000 * PRICE_ANTHROPIC_INPUT_PER_MTOK
        llm_cache_read_cost = self.cache_read_tokens / 1_000_000 * PRICE_ANTHROPIC_CACHE_READ_PER_MTOK
        llm_cache_write_cost = self.cache_write_tokens / 1_000_000 * PRICE_ANTHROPIC_CACHE_WRITE_PER_MTOK
        llm_output_cost = self.output_tokens / 1_000_000 * PRICE_ANTHROPIC_OUTPUT_PER_MTOK
        llm_cost = llm_input_cost + llm_cache_read_cost + llm_cache_write_cost + llm_output_cost

        # Deepgram bills per minute of audio streamed over the open connection.
        # SmallWebRTCTransport streams continuously, so session_duration_sec is
        # the billing basis. stt_audio_sec is an observed cross-check only.
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
