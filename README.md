# Voice Tutor

Self-hosted voice conversation service using Pipecat, Claude, and persistent memory. Runs on Mac Mini, accessed from phone browser via Tailscale.

## Local development setup

Going from a fresh machine to a running tutor.

### 1. Clone

```bash
git clone https://github.com/mattli/voice-tutor.git
cd voice-tutor
```

### 2. Install dependencies

The project uses [`uv`](https://github.com/astral-sh/uv). If it's not installed:

```bash
brew install uv
```

Then:

```bash
uv sync
```

### 3. API keys

Create a `.env` file in the repo root:

```
ANTHROPIC_API_KEY=your-key
DEEPGRAM_API_KEY=your-key
CARTESIA_API_KEY=your-key
# Optional. Default true. Set to false to run the bot without the
# personal-wiki integration — strips the wiki INDEX from the system prompt,
# unregisters the read_wiki_page tool, and drops the wiki tagline from the
# persona instruction. Useful for A/B-testing baseline behavior.
# WIKI_ENABLED=false
```

`.env` is gitignored — never commit it.

### 4. Personal data directory

The app reads and writes `~/.voice-tutor/`. Create it:

```bash
mkdir -p ~/.voice-tutor
```

Optional but recommended: seed `~/.voice-tutor/profile.md` with a short
identity blurb. The model loads it verbatim into the system prompt so the
tutor knows who it's talking to. A few sentences is enough; see the **Data**
section below for the full layout.

If you want to carry over an existing setup from another machine, copy the
whole directory over:

```bash
scp -r user@other-host:~/.voice-tutor/ ~/.voice-tutor/
```

### 5. Wiki integration (optional)

If `WIKI_ENABLED=true` (the default), `wiki.py` reads from
`~/second-brain/resources/wiki/`. Either point that directory at your own
wiki, or set `WIKI_ENABLED=false` in `.env` to skip the integration entirely.

### 6. Run

```bash
./start.sh
```

The server listens on `:7860`. Open `http://localhost:7860/client/` in a
browser (open chat) or `http://localhost:7860/study/` (study mode).

### Tailscale HTTPS (for phone access)

To access from your phone over Tailscale, enable Tailscale Serve:

```bash
tailscale serve --bg 7860
```

Then open `https://<host>.<tailnet>.ts.net/client/` on your phone.

## Usage

```bash
./start.sh
```

Open `http://localhost:7860/client/` in a browser (or the Tailscale URL on phone).

## Architecture

- **STT**: Deepgram Nova-3 (speech → text)
- **LLM**: Claude Sonnet 4.5 via Anthropic API (thinking)
- **TTS**: Cartesia Sonic-3 (voice "British Reading Lady")
- **Transport**: SmallWebRTC via Pipecat (browser ↔ server)

## Study companion mode

A document-grounded variant of the regular voice tutor.

- Open `http://localhost:7860/study/` (or the Tailscale URL with `/study/` appended on phone)
- Upload a PDF / Markdown / plain-text doc (≤5MB, ≤150K characters of extracted text)
- Pick the doc, click **Start session**, grant mic access, talk through it
- Click **End session** when done; the WebRTC connection closes and the existing
  on-disconnect pipeline runs (transcript save + summary + analysis)
- A markdown recap is generated asynchronously by Haiku 4.5 and lands at
  `~/.voice-tutor/artifacts/<session-id>.md`
- The "Refresh" button on the ended view fetches `GET /api/sessions/<id>/artifact`
  and renders the markdown inline once it exists

Study sessions skip `memory.md`, the most-recent transcript, and the wiki INDEX —
the doc is the world for that session. `profile.md` still loads.

Storage:
- `~/.voice-tutor/documents/<uuid>-<original-filename>` — original upload
- `~/.voice-tutor/documents/<uuid>.txt` — extracted text used at session start
- `~/.voice-tutor/transcripts/<uuid>.json` — study session transcripts (UUID
  stem instead of datetime; the `/study/` page generates the UUID client-side)
- `~/.voice-tutor/artifacts/<uuid>.md` — the recap
- A separate `cost-log.jsonl` row with `kind: "artifact"` accounts for the
  artifact-generation Haiku call

The regular `/client/` flow is untouched — same UI, same behavior, same prompt.

## Data

Stored at `~/.voice-tutor/`:

- `profile.md` — hand-maintained identity profile, loaded into system prompt
- `transcripts/` — JSON transcripts saved on session disconnect, last 3 loaded at session start
- `transcripts/<session>.usage.json` — per-session cost breakdown sidecar (see below)

## Usage telemetry

When a session ends, the bot writes a sidecar `<session>.usage.json` next to the transcript with token counts, TTS characters, estimated audio minutes, and an estimated USD cost broken down by LLM / STT / TTS. It also:

- prints a one-line summary to the bot log
- appends a row to the session cost log at `~/second-brain/products/voice-tutor/validation/cost-log.md` (a markdown table — one session per row, easy to scan in Obsidian)

Prices are hardcoded in `bot.py` (constants near the top, verified against the official pricing pages with source URLs in comments). Refresh them when vendors change pricing.

Cost accounting:

- **LLM cost** is exact — token counts come from Anthropic's API responses (live Sonnet via pipecat metrics, post-session Haiku via `resp.usage` directly). Includes cache-read vs cache-write breakdown for the live LLM.
- **TTS cost** is exact — character count comes from pipecat's `TTSUsageMetricsData`, which reports `len(text)` of every chunk submitted to Cartesia's WebSocket (Cartesia's actual billing unit).
- **STT cost** is approximated from session wall-clock duration. Pipecat's Deepgram service doesn't surface billed minutes directly, but `SmallWebRTCTransport` streams continuously, so wall clock should track billable minutes within rounding. `tts_audio_sec_observed` and `stt_audio_sec_observed` are recorded as cross-checks.

For actual billed amounts, check each provider's dashboard.
