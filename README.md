# Voice Tutor

Self-hosted voice conversation service using Pipecat, Claude, and persistent memory. Runs on Mac Mini, accessed from phone browser via Tailscale.

## Setup

### API Keys

Create a `.env` file with:

```
ANTHROPIC_API_KEY=your-key
DEEPGRAM_API_KEY=your-key
CARTESIA_API_KEY=your-key
```

### Install

```bash
uv sync
```

### Tailscale HTTPS

To access from phone, enable Tailscale Serve:

```bash
tailscale serve --bg 7860
```

Then open `https://matts-mac-mini.taild1f9b7.ts.net/client/` on your phone.

## Usage

```bash
./start.sh
```

Open `http://localhost:7860/client/` in a browser (or the Tailscale URL on phone).

## Architecture

- **STT**: Deepgram Nova-3 (speech → text)
- **LLM**: Claude Sonnet 4.5 via Anthropic API (thinking)
- **TTS**: Cartesia Sonic (text → speech)
- **Transport**: SmallWebRTC via Pipecat (browser ↔ server)

## Data

Stored at `~/.voice-tutor/`:

- `profile.md` — hand-maintained identity profile, loaded into system prompt
- `transcripts/` — JSON transcripts saved on session disconnect, last 3 loaded at session start
- `transcripts/<session>.usage.json` — per-session cost breakdown sidecar (see below)

## Usage telemetry

When a session ends, the bot writes a sidecar `<session>.usage.json` next to the transcript with token counts, TTS characters, estimated audio minutes, and an estimated USD cost broken down by LLM / STT / TTS. It also:

- prints a one-line summary to the bot log
- appends a row to the session cost log at `~/second-brain/products/voice-tutor/cost-log.md` (a markdown table — one session per row, easy to scan in Obsidian)

Prices are hardcoded in `bot.py` (constants near the top, verified against the official pricing pages with source URLs in comments). Refresh them when vendors change pricing.

Known approximations:

- **STT cost** estimated from session duration (Pipecat's Deepgram service doesn't emit audio-seconds directly)
- **TTS cost** estimated from character count converted to audio-seconds at ~14 chars/sec, then to Cartesia credits at 15 credits/sec
- **LLM cost** is exact (tokens reported by Anthropic, including cache-read vs cache-write breakdown)

For actual billed amounts, check each provider's dashboard.
