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
