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

The server listens on `:7860`. Open `http://localhost:7860/chat/` in a
browser (open chat) or `http://localhost:7860/study/` (study mode).

### Tailscale HTTPS (for phone access)

To access from your phone over Tailscale, enable Tailscale Serve:

```bash
tailscale serve --bg 7860
```

Then open `https://<host>.<tailnet>.ts.net/chat/` on your phone.

## Usage

```bash
./start.sh
```

Open `http://localhost:7860/chat/` in a browser (or the Tailscale URL on phone).

## Architecture

- **STT**: Deepgram Nova-3 (speech → text)
- **LLM**: Claude Sonnet 4.5 via Anthropic API (thinking)
- **TTS**: Cartesia Sonic-3 (voice "British Reading Lady")
- **Transport**: SmallWebRTC via Pipecat (browser ↔ server)

## Modes

Two voice modes share the same pipeline (Deepgram → Sonnet → Cartesia), the same
post-session work (transcript save → summary → memory append → analysis), and the
same cross-session memory (`profile.md`, `memory.md`). They differ in what
context the model gets in-session and whether a recap is generated:

| | `/chat/` (open chat) | `/study/` (doc-grounded) |
|---|---|---|
| Persona | General tutor | Study companion (different system prompt) |
| `profile.md` | Loaded | Loaded |
| `memory.md` | Loaded | Loaded |
| Wiki INDEX in prompt | Yes (if `WIKI_ENABLED`) | No |
| `read_wiki_page` tool | Yes (if `WIKI_ENABLED`) | No |
| Most-recent transcript in prompt | Yes (verbatim) | No (doc takes its place) |
| Doc text in prompt | — | Yes (whole doc, ≤150K chars) |
| Post-session summary → `memory.md` | Yes (≥2 min sessions) | Yes (≥2 min sessions) |
| Post-session analysis → vault | Yes (≥2 min sessions) | Yes (≥2 min sessions) |
| Post-session recap artifact | No | Yes (Haiku → `artifacts/<id>.md`) |

The shared shape — profile + memory + per-session context — is the same for both
modes. Study mode just swaps the "context" piece from "wiki + most-recent
transcript" to "a specific document."

## Study companion mode

The document-grounded mode.

- Open `http://localhost:7860/study/` (or the Tailscale URL with `/study/` appended on phone)
- Upload a PDF / Markdown / plain-text doc (≤5MB, ≤150K characters of extracted text)
- Pick the doc, then click **Start session** on the next screen, grant mic access, talk through it
- Click **End session** when done; the WebRTC connection closes and the standard
  on-disconnect pipeline runs (transcript save → summary → memory append → analysis)
- A markdown recap is also generated asynchronously by Haiku 4.5 and lands at
  `~/.voice-tutor/artifacts/<session-id>.md` — this is study-mode only
- The "Refresh" button on the ended view fetches `GET /api/sessions/<id>/artifact`
  and renders the markdown inline once it exists

Storage:
- `~/.voice-tutor/documents/<uuid>-<original-filename>` — original upload
- `~/.voice-tutor/documents/<uuid>.txt` — extracted text used at session start
- `~/.voice-tutor/transcripts/<uuid>.json` — study session transcripts (UUID
  stem instead of datetime; the `/study/` page generates the UUID client-side)
- `~/.voice-tutor/artifacts/<uuid>.md` — the recap
- A separate `cost-log.jsonl` row with `kind: "artifact"` accounts for the
  artifact-generation Haiku call

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

## Next steps

**Polish the `/study/` UI.** Currently functional but plain — single-file HTML, system fonts, no design system. Bar: make it feel like a product surface, not a prototype. `/chat/` intentionally stays on the pipecat prebuilt UI; its diagnostic panel (live tool calls, events, connection state) is too valuable for showing the architecture under the hood. The two modes will deliberately read as different — `/chat/` is the under-the-hood view, `/study/` is the polished product surface.

**Figure out how to surface telemetry and artifacts in a demo.** Think through how to show, during a live demo, the `cost-log.md` row, the session analysis output at `~/second-brain/products/voice-tutor/session-analyses/`, the recap artifact at `~/.voice-tutor/artifacts/<id>.md`, and anything else worth pointing at (transcripts, `memory.md` growth, etc.). Today the recap renders inline in `/study/` and the cost log is a separate tab; everything else lives in `~/.voice-tutor/` or the vault.

Proposed direction: extend the `/study/` ended view to render the cost-log row, session analysis, and memory.md append inline alongside the recap — one screen surfaces every artifact a session generates the moment it ends. Beats tab-switching for the "I measure all of this" credibility beat: they watch four artifacts materialize in one visual moment instead of you narrating which tab to look at next.
