# Voice Tutor

## HTTP routing — `/chat/` (prebuilt RTVI UI) requires three routes

The pipecat prebuilt client mounted at `/chat/` does NOT just call `/api/offer`. It expects:
1. `POST /start` → returns `{sessionId, iceConfig?}`
2. `POST /sessions/{sessionId}/api/offer` → forwards to our `offer()` handler
3. `PATCH /sessions/{sessionId}/api/ice-candidate` → forwards to our `ice_candidate()` handler

These mirror `pipecat.runner.run.main`. If you change `app.py`'s routing, do not delete or rename them — `/chat/` will silently break (Not Found, immediate disconnect) while `/study/` keeps working (it talks to `/api/offer` directly).

## Pipecat upgrades

We pin `pipecat-ai` deliberately. The 0.0.x → 1.0.0 cut on 2026-04-14 is a major version with breaking changes to frame/transport/runner APIs — the exact surfaces `bot.py` and `app.py` use. Read the changelog and bump on its own branch; never bundle a pipecat major bump with feature work.

## `./start.sh` has a ~5s cold start before listening

The first thing the script prints is the pipecat banner (from `import pipecat`), but uvicorn hasn't bound to `:7860` yet — heavy ML imports (transformers, onnxruntime, numba, opencv, scipy) take a few more seconds to load. Opening `http://localhost:7860/study/` in this window returns "site can't be reached" / connection refused. Wait for the `INFO: Application startup complete` line before trying the browser; that's uvicorn telling you the port is actually bound.

## Python changes require a server restart; static files do not

`./start.sh` runs uvicorn without `--reload`, so any change to a `.py` file — including module-level string constants like `VIEWER_HTML` in `app.py` — only takes effect after re-running `./start.sh` (which kills the bound port and re-imports). Static files in `static/` (study.html, JS, CSS) are served via `FileResponse` per request and pick up edits without a restart. Don't tell the user "no restart needed" without checking which side of that line the edit lives on.

## `app.py` imports pipecat at module top — test via pure helpers, not `TestClient`

`app.py` does `from pipecat...` and `import bot` at module scope (lines ~25–35), so `import app` pulls in the full pipecat/ML stack and fails in any lightweight / Pipecat-free environment. Don't write route tests that do `from app import app` + a FastAPI `TestClient` — they can't run without the whole stack (and are unwinnable as a dev-harness contract, same family as "import bot without its deps").

Instead follow the repo's established pattern: put logic in pure, importable modules (`documents.py`, `session_state.py`, `grounding.py`) with no pipecat import, keep the `app.py` route a thin wrapper, and test the pure helper hermetically by monkeypatching its module-level path constants (see `tests/conftest.py`). The HTTP route stays untested at the transport layer; the logic is fully covered at the helper layer.

## Pipecat observers fire per processor hop — dedup before accumulating usage

`BaseObserver.on_push_frame` is invoked once for EVERY frame push between processors (`frame_processor.py` calls it on each downstream/upstream hop), and one observer registered on the `PipelineTask` sees every hop pipeline-wide. So accumulating token/audio usage by `+=`-ing on each `MetricsFrame`/`InputAudioRawFrame` with no dedup counts each frame once per hop it travels — multiplying real usage by the hop count. This is a live bug in `UsageAccumulator` (`bot.py:88-126`): provider reconciliation (2026-07-20) measured the ledger over-counting Anthropic cache tokens by an exact per-session integer (5.00× on a controlled day) and inflating `stt_audio_sec_observed` ~8×. Any per-frame accumulation must dedup (track seen frame ids, or filter to one source/direction), or consume usage where it's emitted once. See `products/voice-tutor/validation/2026-07-20-provider-reconciliation.md`. (Rewrite this once the dedup fix lands.)

## Diagnostic tools parse the secrets file directly — don't tell Matt to `source` it

`reconcile_costs.py` (and similar standalone diagnostics) parse `~/.voice-tutor-secrets.env` and the app's `.env` directly at runtime — no `source`/`set -a` needed, and a plain `source` wouldn't export vars into the Python subprocess anyway. Precedence: app `.env` first, then `~/.voice-tutor-secrets.env` overrides it, then real env vars override both. So a usage-scoped Deepgram key + `DEEPGRAM_PROJECT_ID` in the secrets file correctly shadow the app's lower-scoped `.env` `DEEPGRAM_API_KEY`. Keys are never printed. Run is just `.venv/bin/python reconcile_costs.py [--providers ...]`.
