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
