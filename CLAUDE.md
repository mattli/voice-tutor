# Voice Tutor

## HTTP routing — `/client/` (prebuilt RTVI UI) requires three routes

The pipecat prebuilt client at `/client/` does NOT just call `/api/offer`. It expects:
1. `POST /start` → returns `{sessionId, iceConfig?}`
2. `POST /sessions/{sessionId}/api/offer` → forwards to our `offer()` handler
3. `PATCH /sessions/{sessionId}/api/ice-candidate` → forwards to our `ice_candidate()` handler

These mirror `pipecat.runner.run.main`. If you change `app.py`'s routing, do not delete or rename them — `/client/` will silently break (Not Found, immediate disconnect) while `/study/` keeps working (it talks to `/api/offer` directly).

## Pipecat upgrades

We pin `pipecat-ai` deliberately. The 0.0.x → 1.0.0 cut on 2026-04-14 is a major version with breaking changes to frame/transport/runner APIs — the exact surfaces `bot.py` and `app.py` use. Read the changelog and bump on its own branch; never bundle a pipecat major bump with feature work.
