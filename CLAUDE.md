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

**Corollary — before trusting a behavior check against the running server, confirm it actually loaded the code you think it did.** Because there's no `--reload`, a merge/checkout does NOT restart uvicorn — the process keeps serving whatever was on disk when it started. Compare the server process start time (`ps -p <pid> -o lstart`) against the code's on-disk mtime / last merge; if the process predates the code, it's serving stale code and a live check validates the wrong thing. (Caught 2026-07-29: a first-run verification's step-1 gate found the server still on pre-security-fix code hours after the fixes merged.)

## `app.py` imports pipecat at module top — test via pure helpers, not `TestClient`

`app.py` does `from pipecat...` and `import bot` at module scope (lines ~25–35), so `import app` pulls in the full pipecat/ML stack and fails in any lightweight / Pipecat-free environment. Don't write route tests that do `from app import app` + a FastAPI `TestClient` — they can't run without the whole stack (and are unwinnable as a dev-harness contract, same family as "import bot without its deps").

Instead follow the repo's established pattern: put logic in pure, importable modules (`documents.py`, `session_state.py`, `grounding.py`) with no pipecat import, keep the `app.py` route a thin wrapper, and test the pure helper hermetically by monkeypatching its module-level path constants (see `tests/conftest.py`). The HTTP route stays untested at the transport layer; the logic is fully covered at the helper layer.

## Pipecat observers fire per processor hop — usage must be deduped (fixed 2026-07-22)

**Permanent Pipecat mechanism:** `BaseObserver.on_push_frame` is invoked once for EVERY frame push between processors (`frame_processor.py` calls it on each downstream/upstream hop), and one observer registered on the `PipelineTask` sees every hop pipeline-wide. So accumulating token/audio usage by `+=`-ing on each `MetricsFrame`/`InputAudioRawFrame`/`TTSAudioRawFrame` with no dedup counts each frame once per hop it travels — multiplying real usage by the hop count. The multiple equals the number of downstream hops from the emitting processor to the sink, so it differs by frame kind. Runtime tracing (2026-07-22, `VOICE_TUTOR_USAGE_TRACE`) measured it id-stable: exact integer multiples where emission point is fixed (**LLM tokens 5.00×, STT audio 8.00×**), variable where it isn't (**TTS audio 1–3 hops, ~2.63× avg**). This is why the 2026-07-20 provider reconciliation saw the ledger over-count Anthropic cache tokens ~5× and inflate `stt_audio_sec_observed` ~8×.

**Fixed (branch `fix/usage-per-hop-dedup`).** Usage accounting now lives in a pure, Pipecat-free `usage_ledger.py` (`UsageLedger`); `bot.py`'s `UsageAccumulator` is a thin observer adapter over it. The ledger dedups by `frame.id` (pipecat's process-global unique id), decided **once per frame** — so a `MetricsFrame` carrying both LLM and TTS usage still counts once. Gated by `VOICE_TUTOR_USAGE_DEDUP` (default ON; set `0/false/no/off/disable/disabled` to restore the legacy multi-count for a no-rebuild revert). Verified live: every frame counted exactly once (LLM 5×→1×, STT 8×→1×, TTS 2.63×→1×). Any new per-frame accumulator must dedup the same way (or consume usage where it's emitted once). See `products/voice-tutor/validation/2026-07-20-provider-reconciliation.md` (root cause + 2026-07-22 addendum).

## Diagnostic tools parse the secrets file directly — don't tell Matt to `source` it

`reconcile_costs.py` (and similar standalone diagnostics) parse `~/.voice-tutor-secrets.env` and the app's `.env` directly at runtime — no `source`/`set -a` needed, and a plain `source` wouldn't export vars into the Python subprocess anyway. Precedence: app `.env` first, then `~/.voice-tutor-secrets.env` overrides it, then real env vars override both. So a usage-scoped Deepgram key + `DEEPGRAM_PROJECT_ID` in the secrets file correctly shadow the app's lower-scoped `.env` `DEEPGRAM_API_KEY`. Keys are never printed. Run is just `.venv/bin/python reconcile_costs.py [--providers ...]`.

## Claim extraction is a 30–60s LLM call — never run it on the session-start path (2026-07-26)

`claims.generate_claims` (get-or-create) does a live Sonnet extraction on an uncached doc that takes **30–60s**. Study-mode session start must NOT call it — that would hang the WebRTC pipeline while the user waits. The wiring splits into two paths:

- **Session start reads cache-only** via `claims.load_fresh_claims(doc_id, text)` — a non-blocking, `source_hash`-verified read that returns the map only if a *fresh* sidecar exists, else `None`. On `None`, `build_system_instruction` omits the claim map and the session **degrades to plain study mode** rather than blocking. Never swap this for `generate_claims` "to be safe" — you'd reintroduce the hang.
- **Extraction is warmed ahead of time** by `POST /api/documents/{id}/claims/prepare`, fired fire-and-forget from `selectDoc()` in `static/study.html` the moment a doc is picked. It's idempotent + non-blocking (no-ops if `cached`/`in_flight`; runs `generate_claims` off the event loop via `asyncio.to_thread`), so it's safe to call on every selection. The in-flight dedup set relies on the endpoint's await-free prologue being atomic under the single-threaded event loop — keep it await-free.

The map is injected **after the document, before the reminders** (position is load-bearing; `tests/test_study_claim_steering.py` pins it). The live prompt is a **condensed v0**, not the fuller `products/voice-tutor/planning/2026-07-23-study-tutor-prompt-v1.md` draft. Each session row carries a `prompt_hash` (hash of the static base+reminders, doc-independent) so sessions are attributable to the exact prompt version.

## Changing a filename scheme — grep for READERS of the pattern, not just the writer (2026-07-27)

A green suite only proves the **writer**. When you change how files are named, the silent failure is a **reader** that reconstructs the old name — and in this repo the highest-risk readers are the `app.py` routes, which are **deliberately untested at the transport layer** (see "test via pure helpers, not `TestClient`" above), so *nothing in the suite fails*.

**Worked example:** the session-analysis rename to date-first names (`session-analysis-<YYYY-MM-DD-HHMMSS>-<shortid>.md`) updated the writer (`bot.py` via `session_naming.session_analysis_filename`) and had a passing builder test — but `app.py` still looked files up by the **old exact name** `session-analysis-<full-uuid>.md` in two places (`/view/sessions/{id}/analysis` and the `/telemetry` composite). Result: every UUID session's analysis silently 404'd, the "Session analysis" diagnostic vanished from `static/study.html` (it's gated on `data.analysis`), and long sessions polled to their cap because the poll `done`-condition requires `data.analysis`. The suite stayed green the whole time. Fixed by a pure reader `session_naming.find_analysis_path()` (globs on the shortid the writer embeds; `SHORTID_LEN` shared so the two can't drift) routed through both call sites.

**Rule:** before shipping a naming/scheme change, `grep -rn` for **consumers** of the pattern (both `-` and `_` spellings, string literals, and path-building `f"..."`), and pin the **round-trip** — write under the builder's name, assert the finder locates it — not just the builder in isolation. Writer + reader agreeing is the property.

## Client-controllable ids that become file paths — sanitize at a shared choke point (2026-07-29)

Any id that arrives from the client (WebRTC offer body, request path, or a value
read back from a stored log that was itself written from a client value) and is then
used as a **filesystem path component** must be collapsed to a single component with
`Path(x).name` before the path is built — a crafted `../<other-user>/<id>` (or an
absolute path) otherwise escapes the caller's `<user_id>/` namespace and reads or
writes another user's data.

Put the guard at the **shared helper/boundary every path funnels through**, not at
each call site — containment must not depend on every caller remembering:
- `doc_id` → `documents._load_from_dir`, `claims._claims_path`, `claims._resolve_doc_namespace` (a crafted id in the last one also mis-routed `generate_claims`' WRITE into `_shared/`).
- `session_id` → `session_naming.safe_session_id`, applied at the SINGLE `study_meta["session_id"]` construction in `bot.py` (every downstream writer inherits it) and at the `study_history` persisted-log read (a second trust boundary — pre-guard rows).
- `app.py` routes already do `safe_id = Path(id).name` — mirror that everywhere else.

Two traps this class hides behind: (1) `bot.py`'s `bot()` coroutine is deliberately
untested at the transport layer (see "test via pure helpers, not `TestClient`"), so a
green suite never catches an unguarded writer there — pin the property at the pure
helper it uses. (2) The guard itself must be safe when joined **bare**:
`Path("..").name == ".."`, so a helper that returns it is safe only by the convention
that callers append a suffix — instead return a placeholder for `""`/`"."`/`".."` (see
`session_naming.safe_session_id`). Regression tests: `tests/test_cross_user_doc_traversal.py`,
`tests/test_session_id_traversal.py` (and note pathlib+os.stat only traverses `..` when the
caller's OWN dir exists — a traversal test must materialize it or it passes vacuously).

## Production is a launchd agent on :7860; never run `./start.sh` by hand; dev on a worktree (2026-08-01)

Production is the always-on launchd agent `com.voice-tutor.server` (KeepAlive + RunAtLoad)
running `start.sh` on `0.0.0.0:7860`, exposed publicly via Tailscale Funnel
(`https://matts-mac-mini.taild1f9b7.ts.net/` → :7860). It survives crashes and reboots.

- **Don't run `./start.sh` by hand on this machine.** Its `lsof -ti :7860 | xargs kill -9`
  (~line 8) would hard-kill the live server (then fight KeepAlive for the port), destroying
  any in-flight tester session's transcript/recap. A guard now blocks this: `start.sh` refuses
  an interactive run while the agent is loaded and prints the kickstart command instead. It
  passes the launchd-spawned instance through by checking `XPC_SERVICE_NAME == com.voice-tutor.server`
  (the label launchd sets), so production's own restart is unaffected — never change the guard
  in a way that also blocks that instance, or the agent won't restart.
- **Restart production** (only when confirmed idle — a restart drops in-flight sessions):
  `launchctl kickstart -k gui/$(id -u)/com.voice-tutor.server`. No deploy pipeline exists:
  "deploy" = land code on `main` in `~/development/voice-tutor`, then kickstart.
- **Test locally on the isolated worktree, never in the live checkout** — editing `static/` in
  the live checkout changes what testers see *instantly* (files are served per-request). Worktree:
  `~/development/voice-tutor-dev` (branch `local-dev`); `./dev.sh` runs uvicorn on `127.0.0.1:7861`
  with `--reload`, localhost-only, reusing the main venv and reading `.env` read-only. The lane
  isolates **code, not data** — it still writes to shared `~/.voice-tutor/` + `~/second-brain/`
  and uses the same `tokens.json`.
- **Rebase `local-dev` before trusting a local test.** The worktree is a branch, not a mirror: every
  merge to `main` leaves it further behind, and `dev.sh` happily serves the stale code with no
  warning that it's out of date. On 2026-08-02 it sat 4 commits behind — it would still have
  addressed testers as "Matt" and used the retired 120s analysis gate, i.e. reproduced bugs that
  were already fixed in production. `git rebase main local-dev` in the worktree (it carries one
  commit, `dev.sh`, that nothing else touches, so conflicts are unlikely). Check with
  `git log --oneline local-dev..main` before any local verification you intend to believe.

## Provisioning a tester — add a token to `tokens.json` (no restart) (2026-08-01)

Testers authenticate via an invite token in `~/.voice-tutor/tokens.json` (`{token: user_id}`).
To add one: mint a 32-char base62 token, add `{token: "<name>"}`, hand over
`https://matts-mac-mini.taild1f9b7.ts.net/study/?u=<token>`. The registry is read fresh on every
authenticated request (`app.py:164` → `identity.resolve_cookie` → `load_registry`, `identity.py:71`
+ `42`; verified live — a throwaway token resolved on the running server with no restart, pid
unchanged), so a new token works **immediately, no restart**. `tokens.json` gates *every* tester,
so a corrupt write 403s everyone — back it up, write atomically (temp + `os.replace`), preserve
existing entries, and verify with `identity.resolve_user(token, load_registry())`.

## `session-log.jsonl` has ROW KINDS, and an unknown kind is silently dropped (2026-08-05)

The ledger is not homogeneous: rows carry `kind` — `session`, `artifact`, and now
`coverage` — plus legacy rows with no `kind` at all. Every reader dispatches on it,
and **the default branch is silence, not an error**: `reconcile_costs._row_kind`
returns an unrecognized kind verbatim and `summarize_ledger` skips it ("Unknown
kinds contribute nothing"); `cost_audit._classify_and_check_row` returns it
unchecked. So adding a row kind that spends money and stopping at the writer
produces a ledger that looks complete and totals that are quietly short — the
provider then reads as ahead of us, which is exactly the phantom drift
`reconcile_costs` exists to tell apart from real logging errors.

**Adding a `kind` means updating its readers in the same change:**
`reconcile_costs.py` (`_row_kind`, `filter_rows_by_local_range`,
`summarize_ledger`, and `LedgerTotals`' row counters), `cost_audit.py`
(cost-recompute branch), and any consumer that filters `kind == "session"`. A row
with no timestamp of its own (artifact, coverage) must be joined to its session's
`session_start` via `session_id` or it vanishes from every dated range. Same family
as the filename-scheme rule above — caught 2026-08-05, when the coverage judge's
`kind:"coverage"` rows carried real Haiku tokens no reconciliation counted, suite
green throughout.

**Related invariant — coverage sidecars are APPEND-ONLY, so wrong beats missing
is FALSE here.** `coverage_store.write_sidecar` refuses to overwrite, which makes
the accumulated bar monotonic but also makes any number it records permanent
(only `backfill_coverage.py --force` can revise it). When the judge's output is
suspect, the correct degradation is **no sidecar**, not a plausible one — see
`coverage_judge.MassCitationDowngradeError`.
