"""FastAPI app for voice-tutor.

Owns the HTTP surface so we can add study-mode routes alongside the WebRTC
offer flow. Replaces pipecat.runner.run.main — that helper hides the FastAPI
app inside its CLI entry point with no extension hook, so we replicate the
~30 lines of WebRTC plumbing it would have set up.

The voice pipeline lives in bot.py; this module only handles HTTP.
"""

import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from html import escape as html_escape
from http import HTTPMethod
from pathlib import Path
from typing import Any, Dict

import re

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from pipecat.runner.types import SmallWebRTCRunnerArguments
from pipecat.transports.smallwebrtc.connection import IceServer, SmallWebRTCConnection
from pipecat.transports.smallwebrtc.request_handler import (
    IceCandidate,
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)
from pipecat_ai_small_webrtc_prebuilt.frontend import SmallWebRTCPrebuiltUI

import bot
import claims
import documents
import identity
import session_naming
import session_state
import sessions

HOST = os.getenv("VOICE_TUTOR_HOST", "0.0.0.0")

# WebRTC ICE configuration. The public deployment (Tailscale Funnel) tunnels only
# the HTTPS signaling, not the UDP media, so a REMOTE peer cannot reach the
# server's host/tailnet ICE candidates — a TURN relay is required for anyone off
# the local network (local/tailnet peers still connect directly and use no relay).
# Only the username/credential are secret (loaded from .env); the metered relay
# URLs and the STUN URL are public. When the two env vars are ABSENT the config
# degrades to STUN-only (the prior behavior), so removing them + restarting is a
# complete no-rebuild revert. Configured on BOTH sides: the server handler (so the
# server gathers reachable relay candidates) and the browser (via /api/ice-config,
# consumed by static/study.html) — both are required for a remote media path.
_TURN_USERNAME = os.getenv("METERED_TURN_USERNAME")
_TURN_CREDENTIAL = os.getenv("METERED_TURN_CREDENTIAL")
# Two independent STUN providers so metered.ca is not a single point of failure:
# if it is degraded or over quota the browser still has Google's STUN to gather a
# srflx candidate. Browsers query STUN servers in parallel, so the extra entry
# costs no startup latency.
_STUN_URLS = ("stun:stun.relay.metered.ca:80", "stun:stun.l.google.com:19302")
_TURN_URLS = (
    "turn:global.relay.metered.ca:80",
    "turn:global.relay.metered.ca:80?transport=tcp",
    "turn:global.relay.metered.ca:443",
    "turns:global.relay.metered.ca:443?transport=tcp",
)


def ice_servers_dicts() -> list[dict]:
    """ICE servers as plain dicts for the browser's RTCPeerConnection. STUN always;
    the authenticated TURN relays are added only when creds are configured."""
    servers: list[dict] = [{"urls": u} for u in _STUN_URLS]
    if _TURN_USERNAME and _TURN_CREDENTIAL:
        servers += [
            {"urls": u, "username": _TURN_USERNAME, "credential": _TURN_CREDENTIAL}
            for u in _TURN_URLS
        ]
    return servers


def _ice_servers_objs() -> list[IceServer]:
    """Server-side ICE servers for aiortc's own candidate gathering. Deliberately
    LEANER than the browser's list: the server only needs ONE reachable relay
    candidate, so it uses just the UDP TURN endpoint (the one that allocates from
    this host) and NO STUN. The metered TCP/TLS relay endpoints are outbound-blocked
    from the Mini and only add ~5s of gathering latency (failed allocations time
    out), and STUN srflx is unnecessary when a relay exists and its retries on dead
    interfaces stall the answer. Empty when creds are absent → host-only gathering
    (the prior local-only behavior)."""
    if _TURN_USERNAME and _TURN_CREDENTIAL:
        return [
            IceServer(
                urls="turn:global.relay.metered.ca:80",
                username=_TURN_USERNAME,
                credential=_TURN_CREDENTIAL,
            )
        ]
    return []


small_webrtc_handler = SmallWebRTCRequestHandler(
    esp32_mode=False, host=HOST, ice_servers=_ice_servers_objs()
)

# Startup status line for the TURN relay. Without it, a missing/invalid/over-quota
# relay is INDISTINGUISHABLE at the log level from the pre-fix bug: a remote tester
# just spins on "connecting" and nothing is written anywhere. Prints the configured
# state only — never the credentials.
print(
    "[webrtc] TURN relay: CONFIGURED (remote peers can connect)"
    if (_TURN_USERNAME and _TURN_CREDENTIAL)
    else "[webrtc] TURN relay: NOT CONFIGURED — remote/internet peers will FAIL to "
    "connect (set METERED_TURN_USERNAME + METERED_TURN_CREDENTIAL in .env)",
    flush=True,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await small_webrtc_handler.close()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/chat", SmallWebRTCPrebuiltUI)


def require_user(request: Request) -> str:
    """FastAPI dependency: resolve the identity cookie to a user_id, or 403.
    Fail closed — no name-picker, no guessing."""
    user_id = identity.resolve_cookie(request.cookies.get(identity.COOKIE_NAME))
    if user_id is None:
        raise HTTPException(status_code=403, detail="no valid identity")
    return user_id


@app.get("/api/whoami")
async def whoami(user_id: str = Depends(require_user)):
    return {"user_id": user_id}


@app.get("/api/ice-config")
async def ice_config(user_id: str = Depends(require_user)):
    """ICE servers (STUN + authenticated TURN relays) for the browser. Auth-gated
    so only invited users receive the relay credentials; static/study.html fetches
    this right before creating its RTCPeerConnection, and falls back to STUN-only
    if it fails."""
    return {"iceServers": ice_servers_dicts()}


@app.post("/api/offer")
async def offer(
    request: SmallWebRTCRequest, background_tasks: BackgroundTasks, http_request: Request
):
    user_id = identity.resolve_cookie(http_request.cookies.get(identity.COOKIE_NAME))
    if user_id is None:
        raise HTTPException(status_code=403, detail="no valid identity")
    rd = dict(request.request_data or {})
    rd["user_id"] = user_id  # server-stamped; client-supplied value is ignored
    request.request_data = rd

    async def webrtc_connection_callback(connection: SmallWebRTCConnection):
        runner_args = SmallWebRTCRunnerArguments(
            webrtc_connection=connection,
            body=request.request_data,
        )
        background_tasks.add_task(bot.bot, runner_args)

    return await small_webrtc_handler.handle_web_request(
        request=request,
        webrtc_connection_callback=webrtc_connection_callback,
    )


@app.patch("/api/offer")
async def ice_candidate(request: SmallWebRTCPatchRequest):
    await small_webrtc_handler.handle_patch_request(request)
    return {"status": "success"}


# RTVI client (used by the pipecat prebuilt UI at /chat/) bootstraps via
# POST /start, then routes its WebRTC offer/patch through the per-session
# proxy. Both endpoints mirror pipecat.runner.run.main's /start + /sessions
# handlers so the prebuilt UI keeps working alongside our own /study/ flow,
# which talks directly to /api/offer.
active_sessions: Dict[str, Dict[str, Any]] = {}


@app.post("/start")
async def rtvi_start(request: Request, user_id: str = Depends(require_user)):
    # Auth-gated for the same reason as /api/ice-config: the iceConfig below can
    # carry the TURN relay CREDENTIALS. Without this dependency the route handed
    # them to any unauthenticated caller on the public URL (letting a stranger
    # relay traffic on our metered.ca quota). /chat/'s offers already 403 at
    # offer(), so gating here costs that flow nothing it wasn't already denied.
    try:
        request_data = await request.json()
    except Exception:
        request_data = {}
    session_id = str(uuid.uuid4())
    active_sessions[session_id] = request_data.get("body") or {}
    result: Dict[str, Any] = {"sessionId": session_id}
    if request_data.get("enableDefaultIceServers"):
        result["iceConfig"] = {"iceServers": ice_servers_dicts()}
    return result


@app.api_route(
    "/sessions/{session_id}/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
)
async def rtvi_proxy(
    session_id: str, path: str, request: Request, background_tasks: BackgroundTasks
):
    active_session = active_sessions.get(session_id)
    if active_session is None:
        return Response(content="Invalid or not-yet-ready session_id", status_code=404)

    if path.endswith("api/offer"):
        try:
            body = await request.json()
            if request.method == HTTPMethod.POST.value:
                webrtc_request = SmallWebRTCRequest(
                    sdp=body["sdp"],
                    type=body["type"],
                    pc_id=body.get("pc_id"),
                    restart_pc=body.get("restart_pc"),
                    request_data=body.get("request_data")
                    or body.get("requestData")
                    or active_session,
                )
                return await offer(webrtc_request, background_tasks, request)
            if request.method == HTTPMethod.PATCH.value:
                patch = SmallWebRTCPatchRequest(
                    pc_id=body["pc_id"],
                    candidates=[IceCandidate(**c) for c in body.get("candidates", [])],
                )
                return await ice_candidate(patch)
        except Exception:
            return Response(content="Invalid WebRTC request", status_code=400)

    return Response(status_code=200)


@app.post("/api/documents")
async def upload_document(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(require_user),
):
    try:
        raw = await file.read()
        result = documents.save_upload(user_id, file.filename or "untitled", raw)
    except documents.UploadError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)

    # Auto-warm (goal Part 1): a successful upload should leave the doc ready to
    # study — claims extracted automatically — with no ritual picker-click first.
    # Schedule the SAME background warm the prepare/picker path uses, scoped to
    # the uploading user's namespace, so extraction funnels through the shared
    # generate_claims cache guard AND the input tripwire (nothing here re-checks
    # the bound or re-counts words). Upload success is INDEPENDENT of extraction:
    # _warm_claims is best-effort (it swallows every exception), and scheduling a
    # background task cannot fail the response already computed above — so an API
    # error, or the tripwire rejecting an oversized doc, degrades the doc to plain
    # unwarmed study exactly as an unwarmed doc does today, no new upload failure.
    doc_id = result["document_id"]
    _schedule_warm(background_tasks, user_id, doc_id)

    # Surface (display-only) the tripwire's rejection reason for an oversized doc
    # so the uploading user is told WHY it won't be steered — the file still
    # landed and is loadable. This reuses claims' single source of truth for the
    # message (claims.rejection_reason) rather than re-enforcing the bound here;
    # the warm above is scheduled regardless, and the durable rejection log line
    # is emitted by the warm seam when extraction is actually attempted. Additive
    # optional field: null/absent for a doc within the bound.
    # The upload already succeeded and the file already landed (save_upload above);
    # this annotation is DISPLAY-ONLY. Guard the re-read + reason computation so a
    # transient disk-read hiccup (or any failure here) degrades to "no annotation"
    # rather than turning a good upload into an error response.
    try:
        loaded = documents.load_document(user_id, doc_id)
        if loaded is not None:
            reason = claims.rejection_reason(loaded[1])
            if reason is not None:
                result = {**result, "claim_extraction_rejected": reason}
    except Exception as e:  # best-effort annotation; never fails the upload
        print(f"[upload] rejection-reason annotation skipped for {doc_id}: {e!r}", flush=True)
    return result


@app.get("/api/documents")
async def list_documents_route(user_id: str = Depends(require_user)):
    return await documents.list_documents(user_id)


# Claim-map warming. Claim extraction is a 30-60s live LLM call on an uncached
# doc, so it must NOT run on the session-start path (it would hang the pipeline).
# Instead the frontend fires this endpoint the moment a doc is selected, warming
# the sidecar cache in the background so the map is ready by the time a session
# starts. bot.py reads the result via claims.load_fresh_claims (cache-only).
#
# `_claims_warming` de-dups in-flight extractions. Access is confined to the
# synchronous (await-free) prologue of prepare_claims and to _warm_claims's
# terminal discard, so the single-threaded event loop makes check-and-add atomic
# — no lock needed. Keyed by (user_id, doc_id) so two users warming the same
# doc_id never collide.
_claims_warming: set[tuple[str, str]] = set()


async def _warm_claims(user_id: str, doc_id: str) -> None:
    """Background task: run get-or-create for ``user_id``'s ``doc_id`` off the event loop.

    Extraction (``claims.generate_claims``) is blocking, so it is offloaded to a
    worker thread. The in-flight marker is always cleared, even on failure, so a
    transient extraction error doesn't permanently wedge the doc as "warming".
    """
    try:
        loaded = documents.load_document(user_id, doc_id)
        if loaded is None:
            return
        _title, text = loaded
        await asyncio.to_thread(claims.generate_claims, user_id, doc_id, text)
    except Exception as e:  # extraction is best-effort; a failure just means no map
        print(f"[claims-warm] {doc_id} extraction failed: {e!r}", flush=True)
    finally:
        _claims_warming.discard((user_id, doc_id))


def _schedule_warm(background_tasks: BackgroundTasks, user_id: str, doc_id: str) -> bool:
    """Schedule the shared background warm for ``user_id``'s ``doc_id``, de-duped.

    The SINGLE scheduling seam shared by the prepare/picker path and the
    upload-completion auto-warm, so both funnel through the identical
    ``_claims_warming`` in-flight guard and the same ``_warm_claims`` task — and
    thus through generate_claims' source_hash cache guard and the input tripwire.
    Returns True if a task was scheduled, False if one was already in flight for
    this (user_id, doc_id). Never performs extraction itself and never raises;
    idempotency-by-freshness and the bound are enforced INSIDE ``_warm_claims``.
    """
    if (user_id, doc_id) in _claims_warming:
        return False
    _claims_warming.add((user_id, doc_id))
    background_tasks.add_task(_warm_claims, user_id, doc_id)
    return True


@app.post("/api/documents/{doc_id}/claims/prepare", status_code=202)
async def prepare_claims(
    doc_id: str, background_tasks: BackgroundTasks, user_id: str = Depends(require_user)
):
    """Idempotent, non-blocking warm of ``doc_id``'s claim map.

    Returns immediately with a status the caller may ignore (fire-and-forget):
      * ``unknown``   — no such document; nothing to warm.
      * ``cached``    — a fresh claim map already exists (no work scheduled).
      * ``in_flight`` — extraction is already running for this doc.
      * ``warming``   — extraction was just scheduled in the background.

    Safe to call unconditionally on every doc selection.
    """
    safe_id = Path(doc_id).name  # path-traversal guard, mirrors other routes
    loaded = documents.load_document(user_id, safe_id)
    if loaded is None:
        return {"status": "unknown"}
    _title, text = loaded
    if claims.load_fresh_claims(user_id, safe_id, text) is not None:
        return {"status": "cached"}
    if not _schedule_warm(background_tasks, user_id, safe_id):
        return {"status": "in_flight"}
    return {"status": "warming"}


STUDY_HTML = Path(__file__).parent / "static" / "study.html"


@app.get("/study/", include_in_schema=False)
@app.get("/study", include_in_schema=False)
async def study_page(request: Request, u: str | None = Query(None)):
    cookie_uid = identity.resolve_cookie(request.cookies.get(identity.COOKIE_NAME))
    token_uid = identity.resolve_cookie(u) if u else None
    # URL param present + valid → set/refresh cookie, redirect to clean URL.
    if token_uid is not None:
        resp = RedirectResponse(url="/study/", status_code=303)
        resp.set_cookie(
            identity.COOKIE_NAME,
            u,
            max_age=identity.COOKIE_MAX_AGE,
            httponly=True,
            samesite="lax",
            path="/",
            secure=request.url.scheme == "https",
        )
        return resp
    if cookie_uid is not None:
        return FileResponse(STUDY_HTML, media_type="text/html")
    # Neither cookie nor valid token → paste-your-code gate. Fail closed.
    return HTMLResponse(identity.GATE_HTML)


VOICE_TUTOR_DIR = Path.home() / ".voice-tutor"
ARTIFACTS_DIR = VOICE_TUTOR_DIR / "artifacts"
TRANSCRIPTS_DIR = VOICE_TUTOR_DIR / "transcripts"
SESSION_ANALYSES_DIR = Path.home() / "second-brain" / "products" / "voice-tutor" / "session-analyses"
COST_LOG_PATH = Path.home() / "second-brain" / "products" / "voice-tutor" / "validation" / "cost-log.md"


@app.get("/api/sessions/{session_id}/artifact")
async def get_artifact(session_id: str, user_id: str = Depends(require_user)):
    safe_id = Path(session_id).name  # belt and suspenders against path traversal
    if not sessions.session_belongs_to(user_id, safe_id):
        raise HTTPException(status_code=404, detail="artifact not ready or not found")
    path = ARTIFACTS_DIR / user_id / f"{safe_id}.md"
    if not path.exists():
        raise HTTPException(status_code=404, detail="artifact not ready or not found")
    return FileResponse(path, media_type="text/markdown")


@app.get("/api/sessions/latest")
async def get_latest_session(user_id: str = Depends(require_user)):
    """Most recent study session for ``user_id``, used by the picker-screen 'View
    last session' link. Iterates session-log.jsonl in reverse since study session
    rows are appended at session end and carry the UUID + document_id we need."""
    jsonl_path = Path.home() / "second-brain" / "products" / "voice-tutor" / "validation" / "session-log.jsonl"
    if not jsonl_path.exists():
        raise HTTPException(status_code=404, detail="no sessions yet")
    with jsonl_path.open() as f:
        lines = f.readlines()
    for line in reversed(lines):
        try:
            entry = json.loads(line)
        except Exception:
            continue
        if entry.get("kind") != "session" or entry.get("mode") != "study":
            continue
        if entry.get("user_id") != user_id:
            continue
        doc_id = entry.get("document_id")
        loaded = documents.load_document(user_id, doc_id) if doc_id else None
        return {
            "session_id": entry["session_id"],
            "document_id": doc_id,
            "document_title": loaded[0] if loaded else None,
        }
    raise HTTPException(status_code=404, detail="no study session yet")


@app.get("/api/sessions")
async def list_sessions(user_id: str = Depends(require_user)):
    """All completed study sessions for ``user_id``, newest first, for the
    /study/ history surface. Thin wrapper — all listing logic lives in the pure
    sessions.py helper (Pipecat-free, hermetically tested)."""
    return sessions.list_study_sessions(user_id)


SESSION_LOG_JSONL_PATH = Path.home() / "second-brain" / "products" / "voice-tutor" / "validation" / "session-log.jsonl"


def _lookup_session_doc(user_id: str, session_id: str) -> dict | None:
    """Look up the document_id/title for a session from session-log.jsonl.
    Returns None if not found."""
    if not SESSION_LOG_JSONL_PATH.exists():
        return None
    with SESSION_LOG_JSONL_PATH.open() as f:
        for line in f:
            try:
                entry = json.loads(line)
            except Exception:
                continue
            if entry.get("kind") == "session" and entry.get("session_id") == session_id:
                doc_id = entry.get("document_id")
                loaded = documents.load_document(user_id, doc_id) if doc_id else None
                return {
                    "document_id": doc_id,
                    "document_title": loaded[0] if loaded else None,
                }
    return None


@app.get("/api/sessions/{session_id}/telemetry")
async def get_telemetry(session_id: str, user_id: str = Depends(require_user)):
    """Composite endpoint for the /study/ ended view. Each field is null until
    that artifact lands; the frontend polls and renders pieces progressively.

    Includes the recap so the frontend polls a single URL rather than juggling
    `/artifact` + `/telemetry` independently. Non-Matt users get the Matt-only
    fields (analysis, has_prompt) redacted before the response leaves this
    function — see ``sessions.redact_telemetry_for_user``."""
    safe_id = Path(session_id).name
    if not sessions.session_belongs_to(user_id, safe_id):
        raise HTTPException(status_code=404, detail="session not found")
    artifact_path = ARTIFACTS_DIR / user_id / f"{safe_id}.md"
    usage_path = TRANSCRIPTS_DIR / user_id / f"{safe_id}.usage.json"
    summary_path = TRANSCRIPTS_DIR / user_id / f"{safe_id}.summary.md"
    analysis_path = session_naming.find_analysis_path(SESSION_ANALYSES_DIR, user_id, safe_id)
    prompt_path = TRANSCRIPTS_DIR / user_id / f"{safe_id}.prompt.txt"
    doc_info = _lookup_session_doc(user_id, safe_id) or {"document_id": None, "document_title": None}
    result = {
        "recap": artifact_path.read_text() if artifact_path.exists() else None,
        "cost": json.loads(usage_path.read_text()) if usage_path.exists() else None,
        "memory_append": summary_path.read_text() if summary_path.exists() else None,
        "analysis": analysis_path.read_text() if analysis_path else None,
        "has_prompt": prompt_path.exists(),
        # The frontend uses this to decide whether to wait for memory_append /
        # analysis on shorter sessions. Mirrors bot.py's MIN_SUMMARY_DURATION_SEC.
        "min_summary_sec": bot.MIN_SUMMARY_DURATION_SEC,
        # Document context for page-load restoration when URL has ?session=<id>.
        "document_id": doc_info["document_id"],
        "document_title": doc_info["document_title"],
    }
    return sessions.redact_telemetry_for_user(result, user_id)


# ─── Viewer pages: render persistent system state for the demo ──────────
# Each route serves a self-contained HTML page that renders the underlying
# file. New-tab navigation from /study/ deliberately signals "this is the
# underlying data" (different URL = different register) rather than making
# these feel like first-class product features.

_UUID_RE = re.compile(r"^[0-9a-fA-F-]{36}$")


def _back_href(from_session: str | None) -> str:
    """Build the in-page back link target. Only honor a from= value that looks
    like a UUID, so we don't get tricked into linking to arbitrary URLs."""
    if from_session and _UUID_RE.match(from_session):
        return f"/study/?session={from_session}"
    return "/study/"


VIEWER_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · Voice Tutor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,500;0,8..60,600;1,8..60,400&family=Inter:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {{
    --paper: #faf7f0; --paper-2: #f3eee0;
    --ink: #1c1a17; --ink-2: #44403a; --muted: #8a8478;
    --rule: #e6dfcc; --accent: #2d4a6b;
    --serif: "Source Serif 4", "Charter", Georgia, serif;
    --sans: "Inter", -apple-system, system-ui, sans-serif;
    --mono: ui-monospace, SFMono-Regular, Menlo, monospace;
  }}
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; background: var(--paper); color: var(--ink); font-family: var(--sans); line-height: 1.5; -webkit-font-smoothing: antialiased; }}
  .viewer {{ max-width: 720px; margin: 0 auto; padding: 28px 22px 96px; }}
  .back {{ font-family: var(--sans); font-size: 13px; color: var(--muted); text-decoration: none; display: inline-block; margin-bottom: 24px; }}
  .back:hover {{ color: var(--ink); }}
  .viewer__eyebrow {{ font-family: var(--mono); font-size: 11px; color: var(--muted); letter-spacing: 0.14em; text-transform: uppercase; margin: 0 0 8px; }}
  .viewer__title {{ font-family: var(--serif); font-size: 28px; font-weight: 500; margin: 0 0 6px; letter-spacing: -0.014em; }}
  .viewer__sub {{ font-size: 14px; color: var(--muted); margin: 0 0 32px; }}
  .md {{ font-family: var(--serif); font-size: 16px; line-height: 1.65; }}
  .md > *:first-child {{ margin-top: 0; }}
  .md h1 {{ font-size: 22px; font-weight: 600; margin: 32px 0 10px; }}
  .md h2 {{ font-size: 19px; font-weight: 600; margin: 28px 0 10px; }}
  .md h3 {{ font-size: 16px; margin: 22px 0 6px; font-style: italic; color: var(--ink-2); }}
  .md p {{ margin: 0 0 14px; }}
  .md ul, .md ol {{ padding-left: 22px; margin: 0 0 16px; }}
  .md li {{ margin-bottom: 6px; }}
  .md strong {{ font-weight: 600; }}
  .md em {{ font-style: italic; }}
  .md code {{ font-family: var(--mono); font-size: 0.92em; background: var(--paper-2); padding: 1px 5px; border-radius: 3px; }}
  .md table {{ border-collapse: collapse; margin: 16px 0; width: 100%; font-size: 13px; font-family: var(--sans); }}
  .md th, .md td {{ border: 1px solid var(--rule); padding: 6px 10px; text-align: left; }}
  .md th {{ background: var(--paper-2); font-weight: 500; font-size: 11px; letter-spacing: 0.04em; text-transform: uppercase; color: var(--muted); }}
  .md td:nth-child(n+4) {{ font-variant-numeric: tabular-nums; }}
  pre.viewer__pre {{ font-family: var(--mono); font-size: 12px; line-height: 1.6; white-space: pre-wrap; word-break: break-word; color: var(--ink-2); background: var(--paper-2); padding: 18px; border-radius: 6px; border: 1px solid var(--rule); }}
</style>
</head>
<body>
<main class="viewer">
  <a class="back" href="{back_href}">← Back</a>
  <p class="viewer__eyebrow">{eyebrow}</p>
  <h1 class="viewer__title">{title}</h1>
  <p class="viewer__sub">{subtitle}</p>
  <div id="content"></div>
</main>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<script>
  const raw = {raw_js};
  const mode = {mode_js};
  const el = document.getElementById('content');
  if (mode === 'markdown') {{
    el.className = 'md';
    el.innerHTML = marked.parse(raw);
  }} else {{
    const pre = document.createElement('pre');
    pre.className = 'viewer__pre';
    pre.textContent = raw;
    el.appendChild(pre);
  }}
</script>
</body>
</html>
"""


def _js_string(s: str) -> str:
    # json.dumps already escapes quotes, backslashes, control chars. The extra
    # replace closes the '</script>' breakout case (and any '</style>' etc.).
    return json.dumps(s).replace("</", "<\\/")


def _render_viewer(eyebrow: str, title: str, subtitle: str, content: str, mode: str, back_href: str = "/study/") -> str:
    return VIEWER_HTML.format(
        eyebrow=html_escape(eyebrow),
        title=html_escape(title),
        subtitle=html_escape(subtitle),
        back_href=html_escape(back_href, quote=True),
        raw_js=_js_string(content),
        mode_js=_js_string(mode),
    )


@app.get("/view/memory", include_in_schema=False)
async def view_memory(
    from_session: str | None = Query(None, alias="from"), user_id: str = Depends(require_user)
):
    content = session_state.load_memory(user_id) or "_(memory.md is empty.)_"
    return HTMLResponse(_render_viewer(
        "Persistent state",
        "memory.md",
        "Accumulating cross-session memory. One dated section per session, append-only.",
        content,
        "markdown",
        _back_href(from_session),
    ))


@app.get("/view/profile", include_in_schema=False)
async def view_profile(
    from_session: str | None = Query(None, alias="from"), user_id: str = Depends(require_user)
):
    content = session_state.load_profile(user_id) or "_(profile.md is empty.)_"
    return HTMLResponse(_render_viewer(
        "Persistent state",
        "profile.md",
        "Hand-maintained identity blurb. Loaded verbatim into the system prompt of every session.",
        content,
        "markdown",
        _back_href(from_session),
    ))


@app.get("/view/cost-log", include_in_schema=False)
async def view_cost_log(
    from_session: str | None = Query(None, alias="from"), user_id: str = Depends(require_user)
):
    if not sessions.can_view_machine_artifacts(user_id):
        raise HTTPException(status_code=404, detail="not found")
    content = COST_LOG_PATH.read_text() if COST_LOG_PATH.exists() else "_(cost-log.md not found.)_"
    return HTMLResponse(_render_viewer(
        "Persistent state",
        "cost-log.md",
        "Running tally across every session. LLM, STT, and TTS costs computed from ground-truth usage.",
        content,
        "markdown",
        _back_href(from_session),
    ))


@app.get("/view/sessions/{session_id}/prompt", include_in_schema=False)
async def view_prompt(session_id: str, user_id: str = Depends(require_user)):
    if not sessions.can_view_machine_artifacts(user_id):
        raise HTTPException(status_code=404, detail="prompt not found for this session")
    safe_id = Path(session_id).name
    if not sessions.session_belongs_to(user_id, safe_id):
        raise HTTPException(status_code=404, detail="prompt not found for this session")
    path = TRANSCRIPTS_DIR / user_id / f"{safe_id}.prompt.txt"
    if not path.exists():
        raise HTTPException(status_code=404, detail="prompt not found for this session")
    return HTMLResponse(_render_viewer(
        "Per-session artifact",
        "System prompt",
        f"The exact prompt sent to Claude Sonnet for session {safe_id[:8]}. Profile + memory + document + reminders, concatenated.",
        path.read_text(),
        "text",
        _back_href(safe_id),
    ))


@app.get("/view/sessions/{session_id}/analysis", include_in_schema=False)
async def view_analysis(session_id: str, user_id: str = Depends(require_user)):
    if not sessions.can_view_machine_artifacts(user_id):
        raise HTTPException(status_code=404, detail="analysis not found for this session")
    safe_id = Path(session_id).name
    if not sessions.session_belongs_to(user_id, safe_id):
        raise HTTPException(status_code=404, detail="analysis not found for this session")
    path = session_naming.find_analysis_path(SESSION_ANALYSES_DIR, user_id, safe_id)
    if path is None:
        raise HTTPException(status_code=404, detail="analysis not found for this session")
    return HTMLResponse(_render_viewer(
        "Per-session artifact",
        "Session analysis",
        f"Haiku-generated post-session analysis for session {safe_id[:8]} — topics, tool usage, interaction quality.",
        path.read_text(),
        "markdown",
        _back_href(safe_id),
    ))
