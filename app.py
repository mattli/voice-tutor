"""FastAPI app for voice-tutor.

Owns the HTTP surface so we can add study-mode routes alongside the WebRTC
offer flow. Replaces pipecat.runner.run.main — that helper hides the FastAPI
app inside its CLI entry point with no extension hook, so we replicate the
~30 lines of WebRTC plumbing it would have set up.

The voice pipeline lives in bot.py; this module only handles HTTP.
"""

import json
import os
import uuid
from contextlib import asynccontextmanager
from html import escape as html_escape
from http import HTTPMethod
from pathlib import Path
from typing import Any, Dict

import re

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from pipecat.runner.types import SmallWebRTCRunnerArguments
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.request_handler import (
    IceCandidate,
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)
from pipecat_ai_small_webrtc_prebuilt.frontend import SmallWebRTCPrebuiltUI

import bot
import documents
import sessions

HOST = os.getenv("VOICE_TUTOR_HOST", "0.0.0.0")
small_webrtc_handler = SmallWebRTCRequestHandler(esp32_mode=False, host=HOST)


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


@app.post("/api/offer")
async def offer(request: SmallWebRTCRequest, background_tasks: BackgroundTasks):
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
async def rtvi_start(request: Request):
    try:
        request_data = await request.json()
    except Exception:
        request_data = {}
    session_id = str(uuid.uuid4())
    active_sessions[session_id] = request_data.get("body") or {}
    result: Dict[str, Any] = {"sessionId": session_id}
    if request_data.get("enableDefaultIceServers"):
        result["iceConfig"] = {
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }
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
                return await offer(webrtc_request, background_tasks)
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
async def upload_document(file: UploadFile):
    try:
        raw = await file.read()
        return documents.save_upload(file.filename or "untitled", raw)
    except documents.UploadError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)


@app.get("/api/documents")
async def list_documents_route():
    return await documents.list_documents()


STUDY_HTML = Path(__file__).parent / "static" / "study.html"


@app.get("/study/", include_in_schema=False)
@app.get("/study", include_in_schema=False)
async def study_page():
    return FileResponse(STUDY_HTML, media_type="text/html")


VOICE_TUTOR_DIR = Path.home() / ".voice-tutor"
ARTIFACTS_DIR = VOICE_TUTOR_DIR / "artifacts"
TRANSCRIPTS_DIR = VOICE_TUTOR_DIR / "transcripts"
MEMORY_PATH = VOICE_TUTOR_DIR / "memory.md"
PROFILE_PATH = VOICE_TUTOR_DIR / "profile.md"
SESSION_ANALYSES_DIR = Path.home() / "second-brain" / "products" / "voice-tutor" / "session-analyses"
COST_LOG_PATH = Path.home() / "second-brain" / "products" / "voice-tutor" / "validation" / "cost-log.md"


@app.get("/api/sessions/{session_id}/artifact")
async def get_artifact(session_id: str):
    safe_id = Path(session_id).name  # belt and suspenders against path traversal
    path = ARTIFACTS_DIR / f"{safe_id}.md"
    if not path.exists():
        raise HTTPException(status_code=404, detail="artifact not ready or not found")
    return FileResponse(path, media_type="text/markdown")


@app.get("/api/sessions/latest")
async def get_latest_session():
    """Most recent study session, used by the picker-screen 'View last session'
    link. Iterates cost-log.jsonl in reverse since study session rows are
    appended at session end and carry the UUID + document_id we need."""
    jsonl_path = Path.home() / "second-brain" / "products" / "voice-tutor" / "validation" / "cost-log.jsonl"
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
        doc_id = entry.get("document_id")
        loaded = documents.load_document(doc_id) if doc_id else None
        return {
            "session_id": entry["session_id"],
            "document_id": doc_id,
            "document_title": loaded[0] if loaded else None,
        }
    raise HTTPException(status_code=404, detail="no study session yet")


@app.get("/api/sessions")
async def list_sessions():
    """All completed study sessions, newest first, for the /study/ history
    surface. Thin wrapper — all listing logic lives in the pure sessions.py
    helper (Pipecat-free, hermetically tested)."""
    return sessions.list_study_sessions()


COST_LOG_JSONL_PATH = Path.home() / "second-brain" / "products" / "voice-tutor" / "validation" / "cost-log.jsonl"


def _lookup_session_doc(session_id: str) -> dict | None:
    """Look up the document_id/title for a session from cost-log.jsonl.
    Returns None if not found."""
    if not COST_LOG_JSONL_PATH.exists():
        return None
    with COST_LOG_JSONL_PATH.open() as f:
        for line in f:
            try:
                entry = json.loads(line)
            except Exception:
                continue
            if entry.get("kind") == "session" and entry.get("session_id") == session_id:
                doc_id = entry.get("document_id")
                loaded = documents.load_document(doc_id) if doc_id else None
                return {
                    "document_id": doc_id,
                    "document_title": loaded[0] if loaded else None,
                }
    return None


@app.get("/api/sessions/{session_id}/telemetry")
async def get_telemetry(session_id: str):
    """Composite endpoint for the /study/ ended view. Each field is null until
    that artifact lands; the frontend polls and renders pieces progressively.

    Includes the recap so the frontend polls a single URL rather than juggling
    `/artifact` + `/telemetry` independently."""
    safe_id = Path(session_id).name
    artifact_path = ARTIFACTS_DIR / f"{safe_id}.md"
    usage_path = TRANSCRIPTS_DIR / f"{safe_id}.usage.json"
    summary_path = TRANSCRIPTS_DIR / f"{safe_id}.summary.md"
    analysis_path = SESSION_ANALYSES_DIR / f"session-analysis-{safe_id}.md"
    prompt_path = TRANSCRIPTS_DIR / f"{safe_id}.prompt.txt"
    doc_info = _lookup_session_doc(safe_id) or {"document_id": None, "document_title": None}
    return {
        "recap": artifact_path.read_text() if artifact_path.exists() else None,
        "cost": json.loads(usage_path.read_text()) if usage_path.exists() else None,
        "memory_append": summary_path.read_text() if summary_path.exists() else None,
        "analysis": analysis_path.read_text() if analysis_path.exists() else None,
        "has_prompt": prompt_path.exists(),
        # The frontend uses this to decide whether to wait for memory_append /
        # analysis on shorter sessions. Mirrors bot.py's MIN_SUMMARY_DURATION_SEC.
        "min_summary_sec": bot.MIN_SUMMARY_DURATION_SEC,
        # Document context for page-load restoration when URL has ?session=<id>.
        "document_id": doc_info["document_id"],
        "document_title": doc_info["document_title"],
    }


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
async def view_memory(from_session: str | None = Query(None, alias="from")):
    content = MEMORY_PATH.read_text() if MEMORY_PATH.exists() else "_(memory.md is empty.)_"
    return HTMLResponse(_render_viewer(
        "Persistent state",
        "memory.md",
        "Accumulating cross-session memory. One dated section per session, append-only.",
        content,
        "markdown",
        _back_href(from_session),
    ))


@app.get("/view/profile", include_in_schema=False)
async def view_profile(from_session: str | None = Query(None, alias="from")):
    content = PROFILE_PATH.read_text() if PROFILE_PATH.exists() else "_(profile.md is empty.)_"
    return HTMLResponse(_render_viewer(
        "Persistent state",
        "profile.md",
        "Hand-maintained identity blurb. Loaded verbatim into the system prompt of every session.",
        content,
        "markdown",
        _back_href(from_session),
    ))


@app.get("/view/cost-log", include_in_schema=False)
async def view_cost_log(from_session: str | None = Query(None, alias="from")):
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
async def view_prompt(session_id: str):
    safe_id = Path(session_id).name
    path = TRANSCRIPTS_DIR / f"{safe_id}.prompt.txt"
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
async def view_analysis(session_id: str):
    safe_id = Path(session_id).name
    path = SESSION_ANALYSES_DIR / f"session-analysis-{safe_id}.md"
    if not path.exists():
        raise HTTPException(status_code=404, detail="analysis not found for this session")
    return HTMLResponse(_render_viewer(
        "Per-session artifact",
        "Session analysis",
        f"Haiku-generated post-session analysis for session {safe_id[:8]} — topics, tool usage, interaction quality.",
        path.read_text(),
        "markdown",
        _back_href(safe_id),
    ))
