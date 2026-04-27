"""FastAPI app for voice-tutor.

Owns the HTTP surface so we can add study-mode routes alongside the WebRTC
offer flow. Replaces pipecat.runner.run.main — that helper hides the FastAPI
app inside its CLI entry point with no extension hook, so we replicate the
~30 lines of WebRTC plumbing it would have set up.

The voice pipeline lives in bot.py; this module only handles HTTP.
"""

import os
import uuid
from contextlib import asynccontextmanager
from http import HTTPMethod
from pathlib import Path
from typing import Any, Dict

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
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

app.mount("/client", SmallWebRTCPrebuiltUI)


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


# RTVI client (used by the pipecat prebuilt UI at /client/) bootstraps via
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
    return documents.list_documents()


STUDY_HTML = Path(__file__).parent / "static" / "study.html"


@app.get("/study/", include_in_schema=False)
@app.get("/study", include_in_schema=False)
async def study_page():
    return FileResponse(STUDY_HTML, media_type="text/html")


ARTIFACTS_DIR = Path.home() / ".voice-tutor" / "artifacts"


@app.get("/api/sessions/{session_id}/artifact")
async def get_artifact(session_id: str):
    safe_id = Path(session_id).name  # belt and suspenders against path traversal
    path = ARTIFACTS_DIR / f"{safe_id}.md"
    if not path.exists():
        raise HTTPException(status_code=404, detail="artifact not ready or not found")
    return FileResponse(path, media_type="text/markdown")
