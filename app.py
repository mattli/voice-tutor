"""FastAPI app for voice-tutor.

Owns the HTTP surface so we can add study-mode routes alongside the WebRTC
offer flow. Replaces pipecat.runner.run.main — that helper hides the FastAPI
app inside its CLI entry point with no extension hook, so we replicate the
~30 lines of WebRTC plumbing it would have set up.

The voice pipeline lives in bot.py; this module only handles HTTP.
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pipecat.runner.types import SmallWebRTCRunnerArguments
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.request_handler import (
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
