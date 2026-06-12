"""Bird Index Agent — FastAPI application (the only service the frontend calls).

Build step 3 runs fully on stubs, so POST /api/v1/identify works immediately.
Wiring is chosen at startup by CBSI_USE_STUBS.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from contracts import ErrorResponse, UnifiedResponse

from .clients import (
    HttpEnrichment,
    HttpIdentifier,
    HttpRecorder,
    StubCard,
    StubEnrichment,
    StubIdentifier,
    StubRecorder,
)
from .config import settings
from .orchestrator import IdentificationError, Orchestrator


def _build_orchestrator(http: Optional[httpx.AsyncClient]) -> Orchestrator:
    if settings.use_stubs:
        return Orchestrator(
            recorder=StubRecorder(),
            identifier=StubIdentifier(),
            cards=StubCard(),
            enrichment=StubEnrichment(),
        )
    assert http is not None
    from .db import MongoCards

    return Orchestrator(
        recorder=HttpRecorder(http),
        identifier=HttpIdentifier(http),
        cards=MongoCards(),
        enrichment=HttpEnrichment(http),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    http = None if settings.use_stubs else httpx.AsyncClient()
    app.state.http = http
    app.state.orchestrator = _build_orchestrator(http)
    try:
        yield
    finally:
        if http is not None:
            await http.aclose()


app = FastAPI(
    title="CBSI Bird Index Agent",
    version="0.1.0",
    description="Central orchestrator for the Colombian Bird Song Identifier.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "stubs": settings.use_stubs}


@app.post(
    "/api/v1/identify",
    response_model=UnifiedResponse,
    responses={502: {"model": ErrorResponse}, 413: {"model": ErrorResponse}},
)
async def identify(audio: UploadFile = File(...)) -> UnifiedResponse:
    raw = await audio.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty upload")
    if len(raw) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="audio too large")

    orchestrator: Orchestrator = app.state.orchestrator
    try:
        return await orchestrator.identify(raw)
    except IdentificationError as exc:
        raise HTTPException(status_code=502, detail=f"identification failed: {exc}")
