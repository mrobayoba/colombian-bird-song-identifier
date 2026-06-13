"""Audio Recorder Agent — preprocessing only (no ML).

Decodes uploaded audio to mono 32 kHz, runs the shared preprocessing pipeline
(bandpass -> 3 s chunks -> silence rejection -> mel -> fixed-range tensor), and
returns the stacked (K, 128, 188) tensors as a base64 npy payload for the
Identifier. Reuses shared/preprocessing/audio.py so it is guaranteed to match
training.

Build step 3 uses StubRecorder instead; this is the real implementation for a
later step.
"""

from __future__ import annotations

import base64
import io

import numpy as np
from fastapi import FastAPI, HTTPException, Request

from contracts import SpectrogramTensors
from shared.preprocessing import audio as P

app = FastAPI(title="CBSI Audio Recorder Agent", version="0.1.0")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "audio_libs": P.libs_available()}


@app.post("/preprocess", response_model=SpectrogramTensors)
async def preprocess(request: Request, job_id: str) -> SpectrogramTensors:
    if not P.libs_available():
        raise HTTPException(status_code=503, detail="audio libraries unavailable")

    raw = await request.body()
    if not raw:
        raise HTTPException(status_code=400, detail="empty audio")

    # Decode to mono 32 kHz via librosa (handles mp3/wav/ogg/m4a).
    import librosa

    audio, _ = librosa.load(io.BytesIO(raw), sr=P.SR, mono=True)
    tensors = P.audio_bytes_to_tensors(audio)  # (K, 128, 188)

    buf = io.BytesIO()
    np.save(buf, tensors.astype(np.float32))
    return SpectrogramTensors(
        job_id=job_id,
        n_chunks=int(tensors.shape[0]),
        height=P.INPUT_HEIGHT,
        width=P.INPUT_WIDTH,
        npy_b64=base64.b64encode(buf.getvalue()).decode("ascii"),
    )
