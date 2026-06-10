"""Bird Identifier Agent — the ONLY ML component.

Loads your trained Keras model (resnet50v2_fold{K}.keras) and label_map.json,
then serves clip-level predictions by averaging softmax probabilities across
all chunk tensors — exactly the logic from train.ipynb Step 11.

In build step 3 you do NOT need this running; the Index Agent uses StubIdentifier.
This is the real implementation you switch to in a later step. It is written so
the contract (SpectrogramTensors in -> ClassificationResult out) matches the
stub the orchestrator already expects.

Inputs:
  MODEL_PATH      path to the .keras file (mounted from Drive/bird_models)
  LABEL_MAP_PATH  path to label_map.json ({"0": "Turdus_fuscater", ...})
"""

from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException

from contracts import Candidate, ClassificationResult, SpectrogramTensors
from app.model import BirdSongClassifier


_classifier = None
WEIGHTS_PATH   = os.environ.get("WEIGHTS_PATH",   "/models/fold2_best_weights.h5")
LABEL_MAP_PATH = os.environ.get("LABEL_MAP_PATH", "/models/label_map.json")

app = FastAPI(title="CBSI Bird Identifier Agent", version="0.1.0")


def _load() -> None:
    global _classifier
    if _classifier is not None:
        return
    _classifier = BirdSongClassifier(
        weights_path=WEIGHTS_PATH,
        label_map_path=LABEL_MAP_PATH
    )



@app.on_event("startup")
async def _startup() -> None:
    if Path(WEIGHTS_PATH).exists():
        _load()
        _classifier.model.predict(
            np.zeros((1, 128, 188, 1), dtype="float32"), verbose=0
        )


        



def _decode_tensors(t: SpectrogramTensors) -> np.ndarray:
    """Recover the (K, 128, 188) float32 array from the transport payload."""
    if not t.npy_b64:
        raise HTTPException(status_code=400, detail="missing tensor payload")
    arr = np.load(io.BytesIO(base64.b64decode(t.npy_b64)))
    if arr.ndim != 3 or arr.shape[1:] != (t.height, t.width):
        raise HTTPException(status_code=400, detail=f"bad tensor shape {arr.shape}")
    return arr.astype(np.float32)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model_loaded": _classifier is not None}
    


@app.post("/classify", response_model=ClassificationResult)
async def classify(tensors: SpectrogramTensors) -> ClassificationResult:
    _load()
    arr = _decode_tensors(tensors)       # (K, 128, 188) — unchanged
    import json
    result = json.loads(_classifier.predict_top3(arr))
    top    = result[0]
    label  = top["species"].replace(" ", "_")
    alts   = [
        Candidate(
            label=r["species"].replace(" ", "_"),
            display_name=r["species"],
            confidence=r["confidence"]
        )
        for r in result[1:]
    ]
    return ClassificationResult(
        label=label,
        display_name=top["species"],
        confidence=top["confidence"],
        alternatives=alts,
        n_chunks=int(arr.shape[0]),
        model_version=Path(WEIGHTS_PATH).stem,
    )


