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

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/resnet50v2_fold0.keras")
LABEL_MAP_PATH = os.environ.get("LABEL_MAP_PATH", "/models/label_map.json")

app = FastAPI(title="CBSI Bird Identifier Agent", version="0.1.0")

_model = None
_label_map: dict[int, str] = {}


def _load() -> None:
    """Lazy-load the Keras model + label map on first request / startup."""
    global _model, _label_map
    if _model is not None:
        return
    import tensorflow as tf  # heavy import; only here

    _model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABEL_MAP_PATH, encoding="utf-8") as fh:
        _label_map = {int(k): v for k, v in json.load(fh).items()}


def _decode_tensors(t: SpectrogramTensors) -> np.ndarray:
    """Recover the (K, 128, 188) float32 array from the transport payload."""
    if not t.npy_b64:
        raise HTTPException(status_code=400, detail="missing tensor payload")
    arr = np.load(io.BytesIO(base64.b64decode(t.npy_b64)))
    if arr.ndim != 3 or arr.shape[1:] != (t.height, t.width):
        raise HTTPException(status_code=400, detail=f"bad tensor shape {arr.shape}")
    return arr.astype(np.float32)


@app.on_event("startup")
async def _startup() -> None:
    # Warm the model so the first real request isn't slow. Tolerate absence in
    # environments where the model file isn't mounted yet.
    if Path(MODEL_PATH).exists():
        _load()
        # one dummy inference to build the graph
        import numpy as _np

        _model.predict(_np.zeros((1, 128, 188, 1), dtype="float32"), verbose=0)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/classify", response_model=ClassificationResult)
async def classify(tensors: SpectrogramTensors) -> ClassificationResult:
    _load()
    arr = _decode_tensors(tensors)            # (K, 128, 188)
    x = np.expand_dims(arr, axis=-1)          # (K, 128, 188, 1)
    chunk_probs = _model.predict(x, verbose=0)  # (K, N_CLASSES)
    avg = chunk_probs.mean(axis=0)            # average across chunks

    order = np.argsort(avg)[::-1]
    top = int(order[0])
    top_label = _label_map[top]
    alts = [
        Candidate(
            label=_label_map[int(i)],
            display_name=_label_map[int(i)].replace("_", " "),
            confidence=float(avg[int(i)]),
        )
        for i in order[1:4]
    ]
    return ClassificationResult(
        label=top_label,
        display_name=top_label.replace("_", " "),
        confidence=float(avg[top]),
        alternatives=alts,
        n_chunks=int(arr.shape[0]),
        model_version=Path(MODEL_PATH).stem,
    )
