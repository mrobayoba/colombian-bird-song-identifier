"""Downstream clients for the Bird Index Agent.

Protocols + two implementations each:
  * Stub*  -> canned data, no network (build step 3).
  * Http*  -> call the real agents (filled in later steps). The Identifier's
              real path reflects your Keras model: send tensors, get a
              chunk-averaged clip-level result back.

The stubs use REAL Colombian species labels in your underscored label_map
style so the wired-up response looks plausible from day one.
"""

from __future__ import annotations

import hashlib
from typing import Optional, Protocol

import httpx

from contracts import (
    BirdCard,
    Candidate,
    ClassificationResult,
    ConservationStatus,
    EnrichRequest,
    EnrichmentResult,
    SpectrogramTensors,
    Taxonomy,
)

from .config import settings


# --------------------------------------------------------------------------- #
# Protocols
# --------------------------------------------------------------------------- #
class RecorderClient(Protocol):
    async def preprocess(self, job_id: str, audio: bytes) -> SpectrogramTensors: ...


class IdentifierClient(Protocol):
    async def classify(self, tensors: SpectrogramTensors) -> ClassificationResult: ...


class CardClient(Protocol):
    async def lookup(self, label: str) -> Optional[BirdCard]: ...


class EnrichmentClient(Protocol):
    async def enrich(self, req: EnrichRequest) -> EnrichmentResult: ...


# --------------------------------------------------------------------------- #
# Canned catalogue (real Colombian species, underscored label style)
# --------------------------------------------------------------------------- #
_CANNED: dict[str, dict] = {
    "Turdus_fuscater": {
        "scientific_name": "Turdus fuscater",
        "english_name": "Great Thrush",
        "description": "[STUB] Zorzal grande andino, muy común en zonas urbanas y rurales.",
        "conservation": ("LC", "Preocupación menor", "Least Concern"),
        "family": "Turdidae",
        "order": "Passeriformes",
    },
    "Coeligena_prunellei": {
        "scientific_name": "Coeligena prunellei",
        "english_name": "Black Inca",
        "description": "[STUB] Colibrí endémico de Colombia, asociado a bosques andinos.",
        "conservation": ("VU", "Vulnerable", "Vulnerable"),
        "family": "Trochilidae",
        "order": "Apodiformes",
    },
}


def _display(label: str) -> str:
    return label.replace("_", " ")


# --------------------------------------------------------------------------- #
# STUBS (build step 3)
# --------------------------------------------------------------------------- #
class StubRecorder:
    async def preprocess(self, job_id: str, audio: bytes) -> SpectrogramTensors:
        # Pretend a few 3 s chunks survived silence rejection.
        return SpectrogramTensors(job_id=job_id, n_chunks=4, height=128, width=188)


class StubIdentifier:
    async def classify(self, tensors: SpectrogramTensors) -> ClassificationResult:
        labels = list(_CANNED)
        i = int(hashlib.sha256(tensors.job_id.encode()).hexdigest(), 16) % len(labels)
        top, other = labels[i], labels[(i + 1) % len(labels)]
        return ClassificationResult(
            label=top,
            display_name=_display(top),
            confidence=0.92,
            alternatives=[
                Candidate(label=other, display_name=_display(other), confidence=0.05)
            ],
            n_chunks=tensors.n_chunks,
            model_version="stub-0",
        )


class StubCard:
    async def lookup(self, label: str) -> Optional[BirdCard]:
        d = _CANNED.get(label)
        if not d:
            return None
        code, es, en = d["conservation"]
        return BirdCard(
            label=label,
            scientific_name=d["scientific_name"],
            english_name=d["english_name"],
            description=d["description"],
            description_source="stub",
            conservation_status=ConservationStatus(code=code, label_es=es, label_en=en),
            taxonomy=Taxonomy(family=d["family"], order=d["order"]),
        )


class StubEnrichment:
    async def enrich(self, req: EnrichRequest) -> EnrichmentResult:
        return EnrichmentResult(
            summary_es=f"[STUB] {req.scientific_name} es un ave registrada en Colombia.",
            semantic_tags=["stub", "placeholder"],
            alternatives=[],
            read_more_target=req.scientific_name,
            sources=[f"stub:{req.label}"],
            model="stub-0",
        )


# --------------------------------------------------------------------------- #
# HTTP implementations (later steps)
# --------------------------------------------------------------------------- #
class HttpRecorder:
    def __init__(self, client: httpx.AsyncClient) -> None:
        self._http = client

    async def preprocess(self, job_id: str, audio: bytes) -> SpectrogramTensors:
        r = await self._http.post(
            f"{settings.recorder_url}/preprocess",
            params={"job_id": job_id},
            content=audio,
            timeout=settings.downstream_timeout_s,
        )
        r.raise_for_status()
        return SpectrogramTensors(**r.json())


class HttpIdentifier:
    def __init__(self, client: httpx.AsyncClient) -> None:
        self._http = client

    async def classify(self, tensors: SpectrogramTensors) -> ClassificationResult:
        # Real Keras agent receives tensors, runs model.predict, averages over
        # chunks, returns a clip-level ClassificationResult.
        r = await self._http.post(
            f"{settings.identifier_url}/classify",
            json=tensors.model_dump(),
            timeout=settings.downstream_timeout_s,
        )
        r.raise_for_status()
        return ClassificationResult(**r.json())


class HttpEnrichment:
    def __init__(self, client: httpx.AsyncClient) -> None:
        self._http = client

    async def enrich(self, req: EnrichRequest) -> EnrichmentResult:
        r = await self._http.post(
            f"{settings.mcp_url}/enrich",
            json=req.model_dump(),
            timeout=settings.enrich_timeout_s,
        )
        r.raise_for_status()
        return EnrichmentResult(**r.json())
