"""Tests for the Bird Index Agent skeleton (stub mode).

Run from services/bird-index-agent:  pytest -q
Proves orchestration + graceful degradation work before any real agent exists.
"""

from __future__ import annotations

import asyncio

from app.clients import StubCard, StubEnrichment, StubIdentifier, StubRecorder
from app.orchestrator import IdentificationError, Orchestrator
from contracts import UnifiedResponse


def _orch() -> Orchestrator:
    return Orchestrator(
        recorder=StubRecorder(),
        identifier=StubIdentifier(),
        cards=StubCard(),
        enrichment=StubEnrichment(),
    )


def test_happy_path_returns_unified_response():
    resp = asyncio.run(_orch().identify(b"fake-audio"))
    assert isinstance(resp, UnifiedResponse)
    assert resp.identification.label in {"Turdus_fuscater", "Coeligena_prunellei"}
    assert resp.identification.display_name == resp.identification.label.replace("_", " ")
    assert resp.card is not None
    assert resp.enrichment is not None
    assert resp.degraded == []


def test_card_degrades_gracefully():
    class BrokenCards:
        async def lookup(self, label):
            raise RuntimeError("mongo down")

    orch = Orchestrator(
        recorder=StubRecorder(),
        identifier=StubIdentifier(),
        cards=BrokenCards(),
        enrichment=StubEnrichment(),
    )
    resp = asyncio.run(orch.identify(b"clip"))
    assert resp.card is None
    assert "card" in resp.degraded
    assert resp.identification is not None  # still succeeds


def test_no_surviving_chunks_is_required_failure():
    class EmptyRecorder:
        async def preprocess(self, job_id, audio):
            from contracts import SpectrogramTensors

            return SpectrogramTensors(job_id=job_id, n_chunks=0)

    orch = Orchestrator(
        recorder=EmptyRecorder(),
        identifier=StubIdentifier(),
        cards=StubCard(),
        enrichment=StubEnrichment(),
    )
    try:
        asyncio.run(orch.identify(b"silence"))
        assert False, "expected IdentificationError"
    except IdentificationError:
        pass
