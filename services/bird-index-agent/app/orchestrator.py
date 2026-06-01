"""Orchestration logic for the Bird Index Agent.

Flow (adapted to this project):
  1. preprocess audio -> spectrogram tensors      (required)
  2. classify tensors -> ClassificationResult      (required; your Keras model)
  3. card lookup + enrichment                       (independent -> concurrent)
  4. aggregate -> UnifiedResponse, marking degraded optional steps

Depends only on the client Protocols, so stubs and real agents are
interchangeable without touching this file.
"""

from __future__ import annotations

import asyncio
import uuid

from contracts import (
    ClassificationResult,
    EnrichRequest,
    JobStatus,
    UnifiedResponse,
)

from .clients import CardClient, EnrichmentClient, IdentifierClient, RecorderClient
from .config import settings


class IdentificationError(RuntimeError):
    """A REQUIRED step (preprocess or classify) failed."""


class Orchestrator:
    def __init__(
        self,
        recorder: RecorderClient,
        identifier: IdentifierClient,
        cards: CardClient,
        enrichment: EnrichmentClient,
    ) -> None:
        self._recorder = recorder
        self._identifier = identifier
        self._cards = cards
        self._enrichment = enrichment

    async def identify(self, audio: bytes) -> UnifiedResponse:
        job_id = str(uuid.uuid4())
        degraded: list[str] = []

        # --- required path -------------------------------------------------- #
        try:
            tensors = await self._recorder.preprocess(job_id, audio)
            if tensors.n_chunks == 0:
                raise IdentificationError("no audio survived silence rejection")
            cls: ClassificationResult = await self._identifier.classify(tensors)
        except IdentificationError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise IdentificationError(str(exc)) from exc

        # --- optional path (concurrent) ------------------------------------ #
        card, enrichment = await asyncio.gather(
            self._safe_card(cls.label),
            self._safe_enrich(cls),
        )
        if card is None:
            degraded.append("card")
        if enrichment is None:
            degraded.append("enrichment")

        return UnifiedResponse(
            job_id=job_id,
            status=JobStatus.done,
            identification=cls,
            card=card,
            enrichment=enrichment,
            degraded=degraded,
        )

    async def _safe_card(self, label: str):
        try:
            return await self._cards.lookup(label)
        except Exception:  # noqa: BLE001
            return None

    async def _safe_enrich(self, cls: ClassificationResult):
        try:
            req = EnrichRequest(
                label=cls.label,
                scientific_name=cls.display_name,
                confidence=cls.confidence,
            )
            return await asyncio.wait_for(
                self._enrichment.enrich(req), timeout=settings.enrich_timeout_s
            )
        except (Exception, asyncio.TimeoutError):  # noqa: BLE001
            return None
