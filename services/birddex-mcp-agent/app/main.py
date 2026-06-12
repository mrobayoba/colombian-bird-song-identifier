"""BirdDex MCP Agent — LLM/MCP semantic enrichment.

Invoked ONLY by the Bird Index Agent. Given a species, it produces a grounded
Spanish summary, behavior notes, and semantic tags, using the species' card as
trusted context (anti-hallucination: the LLM may only use supplied facts).

This is a skeleton: the /enrich endpoint returns a deterministic, card-grounded
response and marks where the real MCP+LLM call goes. In build step 3 the Index
Agent uses StubEnrichment and never calls this. Wire it in a later step.
"""

from __future__ import annotations

import os

from fastapi import FastAPI

from contracts import EnrichRequest, EnrichmentResult

app = FastAPI(title="CBSI BirdDex MCP Agent", version="0.1.0")

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://mongo:27017")
MONGO_DB = os.environ.get("MONGO_DB", "cbsi")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/enrich", response_model=EnrichmentResult)
async def enrich(req: EnrichRequest) -> EnrichmentResult:
    # ---- where the real MCP/LLM workflow goes -------------------------------
    # 1. Fetch the trusted card from Mongo (get_bird_card MCP tool).
    # 2. Pass ONLY those fields to the LLM with a system prompt that forbids
    #    inventing facts; require strict JSON + a `sources` provenance list.
    # 3. Validate the JSON against EnrichmentResult; on failure, fall back to a
    #    template summary built directly from the card (what we do below).
    # 4. Cache the validated result in Mongo keyed by label to avoid re-spend.
    # -------------------------------------------------------------------------
    summary = (
        f"{req.scientific_name} es una especie de ave registrada en Colombia. "
        "Resumen generado a partir de la ficha ornitológica verificada."
    )
    return EnrichmentResult(
        summary_es=summary,
        behavior_notes=None,
        semantic_tags=["colombia", "ave"],
        sources=[f"card:{req.label}"],
        model="template-fallback",
    )
