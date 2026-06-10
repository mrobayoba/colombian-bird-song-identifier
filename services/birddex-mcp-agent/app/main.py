"""BirdDex MCP Agent — LLM enrichment via Google Gemini.

Invoked ONLY by the Bird Index Agent. Given a species, it returns a grounded
Spanish summary, behavior notes, and semantic tags.

Anti-hallucination design (defense in depth):
  1. The LLM is given ONLY the trusted card fields fetched from MongoDB. It is
     never asked to recall facts from its own training.
  2. Output is forced to strict JSON via response_schema, so it always parses
     into EnrichmentResult.
  3. The system instruction forbids inventing data not present in the card.
  4. `sources` records provenance (the card label + the model id).
  5. If the API key is missing OR the call fails OR the card is absent, we fall
     back to a deterministic template. Enrichment is optional downstream, so a
     failure here never breaks identification.
  6. Validated results are cached in Mongo (enrichment_cache) keyed by label,
     so we never pay for the same species twice.

Environment variables:
  GEMINI_API_KEY   your Google Gemini API key (REQUIRED for real enrichment)
  GEMINI_MODEL     model id (default: gemini-2.5-flash)
  MONGO_URI        mongodb connection (default: mongodb://mongo:27017)
  MONGO_DB         database name (default: cbsi)
"""

from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from contracts import EnrichRequest, EnrichmentResult

app = FastAPI(title="CBSI BirdDex MCP Agent", version="0.1.0")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://mongo:27017")
MONGO_DB = os.environ.get("MONGO_DB", "cbsi")


# --------------------------------------------------------------------------- #
# Schema the LLM must return (forces structured, parseable output)
# --------------------------------------------------------------------------- #
class _LLMEnrichment(BaseModel):
    summary_es: str
    behavior_notes: Optional[str] = None
    semantic_tags: list[str] = []


# --------------------------------------------------------------------------- #
# Lazy singletons (Mongo + Gemini client created once, on first use)
# --------------------------------------------------------------------------- #
_mongo = None
_gemini = None


def _db():
    global _mongo
    if _mongo is None:
        from motor.motor_asyncio import AsyncIOMotorClient

        _mongo = AsyncIOMotorClient(MONGO_URI)[MONGO_DB]
    return _mongo


def _gemini_client():
    global _gemini
    if _gemini is None and GEMINI_API_KEY:
        from google import genai

        _gemini = genai.Client(api_key=GEMINI_API_KEY)
    return _gemini


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
async def _fetch_card(label: str) -> Optional[dict]:
    """The trusted grounding data. Everything the LLM may state comes from here."""
    try:
        return await _db().cards.find_one({"label": label}, {"_id": 0})
    except Exception:
        return None


async def _cached(label: str) -> Optional[EnrichmentResult]:
    try:
        doc = await _db().enrichment_cache.find_one({"label": label}, {"_id": 0})
        return EnrichmentResult(**doc["result"]) if doc else None
    except Exception:
        return None


async def _save_cache(label: str, result: EnrichmentResult) -> None:
    try:
        await _db().enrichment_cache.update_one(
            {"label": label},
            {"$set": {"label": label, "result": result.model_dump()}},
            upsert=True,
        )
    except Exception:
        pass  # caching is best-effort; never block the response


def _template(req: EnrichRequest, card: Optional[dict]) -> EnrichmentResult:
    """Deterministic fallback used when the LLM is unavailable."""
    family = (card or {}).get("taxonomy", {}).get("family")
    extra = f" Pertenece a la familia {family}." if family else ""
    return EnrichmentResult(
        summary_es=(
            f"{req.scientific_name} es una especie de ave registrada en Colombia.{extra} "
            "Resumen generado a partir de la ficha ornitologica verificada."
        ),
        behavior_notes=None,
        semantic_tags=["colombia", "ave"],
        sources=[f"card:{req.label}"],
        model="template-fallback",
    )


def _build_prompt(card: dict) -> str:
    """Give the model ONLY trusted fields; instruct it to use nothing else."""
    import json

    trusted = {
        "scientific_name": card.get("scientific_name"),
        "english_name": card.get("english_name"),
        "family": card.get("taxonomy", {}).get("family"),
        "order": card.get("taxonomy", {}).get("order"),
        "conservation_status": card.get("conservation_status", {}).get("code"),
        "description": card.get("description"),
    }
    return (
        "Eres un ornitologo. Con base UNICAMENTE en los datos verificados de la "
        "siguiente ficha, escribe un resumen breve en espanol (2-3 frases), notas "
        "de comportamiento si la ficha las respalda, y de 3 a 6 etiquetas "
        "semanticas. No inventes datos que no esten en la ficha. Si un campo "
        "falta, no lo menciones.\n\n"
        f"FICHA VERIFICADA:\n{json.dumps(trusted, ensure_ascii=False, indent=2)}"
    )


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "gemini_configured": bool(GEMINI_API_KEY)}


@app.post("/enrich", response_model=EnrichmentResult)
async def enrich(req: EnrichRequest) -> EnrichmentResult:
    # 1) Return a previously validated result if we have one.
    cached = await _cached(req.label)
    if cached:
        return cached

    # 2) Fetch the trusted grounding card.
    card = await _fetch_card(req.label)

    # 3) No key or no card -> safe template fallback.
    client = _gemini_client()
    if client is None or card is None:
        return _template(req, card)

    # 4) Ask Gemini, constrained to JSON grounded in the card.
    try:
        from google.genai import types

        resp = await _call_gemini(client, types, _build_prompt(card))
        parsed = _LLMEnrichment.model_validate_json(resp)
        result = EnrichmentResult(
            summary_es=parsed.summary_es,
            behavior_notes=parsed.behavior_notes,
            semantic_tags=parsed.semantic_tags,
            sources=[f"card:{req.label}", f"gemini:{GEMINI_MODEL}"],
            model=GEMINI_MODEL,
        )
    except Exception:
        # Any failure (network, quota, bad JSON) -> template, never an error.
        return _template(req, card)

    # 5) Cache and return.
    await _save_cache(req.label, result)
    return result


async def _call_gemini(client, types, prompt: str) -> str:
    """Run the (synchronous) SDK call in a thread so we don't block the loop."""
    import anyio

    def _sync() -> str:
        r = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_LLMEnrichment,
                temperature=0.4,
            ),
        )
        return r.text

    return await anyio.to_thread.run_sync(_sync)
