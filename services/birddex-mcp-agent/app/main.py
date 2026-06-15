"""BirdDex MCP Agent — LLM enrichment via Google Gemini.

Invoked ONLY by the Bird Index Agent. Given a species, it returns a grounded
Spanish summary, behavior notes, semantic tags, taxonomically related
alternatives, and a read_more_target identifier for the frontend UI.

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

app = FastAPI(title="CBSI BirdDex MCP Agent", version="0.2.0")

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
    alternatives: list[str] = []
    read_more_target: str


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
        alternatives=[],
        read_more_target=req.scientific_name,
        sources=[f"card:{req.label}"],
        model="template-fallback",
    )


def _build_prompt(card: dict) -> str:
    """Give the model ONLY trusted fields and precise extraction instructions."""
    import json

    trusted = {
        "scientific_name": card.get("scientific_name"),
        "english_name": card.get("english_name"),
        "family": card.get("taxonomy", {}).get("family"),
        "order": card.get("taxonomy", {}).get("order"),
        "genus": card.get("taxonomy", {}).get("genus"),
        "conservation_status_code": card.get("conservation_status", {}).get("code"),
        "conservation_status_es": card.get("conservation_status", {}).get("label_es"),
        "description": card.get("description"),
        "physical_description": card.get("physical_description"),
        "plumage": card.get("plumage"),
        "habitat": card.get("habitat"),
        "behavior": card.get("behavior"),
        "diet": card.get("diet"),
        "related_species": card.get("related_species"),
        "lookalike_species": card.get("lookalike_species"),
        "inat_sightings": card.get("inat_sightings"),
        "xeno_canto_recordings": card.get("xeno_canto_recordings"),
    }
    # Strip absent fields so the LLM is not misled by explicit nulls.
    trusted = {k: v for k, v in trusted.items() if v is not None}

    return (
        "Eres un ornitologo experto en aves colombianas. Tu unica fuente de "
        "informacion es la ficha verificada que aparece al final. PROHIBIDO "
        "inventar, extrapolar o usar conocimiento de tu entrenamiento que no "
        "este explicitamente presente en dicha ficha.\n\n"
        "Produce un objeto JSON con exactamente los siguientes cinco campos:\n\n"
        "1. summary_es (string): Resumen de divulgacion en español"
        "   mostrando de a 3 a 5 ideas sobre el ave en cuestion y de manera coherente."
        "   Incluye morfologia (tamaño, coloracion, plumaje) si la ficha "
        "   los describe, estado de conservacion IUCN con su significado, y "
        "   distribucion o habitat si estan disponibles. No menciones campos "
        "   ausentes. Asegurate de no cortar ningunafrase en el medio de una idea.\n\n"
        "2. behavior_notes (string | null): Una o dos frases sobre comportamiento, "
        "   dieta o vocalizaciones si la ficha los respalda. null si no hay datos.\n\n"
        "3. semantic_tags (array de strings): Entre 3 y 6 etiquetas cortas en "
        "   español utiles para la busqueda en la web (familia, habitat, codigo IUCN, "
        "   orden, etc.).\n\n"
        "4. alternatives (array de strings): Nombres cientificos de 2 a 3 especies "
        "   taxonomicamente cercanas o visualmente similares. Usa EXCLUSIVAMENTE "
        "   los valores de los campos 'related_species' o 'lookalike_species' de "
        "   la ficha. Si esos campos no existen o estan vacios, devuelve [].\n\n"
        "5. read_more_target (string): El nombre cientifico canonico de esta "
        "   especie en formato binomial limpio (sin guiones bajos, sin abreviaturas), "
        "   por ejemplo 'Turdus fuscater'. El frontend lo usara como parametro del "
        "   boton 'Leer mas'.\n\n"
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
            alternatives=parsed.alternatives,
            read_more_target=parsed.read_more_target,
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
