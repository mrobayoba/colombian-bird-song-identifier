"""Shared contracts for the Colombian Bird Song Identifier backend.

These shapes are adapted to YOUR existing pipeline:

* Species identity is the underscored name from `label_map.json`
  (e.g. "Turdus_fuscater"), exactly as the trained Keras model emits via its
  output index. We carry both the raw label and a display name.
* BirdCard mirrors the "ornithological card" JSON your pre-processing notebook
  already builds (colombian_birds.json): scientific_name, english_name,
  description, conservation_status, taxonomy, image_url, etc. That file is the
  natural MongoDB seed, so the contract matches it field-for-field.
* The classifier averages probabilities across chunks (train.ipynb Step 11),
  so ClassificationResult reflects a clip-level, chunk-averaged result.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class JobStatus(str, Enum):
    received = "received"
    preprocessing = "preprocessing"
    classifying = "classifying"
    enriching = "enriching"
    done = "done"
    failed = "failed"


# --------------------------------------------------------------------------- #
# Classification (output of the ONLY ML agent — your Keras ResNet50V2)
# --------------------------------------------------------------------------- #
class Candidate(BaseModel):
    label: str = Field(..., description="Raw label_map value, e.g. 'Turdus_fuscater'")
    display_name: str = Field(..., description="Human form, e.g. 'Turdus fuscater'")
    confidence: float = Field(..., ge=0.0, le=1.0)


class ClassificationResult(BaseModel):
    """Clip-level result after averaging softmax over all surviving chunks."""

    label: str = Field(..., description="Top label_map value")
    display_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    alternatives: list[Candidate] = Field(default_factory=list)
    n_chunks: int = Field(0, description="How many 3 s chunks were classified")
    model_version: Optional[str] = None


# --------------------------------------------------------------------------- #
# Metadata — mirrors colombian_birds.json "ornithological cards"
# --------------------------------------------------------------------------- #
class ConservationStatus(BaseModel):
    code: str = Field("NE", description="IUCN code: LC/NT/VU/EN/CR/EW/EX/DD/NE")
    label_es: str = ""
    label_en: str = ""


class Taxonomy(BaseModel):
    kingdom: Optional[str] = None
    phylum: Optional[str] = None
    class_: Optional[str] = Field(None, alias="class")
    order: Optional[str] = None
    family: Optional[str] = None
    genus: Optional[str] = None
    species: Optional[str] = None

    model_config = {"populate_by_name": True}


class BirdCard(BaseModel):
    """One ornithological card, as built by pre-processing.ipynb Step 4.5."""

    label: str = Field(..., description="Underscored key used by the classifier")
    scientific_name: str
    english_name: str = ""
    description: str = ""
    description_source: str = ""
    conservation_status: ConservationStatus = Field(default_factory=ConservationStatus)
    taxonomy: Taxonomy = Field(default_factory=Taxonomy)
    inat_sightings: int = 0
    xeno_canto_recordings: int = 0
    image_url: str = ""
    info_url: str = ""


# --------------------------------------------------------------------------- #
# Enrichment (BirdDex MCP agent) — additive semantic layer on top of the card
# --------------------------------------------------------------------------- #
class EnrichmentResult(BaseModel):
    summary_es: str
    behavior_notes: Optional[str] = None
    semantic_tags: list[str] = Field(default_factory=list)
    alternatives: list[str] = Field(
        default_factory=list,
        description="Scientific names of taxonomically related or lookalike species",
    )
    read_more_target: str = Field(
        "", description="Canonical binomial name used by the frontend Read More button"
    )
    sources: list[str] = Field(
        default_factory=list, description="Provenance, e.g. ['card:Turdus_fuscater']"
    )
    model: Optional[str] = None


# --------------------------------------------------------------------------- #
# Unified response to the frontend
# --------------------------------------------------------------------------- #
class UnifiedResponse(BaseModel):
    job_id: str
    status: JobStatus = JobStatus.done
    identification: ClassificationResult
    card: Optional[BirdCard] = None
    enrichment: Optional[EnrichmentResult] = None
    degraded: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)


# --------------------------------------------------------------------------- #
# Internal inter-agent payloads
# --------------------------------------------------------------------------- #
class SpectrogramTensors(BaseModel):
    """Preprocessed tensors handed from Recorder -> Identifier.

    Shape is (n_chunks, 128, 188). In production this travels as a compact
    binary (e.g. gRPC bytes / base64 npy); the JSON form here is for stubs/tests.
    """

    job_id: str
    n_chunks: int
    height: int = 128
    width: int = 188
    # base64-encoded float32 npy payload in real use; omitted by stubs
    npy_b64: Optional[str] = None


class EnrichRequest(BaseModel):
    label: str
    scientific_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    job_id: Optional[str] = None
