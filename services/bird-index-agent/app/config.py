"""Configuration for the Bird Index Agent (orchestrator).

Env-driven so the same image runs in docker-compose (dev) and elsewhere.
CANTOS_USE_STUBS toggles between in-process stubs (build step 3) and real
downstream agents (later steps).
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _get(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    use_stubs: bool = _get("CBSI_USE_STUBS", "true").lower() == "true"

    recorder_url: str = _get("RECORDER_URL", "http://recorder:8000")
    identifier_url: str = _get("IDENTIFIER_URL", "http://identifier:8000")
    mcp_url: str = _get("MCP_URL", "http://birddex-mcp:8000")

    mongo_uri: str = _get("MONGO_URI", "mongodb://mongo:27017")
    mongo_db: str = _get("MONGO_DB", "cbsi")
    redis_url: str = _get("REDIS_URL", "redis://redis:6379/0")

    enrich_timeout_s: float = float(_get("ENRICH_TIMEOUT_S", "6.0"))
    downstream_timeout_s: float = float(_get("DOWNSTREAM_TIMEOUT_S", "15.0"))

    max_upload_bytes: int = int(_get("MAX_UPLOAD_BYTES", str(15 * 1024 * 1024)))


settings = Settings()
