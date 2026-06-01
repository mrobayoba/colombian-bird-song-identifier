"""MongoDB card lookup for the Bird Index Agent.

The `cards` collection is seeded directly from your colombian_birds.json
(pre-processing.ipynb Step 4.5). Each document is keyed by `label` (the
underscored species name the classifier emits), so lookup is an exact match
on the model's output.

Used only when CBSI_USE_STUBS=false; motor is imported lazily.
"""

from __future__ import annotations

from typing import Optional

from contracts import BirdCard

from .config import settings


class MongoCards:
    def __init__(self) -> None:
        from motor.motor_asyncio import AsyncIOMotorClient

        self._client = AsyncIOMotorClient(settings.mongo_uri, maxPoolSize=50)
        self._db = self._client[settings.mongo_db]

    async def ensure_indexes(self) -> None:
        await self._db.cards.create_index("label", unique=True)

    async def lookup(self, label: str) -> Optional[BirdCard]:
        doc = await self._db.cards.find_one({"label": label}, {"_id": 0})
        return BirdCard(**doc) if doc else None

    async def close(self) -> None:
        self._client.close()
