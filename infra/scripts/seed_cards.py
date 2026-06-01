"""Seed MongoDB `cards` collection from your colombian_birds.json.

This bridges your existing pre-processing output (the "ornithological cards"
built in pre-processing.ipynb Step 4.5) into the backend's metadata store. It
keys each document by `label` — the underscored species name your classifier
emits via label_map.json — so the Index Agent can look up a card by the exact
string the model predicts.

Usage:
    pip install pymongo
    python infra/scripts/seed_cards.py \
        --json /path/to/colombian_birds.json \
        --mongo mongodb://localhost:27017 --db cbsi

The input JSON is expected to be either:
    {"species": {"<scientific or key>": {card...}, ...}}   (preferred)
or a flat list/dict of cards. Each card carries scientific_name; we derive the
underscored `label` from scientific_name unless an explicit label is present.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def derive_label(card: dict, fallback_key: str) -> str:
    # Prefer an explicit label; else underscore the scientific name; else the key.
    if card.get("label"):
        return card["label"]
    sci = card.get("scientific_name") or fallback_key
    return sci.strip().replace(" ", "_")


def normalize_cards(raw) -> list[dict]:
    """Return a list of card dicts regardless of the container shape."""
    if isinstance(raw, dict) and "species" in raw and isinstance(raw["species"], dict):
        items = raw["species"].items()
    elif isinstance(raw, dict):
        items = raw.items()
    elif isinstance(raw, list):
        items = [(c.get("scientific_name", str(i)), c) for i, c in enumerate(raw)]
    else:
        raise ValueError("Unrecognized colombian_birds.json structure")

    out = []
    for key, card in items:
        if not isinstance(card, dict):
            continue
        doc = dict(card)
        doc["label"] = derive_label(card, str(key))
        out.append(doc)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Path to colombian_birds.json")
    ap.add_argument("--mongo", default="mongodb://localhost:27017")
    ap.add_argument("--db", default="cbsi")
    ap.add_argument("--dry-run", action="store_true", help="Parse only; no DB writes")
    args = ap.parse_args()

    path = Path(args.json)
    if not path.exists():
        print(f"ERROR: {path} not found", file=sys.stderr)
        return 1

    raw = json.loads(path.read_text(encoding="utf-8"))
    cards = normalize_cards(raw)
    print(f"Parsed {len(cards)} cards. Sample labels: "
          f"{[c['label'] for c in cards[:5]]}")

    if args.dry_run:
        print("Dry run — no writes.")
        return 0

    from pymongo import MongoClient, UpdateOne

    client = MongoClient(args.mongo)
    coll = client[args.db].cards
    coll.create_index("label", unique=True)

    ops = [UpdateOne({"label": c["label"]}, {"$set": c}, upsert=True) for c in cards]
    if ops:
        res = coll.bulk_write(ops, ordered=False)
        print(f"Upserted {res.upserted_count}, modified {res.modified_count}.")
    print(f"cards collection now holds {coll.count_documents({})} documents.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
