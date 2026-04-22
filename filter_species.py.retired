import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

INPUT_PATH = Path("data/colombia_birds_by_species.json")
OUTPUT_PATH = Path("data/colombia_birds_filtered.json")

MIN_LENGTH_SEC = 3
MAX_LENGTH_SEC = 60
ACCEPTABLE_QUALITY = frozenset({"A", "B"})
MIN_SPECIES_RECORDINGS = 35


def length_seconds(s: str) -> int:
    """Convert 'MM:SS' or 'HH:MM:SS' to total seconds."""
    parts = [int(p) for p in s.strip().split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    raise ValueError(f"Unexpected length format: {s!r}")


def main() -> None:
    # --- Load & validate input ---
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    with INPUT_PATH.open(encoding="utf-8") as f:
        data = json.load(f)

    if "species" not in data:
        raise KeyError("Input JSON is missing the 'species' key.")

    new_species: dict = {}
    total_recordings = 0

    for name, info in data["species"].items():
        if info.get("recording_count", 0) < MIN_SPECIES_RECORDINGS:
            continue

        kept = []
        for r in info["recordings"]:
            if not r.get("length"):
                logger.warning("Skipping recording with missing 'length' in species %r", name)
                continue

            q = (r.get("quality") or "").strip()
            if q not in ACCEPTABLE_QUALITY:
                continue

            duration = length_seconds(r["length"])
            if not (MIN_LENGTH_SEC <= duration <= MAX_LENGTH_SEC):
                continue

            kept.append(r)

        if len(kept) <= MIN_SPECIES_RECORDINGS:
            logger.info(
                "Dropping %r: only %d recordings survive filtering (need > %d).",
                name, len(kept), MIN_SPECIES_RECORDINGS,
            )
            continue

        species_out = dict(info)
        species_out["recordings"] = kept
        species_out["recording_count"] = len(kept)
        new_species[name] = species_out
        total_recordings += len(kept)

    # --- Assertion: hard guarantee on every species in the output ---
    for name, species_out in new_species.items():
        assert species_out["recording_count"] > MIN_SPECIES_RECORDINGS, (
            f"Integrity check failed for {name!r}: "
            f"recording_count={species_out['recording_count']}"
        )

    out_data = {
        "query": data["query"],
        "total_recordings": total_recordings,
        "total_species": len(new_species),
        "species": new_species,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    logger.info(
        "Done. %d species | %d recordings → %s",
        len(new_species), total_recordings, OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()