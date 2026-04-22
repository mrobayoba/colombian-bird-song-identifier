"""
fetch_metadata.py
-----------------
Fetches recording metadata from the Xeno-Canto API v3 for all bird recordings
made in Colombia (cnt:colombia grp:birds).

API v3 key differences from v2:
  - Endpoint: https://xeno-canto.org/api/3/recordings
  - `key` query parameter is required on every request.
  - Queries MUST use search tags (e.g. cnt:, grp:); tag-less terms are rejected.
  - Recording object uses `grp` (not `group`) and `lon` (not `lng`).
  - `per_page` controls page size (50–500, default 100); we use 500 for efficiency.
  - Strict rate limit has been lifted.

What this script does:
  1. Pages through the full result set using per_page=500.
  2. Saves every raw API response as data/raw/page_NNNN.json.
  3. After all pages are fetched, validates each recording:
       - cnt  == "Colombia"
       - grp  == "birds"
       - lat/lon fall inside Colombia's bounding box (approx)
  4. Aggregates validated recordings into data/colombia_birds_by_species.json
     keyed by "Genus species".

Usage:
    pip install -r requirements.txt
    python fetch_metadata.py

Output structure:
    data/
      raw/
        page_0001.json   # raw API responses (one per page)
        page_0002.json
        ...
      colombia_birds_by_species.json   # final grouped output
"""

import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

API_KEY: str = os.getenv("XC_API_KEY", "")
if not API_KEY:
    raise RuntimeError("XC_API_KEY is not set. Add it to your .env file.")

BASE_URL: str = "https://xeno-canto.org/api/3/recordings"

# API v3 query — only search tags are accepted (tag-less queries are rejected)
# Use spaces between tags; requests handles URL-encoding correctly.
QUERY: str = "cnt:colombia grp:birds"

# API v3 has lifted the strict rate limit; use 500 results per page for efficiency.
PER_PAGE: int = 500

# Colombia's geographic bounding box (loose, includes territorial waters)
COL_LAT_MIN, COL_LAT_MAX = -4.5, 13.0
COL_LNG_MIN, COL_LNG_MAX = -82.0, -66.5

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def fetch_page(query: str, page: int) -> dict:
    """
    Fetch one page of results from the Xeno-Canto API v3.

    `key` is required by v3 and is always included.
    Raises requests.HTTPError on non-2xx responses.
    """
    params: dict = {
        "query": query,
        "key": API_KEY,
        "per_page": PER_PAGE,
        "page": page,
    }

    response = requests.get(
        BASE_URL,
        params=params,
        headers={"Accept": "application/json"},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# Fetch all pages
# ---------------------------------------------------------------------------

def fetch_all_pages() -> list[dict]:
    """
    Iterate through all pages for the Colombian birds query.
    Saves each raw page response to data/raw/.
    Returns a flat list of all recording dicts.
    """
    page = 1
    print(f"[page {page}] fetching …")
    data = fetch_page(QUERY, page)

    num_recordings = data.get("numRecordings", "?")
    num_species = data.get("numSpecies", "?")
    num_pages = int(data.get("numPages", 1))

    print(f"  Total recordings : {num_recordings}")
    print(f"  Total species    : {num_species}")
    print(f"  Total pages      : {num_pages}")

    all_recordings: list[dict] = list(data.get("recordings", []))
    _save_raw_page(data, page)

    for page in range(2, num_pages + 1):
        print(f"[page {page}/{num_pages}] fetching …")
        data = fetch_page(QUERY, page)
        recordings = data.get("recordings", [])
        all_recordings.extend(recordings)
        _save_raw_page(data, page)
        print(f"  collected {len(recordings)} recordings (total so far: {len(all_recordings)})")

    return all_recordings


def _save_raw_page(data: dict, page: int) -> None:
    path = RAW_DIR / f"page_{page:04d}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _in_colombia_bbox(lat: str, lng: str) -> bool:
    """Return True when decimal lat/lng fall within Colombia's bounding box."""
    try:
        flat, flng = float(lat), float(lng)
    except (ValueError, TypeError):
        return True  # no coordinates → do not discard, trust cnt field
    return (COL_LAT_MIN <= flat <= COL_LAT_MAX) and (COL_LNG_MIN <= flng <= COL_LNG_MAX)


def validate_colombian_birds(recordings: list[dict]) -> list[dict]:
    """
    Keep only recordings that satisfy all of:
      - grp == "birds"   (API v3 field name)
      - cnt == "Colombia"  (case-insensitive)
      - lat/lon inside Colombia's bounding box (when coordinates are present)
    """
    validated = []
    for r in recordings:
        if r.get("grp", "").lower() != "birds":
            continue
        if r.get("cnt", "").lower() != "colombia":
            continue
        if not _in_colombia_bbox(r.get("lat", ""), r.get("lon", "")):
            continue
        validated.append(r)
    return validated


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------

def group_by_species(recordings: list[dict]) -> dict:
    """
    Return a dict keyed by "Genus species" (scientific name).
    Each value contains species-level metadata plus a list of recordings.
    """
    species_map: dict = {}

    for r in recordings:
        genus = r.get("gen", "").strip()
        epithet = r.get("sp", "").strip()
        key = f"{genus} {epithet}".strip()

        if key not in species_map:
            species_map[key] = {
                "scientific_name": key,
                "genus": genus,
                "species": epithet,
                "subspecies_seen": sorted({r.get("ssp", "").strip()} - {""}),
                "english_name": r.get("en", ""),
                "group": r.get("grp", ""),
                "recording_count": 0,
                "recordings": [],
            }

        entry = species_map[key]

        # Track subspecies encountered
        ssp = r.get("ssp", "").strip()
        if ssp and ssp not in entry["subspecies_seen"]:
            entry["subspecies_seen"].append(ssp)

        entry["recording_count"] += 1
        entry["recordings"].append({
            "id": r.get("id", ""),
            "quality": r.get("q", ""),
            "length": r.get("length", ""),
            "file": r.get("file", ""),
            "file_name": r.get("file-name", ""),
            "type": r.get("type", ""),
            "sex": r.get("sex", ""),
            "stage": r.get("stage", ""),
            "method": r.get("method", ""),
            "location": r.get("loc", ""),
            "lat": r.get("lat", ""),
            "lon": r.get("lon", ""),   # v3: lon (not lng)
            "date": r.get("date", ""),
            "recordist": r.get("rec", ""),
            "license": r.get("lic", ""),
            "xc_url": r.get("url", ""),
            "also": r.get("also", []),
            "animal_seen": r.get("animal-seen", ""),
            "remarks": r.get("rmk", ""),
        })

    return species_map


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Xeno-Canto – Colombian bird song metadata fetcher (API v3)")
    print("=" * 60)

    # Step 1 – fetch all raw pages
    all_recordings = fetch_all_pages()
    print(f"\nFetched {len(all_recordings)} total recordings across all pages.")

    # Step 2 – validate
    print("\nValidating recordings (country + bounding box) …")
    validated = validate_colombian_birds(all_recordings)
    discarded = len(all_recordings) - len(validated)
    print(f"  Validated : {len(validated)}")
    print(f"  Discarded : {discarded} (wrong country or outside Colombia's bbox)")

    # Step 3 – group by species
    print("\nGrouping by species …")
    species_map = group_by_species(validated)
    print(f"  Species found : {len(species_map)}")

    # Step 4 – save final output
    output = {
        "query": QUERY,
        "total_recordings": len(validated),
        "total_species": len(species_map),
        # Sort alphabetically by scientific name for readability
        "species": dict(sorted(species_map.items())),
    }

    out_path = Path("data/colombia_birds_by_species.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False, indent=2)

    print(f"\nGrouped metadata saved to: {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
