#!/usr/bin/env python3
"""
bird_card.py  —  Generate an ornithological field card from colombian_birds.json

Usage:
    python bird_card.py "Ara macao"
    python bird_card.py "Ara macao" --json path/to/colombian_birds.json
    python bird_card.py "Ara macao" --out my_card.html
"""

import argparse
import json
import sys
from pathlib import Path

# ── IUCN badge colours ─────────────────────────────────────────────────────
IUCN_COLOURS = {
    "LC": "#5a8a3c",
    "NT": "#9dbf4d",
    "VU": "#f2a500",
    "EN": "#e06000",
    "CR": "#c00020",
    "EW": "#542344",
    "EX": "#000000",
    "DD": "#7b7b7b",
    "NE": "#aaaaaa",
}

# ── HTML template ──────────────────────────────────────────────────────────
HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{english_name} · {scientific_name}</title>
<style>
  /* ── Reset & base ── */
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    background: #0e1510;
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 2rem 1rem;
    font-family: 'Georgia', serif;
  }}

  /* ── Card shell ── */
  .card {{
    width: 100%;
    max-width: 720px;
    background: #f7f2e8;
    border-radius: 2px;
    overflow: hidden;
    box-shadow: 0 8px 40px rgba(0,0,0,0.55);
    position: relative;
  }}

  /* ── Photo panel ── */
  .photo-panel {{
    position: relative;
    height: 340px;
    background: #1a2318;
    overflow: hidden;
  }}

  .photo-panel img {{
    width: 100%;
    height: 100%;
    object-fit: cover;
    object-position: center 30%;
    display: block;
    opacity: 0.88;
  }}

  .photo-panel .overlay {{
    position: absolute;
    inset: 0;
    background: linear-gradient(
      to bottom,
      transparent 40%,
      rgba(14, 21, 16, 0.82) 100%
    );
  }}

  /* ── Names on photo ── */
  .names {{
    position: absolute;
    bottom: 1.5rem;
    left: 1.75rem;
    right: 1.75rem;
    color: #fff;
  }}

  .names .common {{
    font-family: 'Georgia', serif;
    font-size: 1.65rem;
    font-weight: normal;
    letter-spacing: 0.01em;
    line-height: 1.15;
    text-shadow: 0 1px 6px rgba(0,0,0,0.6);
  }}

  .names .scientific {{
    font-family: 'Georgia', serif;
    font-size: 0.95rem;
    font-style: italic;
    color: rgba(255,255,255,0.72);
    margin-top: 0.3rem;
    letter-spacing: 0.03em;
  }}

  /* ── IUCN badge on photo ── */
  .iucn-badge {{
    position: absolute;
    top: 1.25rem;
    right: 1.25rem;
    padding: 0.35rem 0.75rem;
    border-radius: 2px;
    font-family: 'Courier New', monospace;
    font-size: 0.78rem;
    font-weight: bold;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #fff;
    background: {iucn_colour};
    box-shadow: 0 2px 8px rgba(0,0,0,0.4);
  }}

  .iucn-badge .iucn-label {{
    display: block;
    font-size: 0.6rem;
    font-weight: normal;
    letter-spacing: 0.05em;
    opacity: 0.85;
    margin-bottom: 0.15rem;
  }}

  /* ── Body ── */
  .body {{
    padding: 1.75rem 1.75rem 2rem;
  }}

  /* ── Stats row ── */
  .stats {{
    display: flex;
    gap: 0;
    border: 1px solid #d4cbb8;
    border-radius: 1px;
    overflow: hidden;
    margin-bottom: 1.5rem;
  }}

  .stat {{
    flex: 1;
    padding: 0.7rem 0.6rem;
    text-align: center;
    border-right: 1px solid #d4cbb8;
  }}
  .stat:last-child {{ border-right: none; }}

  .stat .value {{
    display: block;
    font-family: 'Courier New', monospace;
    font-size: 1.2rem;
    color: #2c4a22;
    font-weight: bold;
    letter-spacing: -0.01em;
  }}

  .stat .label {{
    display: block;
    font-family: sans-serif;
    font-size: 0.6rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #8a7f6a;
    margin-top: 0.2rem;
  }}

  /* ── Description ── */
  .description {{
    font-size: 0.95rem;
    line-height: 1.75;
    color: #2e2a1e;
    margin-bottom: 1.5rem;
    text-align: justify;
    hyphens: auto;
  }}

  /* ── Taxonomy table ── */
  .section-label {{
    font-family: sans-serif;
    font-size: 0.6rem;
    font-weight: bold;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: #8a7f6a;
    border-bottom: 1px solid #d4cbb8;
    padding-bottom: 0.4rem;
    margin-bottom: 0.85rem;
  }}

  .taxonomy {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.5rem 0.75rem;
    margin-bottom: 1.5rem;
  }}

  .tax-item .tax-rank {{
    font-family: sans-serif;
    font-size: 0.58rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #8a7f6a;
  }}

  .tax-item .tax-value {{
    font-family: 'Georgia', serif;
    font-size: 0.85rem;
    font-style: italic;
    color: #2e2a1e;
    margin-top: 0.1rem;
  }}

  .tax-item.no-italic .tax-value {{
    font-style: normal;
  }}

  /* ── Footer ── */
  .footer {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-top: 1.1rem;
    border-top: 1px solid #d4cbb8;
    gap: 1rem;
  }}

  .footer a {{
    font-family: sans-serif;
    font-size: 0.68rem;
    color: #5a7a40;
    text-decoration: none;
    letter-spacing: 0.04em;
    border-bottom: 1px solid #b0c898;
    padding-bottom: 1px;
  }}

  .footer a:hover {{ color: #2c4a22; border-color: #2c4a22; }}

  .footer .source-tag {{
    font-family: 'Courier New', monospace;
    font-size: 0.62rem;
    color: #aaa099;
    letter-spacing: 0.04em;
    white-space: nowrap;
  }}

  /* ── Watermark stripe ── */
  .stripe {{
    height: 4px;
    background: linear-gradient(90deg, #2c4a22 0%, #5a8a3c 50%, #9dbf4d 100%);
  }}
</style>
</head>
<body>
<div class="card">
  <div class="stripe"></div>

  <div class="photo-panel">
    <img src="{image_url}" alt="{english_name}" onerror="this.style.display='none'">
    <div class="overlay"></div>
    <div class="iucn-badge" style="background:{iucn_colour};">
      <span class="iucn-label">IUCN</span>
      {iucn_code}
    </div>
    <div class="names">
      <div class="common">{english_name}</div>
      <div class="scientific">{scientific_name}</div>
    </div>
  </div>

  <div class="body">

    <div class="stats">
      <div class="stat">
        <span class="value">{inat_sightings}</span>
        <span class="label">iNat sightings</span>
      </div>
      <div class="stat">
        <span class="value">{xeno_canto_recordings}</span>
        <span class="label">Xeno-canto recordings</span>
      </div>
      <div class="stat">
        <span class="value">{order}</span>
        <span class="label">Order</span>
      </div>
      <div class="stat">
        <span class="value">{family}</span>
        <span class="label">Family</span>
      </div>
    </div>

    <p class="description">{description}</p>

    <div class="section-label">Taxonomy</div>
    <div class="taxonomy">
      <div class="tax-item no-italic">
        <div class="tax-rank">Kingdom</div>
        <div class="tax-value">{kingdom}</div>
      </div>
      <div class="tax-item no-italic">
        <div class="tax-rank">Phylum</div>
        <div class="tax-value">{phylum}</div>
      </div>
      <div class="tax-item no-italic">
        <div class="tax-rank">Class</div>
        <div class="tax-value">{class_}</div>
      </div>
      <div class="tax-item no-italic">
        <div class="tax-rank">Order</div>
        <div class="tax-value">{order}</div>
      </div>
      <div class="tax-item no-italic">
        <div class="tax-rank">Family</div>
        <div class="tax-value">{family}</div>
      </div>
      <div class="tax-item">
        <div class="tax-rank">Genus</div>
        <div class="tax-value">{genus}</div>
      </div>
      <div class="tax-item">
        <div class="tax-rank">Species</div>
        <div class="tax-value">{species}</div>
      </div>
      <div class="tax-item no-italic">
        <div class="tax-rank">Status</div>
        <div class="tax-value" style="color:{iucn_colour}; font-style:normal; font-weight:bold;">
          {iucn_code} · {iucn_label_es}
        </div>
      </div>
    </div>

    <div class="footer">
      <a href="{info_url}" target="_blank" rel="noopener">Wikipedia →</a>
      <span class="source-tag">desc: {description_source}</span>
    </div>

  </div>
</div>
</body>
</html>
"""


def fmt_number(n: int) -> str:
    """Format integer with thousands separator."""
    return f"{n:,}".replace(",", ".")


def build_card(entry: dict, out_path: Path) -> None:
    """Render a single species entry to an HTML file."""
    tax  = entry.get("taxonomy", {})
    cs   = entry.get("conservation_status") or {}
    code = cs.get("code", "NE")

    html = HTML_TEMPLATE.format(
        scientific_name      = entry.get("scientific_name", ""),
        english_name         = entry.get("english_name", entry.get("scientific_name", "")),
        image_url            = entry.get("image_url", ""),
        info_url             = entry.get("info_url", "#"),
        description          = entry.get("description", "No description available."),
        description_source   = entry.get("description_source", "—"),
        inat_sightings       = fmt_number(entry.get("inat_sightings", 0)),
        xeno_canto_recordings= fmt_number(entry.get("xeno_canto_recordings", 0)),
        iucn_code            = code,
        iucn_colour          = IUCN_COLOURS.get(code, IUCN_COLOURS["NE"]),
        iucn_label_es        = cs.get("label_es", "No evaluada"),
        kingdom              = tax.get("kingdom", ""),
        phylum               = tax.get("phylum", ""),
        class_               = tax.get("class", ""),
        order                = tax.get("order", ""),
        family               = tax.get("family", ""),
        genus                = tax.get("genus", ""),
        species              = tax.get("species", ""),
    )

    out_path.write_text(html, encoding="utf-8")
    print(f"Card saved → {out_path}")


def load_catalogue(json_path: Path) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Support both raw dict-of-species and the wrapped {"species": {...}} format
    if isinstance(data, dict) and "species" in data:
        return data["species"]
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an HTML bird card from colombian_birds.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "scientific_name",
        help='Scientific name of the species, e.g. "Ara macao"',
    )
    parser.add_argument(
        "--json",
        default="colombian_birds.json",
        metavar="PATH",
        help="Path to colombian_birds.json  (default: ./colombian_birds.json)",
    )
    parser.add_argument(
        "--out",
        default=None,
        metavar="PATH",
        help="Output HTML file  (default: <Genus_species>_card.html)",
    )
    args = parser.parse_args()

    json_path = Path(args.json)
    if not json_path.exists():
        sys.exit(f"Error: JSON file not found at '{json_path}'")

    catalogue = load_catalogue(json_path)

    # Case-insensitive lookup
    name = args.scientific_name.strip()
    entry = catalogue.get(name)
    if entry is None:
        # Try case-insensitive fallback
        lower = name.lower()
        entry = next(
            (v for k, v in catalogue.items() if k.lower() == lower),
            None,
        )
    if entry is None:
        available = "\n  ".join(sorted(catalogue.keys())[:20])
        sys.exit(
            f"Error: '{name}' not found in catalogue.\n\n"
            f"First 20 available species:\n  {available}\n"
            f"  … ({len(catalogue)} total)"
        )

    out_path = Path(args.out) if args.out else Path(
        name.replace(" ", "_") + "_card.html"
    )
    build_card(entry, out_path)


if __name__ == "__main__":
    main()
