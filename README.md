<div align="center">

# 🦜 Colombian Bird Song Identifier

**An end-to-end pipeline for building an AI classifier that identifies Colombian bird species from audio recordings.**

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Data source](https://img.shields.io/badge/Data-Xeno--Canto%20API%20v3-green)](https://xeno-canto.org/explore/api)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

</div>

---

## Overview

This project provides a complete pipeline — from data collection to model inference — for classifying bird species
found in Colombia based on their songs and calls. The classifier is trained on recordings sourced from
[Xeno-Canto](https://xeno-canto.org) via their API v3.

**Colombia is one of the most biodiverse countries in the world, home to over 1,900 bird species.**
This project currently covers **1,570+ species** with **36,900+ recordings**.

---

## Pipeline

```
Fetch metadata  →  Filter species  →  Download audio  →  Train model  →  Predict
     ✅                  🔜                 🔜                🔜             🔜
```

---

## Project Structure

```
colombian-bird-song-identifier/
│
├── data/                               # Downloaded data — gitignored
│   ├── raw/                            # Raw API responses (one JSON per page)
│   │   ├── page_0001.json
│   │   ├── page_0002.json
│   │   └── ...
│   ├── audio/                          # Audio files organised by species
│   │   ├── Amazona amazonica/
│   │   │   ├── XC123456.mp3
│   │   │   └── XC123457.mp3
│   │   └── ...
│   ├── colombia_birds_by_species.json  # Full grouped metadata
│   └── colombia_birds_filtered.json   # Metadata after threshold filtering
│
├── fetch_metadata.py                   # Step 1 — fetch metadata from Xeno-Canto
├── filter_species.py                   # Step 2 — filter by quality, length, count
├── requirements.txt                    # Python dependencies
└── .env                                # API key (gitignored)
```

---

## Setup

**Prerequisites:** [Python 3.12](https://www.python.org/) and [uv](https://github.com/astral-sh/uv).

```bash
# Clone the repository
git clone https://github.com/your-username/colombian-bird-song-identifier.git
cd colombian-bird-song-identifier

# Create a virtual environment and install dependencies
uv venv .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

Create a `.env` file in the project root with your [Xeno-Canto API key](https://xeno-canto.org/account):

```env
XC_API_KEY=your_key_here
```

---

## Usage

### Step 1 — Fetch recording metadata ✅

Downloads metadata for all bird recordings made in Colombia from Xeno-Canto and groups them by species.
Raw API responses are cached in `data/raw/` so the step does not need to be re-run if the data is already present.

```bash
.venv/Scripts/python.exe fetch_metadata.py
```

**Output:** `data/colombia_birds_by_species.json`

Each recording entry includes:

| Field | Description |
|---|---|
| `id` | Xeno-Canto recording ID |
| `quality` | Quality rating (`A` best → `E` worst) |
| `length` | Duration (`mm:ss`) |
| `file` | Direct MP3 download URL |
| `file_name` | Original filename (`XC{id}.mp3`) |
| `type` | Sound type (song, call, …) |
| `lat` / `lon` | Recording coordinates |
| `date` | Recording date |
| `also` | Background species present |

---

### Step 2 — Filter species 🔜

Pre-process the metadata before downloading audio by discarding species and individual recordings that do not
meet configurable thresholds. Species with too few valid recordings after filtering are dropped entirely.

```bash
.venv/Scripts/python.exe filter_species.py [OPTIONS]
```

| Option | Description | Default |
|---|---|---|
| `--min-quality` | Minimum quality rating to keep (`A`–`E`) | `C` |
| `--min-length` | Minimum recording length (seconds) | — |
| `--max-length` | Maximum recording length (seconds) | — |
| `--min-recordings` | Minimum recordings a species must have after filtering | — |

**Output:** `data/colombia_birds_filtered.json`

---

### Step 3 — Download audio 🔜

Download the MP3 files for all recordings in the filtered dataset, saving them into per-species subfolders
under `data/audio/`. Files are named `XC{id}.{ext}` (e.g. `XC694038.mp3`).

---

### Step 4 — Train the model 🔜

Convert audio to mel-spectrograms and train a CNN-based classifier on the resulting images.

---

### Step 5 — Predict 🔜

Run inference on a new audio file to return the most likely bird species.

---

## Data Source

Recording metadata and audio are provided by [Xeno-Canto](https://xeno-canto.org), a platform for sharing
wildlife sounds from around the world, powered by the Xeno-Canto Foundation and Naturalis Biodiversity Center.

> Recordings are © their respective recordists. See the `license` field of each recording for terms of use.

