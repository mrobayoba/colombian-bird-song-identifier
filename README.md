<div align="center">

# 🦜 Colombian Bird Song Identifier

**An end-to-end pipeline for building an AI classifier that identifies Colombian bird species from audio recordings.**

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Data source](https://img.shields.io/badge/Data-Xeno--Canto%20API%20v3-green)](https://xeno-canto.org/explore/api)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

</div>

---

## Overview

This project provides a notebook-first pipeline for collecting and preparing Colombian bird song data from
[Xeno-Canto](https://xeno-canto.org) API v3. The current workflow fetches metadata, filters species/recordings,
and downloads MP3 files to Google Drive, ready for mel-spectrogram and tensor generation.

The main execution entrypoint is [pre-processing.ipynb](pre-processing.ipynb).

---

## Pipeline

```
Setup/Auth  →  Mount Drive  →  Fetch metadata  →  Filter species  →  Download audio  →  Mel-spectrogram  →  Tensor export
          ✅              ✅               ✅                 ✅                ✅                 🔜                🔜
```

---

## Project Structure

```
colombian-bird-song-identifier/
│
├── data/                               # Local/legacy data artifacts
│   ├── raw/                            # Raw API responses (one JSON per page)
│   ├── recordings/                     # Downloaded audio (if running locally)
│   ├── colombia_birds_by_species.json  # Grouped metadata output
│   └── colombia_birds_filtered.json    # Filtered metadata output
│
├── pre-processing.ipynb                # Main Colab workflow (active)
├── PREPROCESSING_CONTEXT.md            # Plan for mel/tensor phases
├── fetch_metadata.py.retired           # Legacy script (retired)
├── filter_species.py.retired           # Legacy script (retired)
├── requirements.txt                    # Local Python dependencies
└── .env                                # Optional local env file (gitignored)
```

---

## Quick Start (Colab)

1. Open [pre-processing.ipynb](pre-processing.ipynb) in Google Colab.
2. Run Cell 1 and Cell 2.
3. In Cell 3, paste your Xeno-Canto API key when prompted.

Generate your key from: https://xeno-canto.org/account/api

Important: you must provide your own key. The notebook cannot access your account or generate keys.

### Colab in VS Code-based IDEs

You can also run this notebook from VS Code-based IDEs that support Google Colab notebooks.

- Open [pre-processing.ipynb](pre-processing.ipynb) from your IDE and connect it to a Colab runtime.
- Use the same execution order as in Colab (Cell 1, Cell 2, then Cell 3 for API key input).
- Keep in mind that local file-upload dialogs may differ by IDE integration, but this notebook avoids that flow by fetching metadata directly in runtime.

---

## Notebook Outputs

- Temporary runtime metadata:
     - /content/data/raw/page_NNNN.json
     - /content/data/colombia_birds_by_species.json
     - /content/data/colombia_birds_filtered.json
- Persistent audio output on Drive:
     - /content/drive/MyDrive/bird_songs/<Species_Name>/XC{id}.mp3

---

## Local Setup (Optional)

**Prerequisites:** [Python 3.12](https://www.python.org/) and [uv](https://github.com/astral-sh/uv).

```bash
# Clone the repository
git clone https://github.com/mrobayoba/colombian-bird-song-identifier.git
cd colombian-bird-song-identifier

# Create a virtual environment and install dependencies
uv venv .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

Create a .env file in the project root with your Xeno-Canto API key:

```env
XC_API_KEY=your_key_here
```

---

## Current Status

- Implemented in notebook:
     - Step 1 Setup and API key input
     - Step 2 Drive mount and runtime path setup
     - Step 3 Metadata fetch and species grouping
     - Step 4 Metadata filtering
     - Step 5 Concurrent audio download to Drive
- Planned next:
     - Step 6 Automated chunking and mel-spectrogram generation
     - Step 7 Tensor export for model training

---

## Data Source

Recording metadata and audio are provided by [Xeno-Canto](https://xeno-canto.org), a platform for sharing
wildlife sounds from around the world, powered by the Xeno-Canto Foundation and Naturalis Biodiversity Center.

> Recordings are © their respective recordists. See the `license` field of each recording for terms of use.

---

## License

This project is licensed under MIT. See [LICENSE](LICENSE).

