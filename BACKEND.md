# Backend — Serving Layer (build steps 1–3)

This directory tree adds the **multi-agent serving backend** on top of the
existing notebook pipeline. The notebooks (`pre-processing.ipynb`, `train.ipynb`)
remain the data + training pipeline; this backend serves the trained model.

It currently implements build-order steps 1–3:

1. **Shared contracts** (`libs/contracts`) — Pydantic models matching *your*
   data: `ClassificationResult` (underscored `label_map` labels, chunk-averaged),
   `BirdCard` (mirrors `colombian_birds.json` ornithological cards), `EnrichmentResult`.
2. **Local infra** (`infra/docker/docker-compose.dev.yml`) — Mongo, Redis,
   Elasticsearch, NATS, plus the Index Agent in stub mode.
3. **Bird Index Agent skeleton** (`services/bird-index-agent`) — working
   `POST /api/v1/identify`, every downstream stubbed.

## How it maps to your notebooks

| Notebook artifact | Backend component |
|---|---|
| `resnet50v2_fold{K}.keras` + `label_map.json` | `services/bird-identifier-agent` loads + serves it |
| Inference helper (train.ipynb Step 11) | `shared/preprocessing/audio.py` (single source of train/serve parity) |
| `colombian_birds.json` cards (pre Step 4.5) | seeded into Mongo via `infra/scripts/seed_cards.py`; served as `BirdCard` |
| chunk-averaged top-3 prediction | `ClassificationResult` with `alternatives` |

> Parity note: the inference helper uses a **1000 Hz** bandpass low cut while
> `PREPROCESSING_CONTEXT.md` says **500 Hz**. `shared/preprocessing/audio.py`
> follows the notebook (1000 Hz) so serving matches the trained model. Reconcile
> the doc when convenient.

## Layout (added)

```
libs/contracts/              # step 1
services/
  bird-index-agent/          # step 3 — orchestrator (stubbed downstreams)
  bird-identifier-agent/     # the ONLY ML agent — loads your .keras model
  audio-recorder-agent/      # preprocessing, reuses shared/preprocessing
  birddex-mcp-agent/         # LLM/MCP enrichment (skeleton)
shared/preprocessing/        # train/serve-parity audio pipeline
infra/docker/                # step 2 — docker-compose dev stack
infra/scripts/               # seed_cards.py, make_sample_wav.py
```

## Quickstart (stub mode — no model needed)

```bash
cd services/bird-index-agent
pip install -r requirements.txt
CBSI_USE_STUBS=true uvicorn app.main:app --reload --port 8000

# in another shell:
python ../../infra/scripts/make_sample_wav.py /tmp/sample.wav
curl -s -F "audio=@/tmp/sample.wav" http://localhost:8000/api/v1/identify | python3 -m json.tool
```

Or the full stack:

```bash
docker compose -f infra/docker/docker-compose.dev.yml up -d --build
```

## Tests

```bash
cd services/bird-index-agent && pytest -q
```

## Going live with the real model (later steps)

1. Seed your cards: `python infra/scripts/seed_cards.py --json colombian_birds.json --mongo mongodb://localhost:27017 --db cbsi`
2. Copy `resnet50v2_fold0.keras` + `label_map.json` to a `bird_models/` dir.
3. Uncomment `recorder` / `identifier` / `birddex-mcp` in the compose file and
   set `CBSI_USE_STUBS=false` on `index-agent`.
4. Re-run the same smoke test — same contract, now real predictions.
