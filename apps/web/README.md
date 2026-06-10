# Cantos — Web Frontend (React PWA)

The mobile-first frontend for the Colombian Bird Song Identifier. Records a
bird's song in the browser, sends it to the Bird Index Agent, and shows the
identified species with its conservation status and description.

It is a **PWA** — installable on Android, iOS, and desktop from the browser,
no app store required.

## What it does

1. Records audio with the device microphone (live waveform while recording).
2. Enforces a 3-second minimum (the model needs >3s to extract a chunk).
3. POSTs the recording to `POST /api/v1/identify` on the Index Agent.
4. Renders the `UnifiedResponse`: top species, confidence, alternatives,
   conservation status, image, and the Spanish description/enrichment.
5. Degrades gracefully — if `card` or `enrichment` is missing, it shows a
   quiet "no disponible" note instead of breaking.

## Prerequisites

- Node.js 18+ and npm
- The Bird Index Agent running on `http://localhost:8000`

## Run it

```bash
cd apps/web
npm install
npm run dev
```

Open `http://localhost:5173`. The dev server proxies `/api` to
`http://localhost:8000`, so there are no CORS issues in development.

> Microphone access requires a secure context. `localhost` counts as secure,
> so the mic works in dev. In production you must serve over HTTPS.

## Build for production

```bash
npm run build      # outputs to apps/web/dist
npm run preview    # serve the production build locally to test
```

Deploy the `dist/` folder to any static host (Netlify, Vercel, Cloudflare
Pages, S3+CloudFront). Set the API origin at build time:

```bash
VITE_API_BASE=https://api.your-domain.co npm run build
```

## How it connects to the backend

| Frontend | Backend |
|---|---|
| `src/lib/types.ts` | mirrors `libs/contracts/contracts/models.py` |
| `src/lib/api.ts` → `POST /api/v1/identify` | `bird-index-agent` `identify` endpoint |
| multipart field name `audio` | `audio: UploadFile = File(...)` |
| reads `degraded`, `card`, `enrichment` | `UnifiedResponse` |

The CORS allow-list in the Index Agent already includes
`http://localhost:5173`, so the production build works once you add your
deployed origin there too.

## Project layout

```
apps/web/
├── index.html
├── vite.config.ts          # PWA + dev proxy to :8000
├── src/
│   ├── main.tsx            # entry
│   ├── App.tsx             # record → loading → result state machine
│   ├── App.css             # visual identity
│   ├── components/
│   │   ├── Waveform.tsx    # live audio visualization (signature element)
│   │   └── ResultCard.tsx  # identification result UI
│   ├── lib/
│   │   ├── api.ts          # fetch client for the Index Agent
│   │   ├── types.ts        # TypeScript mirror of the contract
│   │   └── useRecorder.ts  # MediaRecorder hook + live amplitude
│   └── styles/global.css   # design tokens
└── public/favicon.svg
```
