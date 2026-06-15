# ── BirdDex MCP Agent (LLM enrichment via Google Gemini) ─────────────────────
# Invoked ONLY by the Index Agent. Grounds Gemini on the trusted card data and
# returns a Spanish summary + tags. Falls back to a template if the API key is
# absent or the call fails.
#
# Build context MUST be the repo root:
#   docker build -f services/birddex-mcp-agent/Dockerfile -t cbsi-birddex .
FROM python:3.12-slim
WORKDIR /app

COPY libs/contracts /app/libs/contracts

RUN pip install --no-cache-dir \
    fastapi "uvicorn[standard]" httpx motor anyio "google-genai>=1.0" && \
    pip install --no-cache-dir /app/libs/contracts

COPY services/birddex-mcp-agent/app /app/app

ENV MONGO_URI=mongodb://mongo:27017
ENV MONGO_DB=cbsi
ENV GEMINI_MODEL=gemini-2.5-flash
# GEMINI_API_KEY is provided at runtime (compose env_file / -e), NOT baked in.
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
