FROM python:3.12-slim
WORKDIR /app
COPY libs/contracts /app/libs/contracts
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" httpx motor && \
    pip install --no-cache-dir /app/libs/contracts
COPY services/birddex-mcp-agent/app /app/app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
