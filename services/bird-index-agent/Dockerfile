FROM python:3.12-slim
WORKDIR /app
COPY libs/contracts /app/libs/contracts
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" httpx python-multipart \
    "pydantic>=2.6" motor redis && \
    pip install --no-cache-dir /app/libs/contracts
COPY services/bird-index-agent/app /app/app
ENV CBSI_USE_STUBS=true
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
