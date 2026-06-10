FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg && rm -rf /var/lib/apt/lists/*
COPY libs/contracts /app/libs/contracts
COPY shared /app/shared
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" numpy \
    librosa==0.11.0 soundfile==0.13.1 "scipy>=1.13" && \
    pip install --no-cache-dir /app/libs/contracts
COPY services/audio-recorder-agent/app /app/app
ENV PYTHONPATH=/app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


