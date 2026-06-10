FROM python:3.12-slim
WORKDIR /app

# Shared contracts (weights are NOT copied — they are mounted at runtime).
COPY libs/contracts /app/libs/contracts

# TensorFlow is large. Pin to the version that trained your weights.
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" numpy "tensorflow-cpu>=2.16,<2.17" && \
    pip install --no-cache-dir /app/libs/contracts

# app/ must contain BOTH main.py and model.py (BirdSongClassifier).
COPY services/bird-identifier-agent/app /app/app

# Point at where the mount makes the files appear inside the container.
ENV WEIGHTS_PATH=/models/fold2_best.weights.h5
ENV LABEL_MAP_PATH=/models/label_map.json
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]