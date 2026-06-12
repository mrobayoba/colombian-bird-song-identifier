FROM python:3.12-slim
WORKDIR /app
COPY libs/contracts /app/libs/contracts
COPY services/bird-identifier-agent/requirements.txt /app/req.txt
# tensorflow is large; pin to the version that trained your .keras model
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" numpy "tensorflow>=2.16" && \
    pip install --no-cache-dir /app/libs/contracts
COPY services/bird-identifier-agent/app /app/app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
