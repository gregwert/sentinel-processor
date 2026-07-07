FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml .
COPY sentinel_backend/ ./sentinel_backend/
RUN pip install --no-cache-dir .

EXPOSE 8000
CMD ["uvicorn", "sentinel_backend.api.main:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
