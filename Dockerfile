# syntax=docker/dockerfile:1.6
# ---------------------------------------------------------------------------
# Chatbot image — FastAPI server in `langraph pipeline/`.
# Project layout has shared modules at the repo root (db_service.py,
# opensearch_service.py, embedding_service.py, models.py, config.py) which
# the chatbot imports via _bootstrap.py adding the parent dir to sys.path.
# We therefore COPY the full project into /app and run uvicorn from
# `/app/langraph pipeline` (the dir-with-space is intentional, _bootstrap
# already handles it).
# ---------------------------------------------------------------------------
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# curl is used by the healthcheck below; libpq is already bundled in
# psycopg2-binary wheels so no postgres-dev needed.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy the codebase. In dev (docker-compose) this layer is shadowed by a
# bind-mount so changes on the host hot-reload inside the container.
COPY . .

WORKDIR "/app/langraph pipeline"

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=5 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
