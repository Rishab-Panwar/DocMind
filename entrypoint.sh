#!/usr/bin/env bash
set -e

# The GCP service-account JSON is injected as the GCP_SA_JSON env var (from AWS
# Secrets Manager). Write it to a file so the Vertex AI client can authenticate.
if [ -n "$GCP_SA_JSON" ]; then
  printf '%s' "$GCP_SA_JSON" > /app/gcp-vertex-sa.json
  export GOOGLE_APPLICATION_CREDENTIALS=/app/gcp-vertex-sa.json
fi

# Production server: no --reload, a couple of workers for concurrency.
exec uvicorn api.main:app --host 0.0.0.0 --port 8080 --workers "${UVICORN_WORKERS:-2}"
