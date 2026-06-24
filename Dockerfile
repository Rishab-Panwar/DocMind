# Use official Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set workdir
WORKDIR /app

# Install OS dependencies
RUN apt-get update && apt-get install -y build-essential poppler-utils && rm -rf /var/lib/apt/lists/*

# Copy requirements and install first (cached layer — only rebuilds when requirements.txt changes)
COPY requirements.txt setup.py ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files (separate layer so code changes don't trigger pip reinstall)
COPY . .

# Entrypoint writes the GCP SA key (from env) and starts uvicorn in prod mode.
RUN chmod +x /app/entrypoint.sh

# Expose port
EXPOSE 8080

# Production start (no --reload). Entrypoint handles GCP creds + uvicorn workers.
CMD ["/app/entrypoint.sh"]
