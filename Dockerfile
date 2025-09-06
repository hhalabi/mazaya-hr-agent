# Cloud Run-ready Dockerfile for Chainlit app
# Minimal, production-lean image using Python 3.11 slim

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install system packages only if needed. Most deps have wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the app code
COPY . .

# Create non-root user for security
RUN useradd --create-home --uid 1001 appuser \
    && chown -R appuser:appuser /app
USER appuser

# Cloud Run provides PORT env var. Default to 8080 for local runs.
ENV PORT=8080
EXPOSE 8080

# Chainlit serves the app. Honor Cloud Run's $PORT.
CMD ["/bin/sh", "-lc", "chainlit run app.py --host 0.0.0.0 --port ${PORT:-8080}"]

