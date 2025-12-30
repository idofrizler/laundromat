# Laundromat Inference Server
# Multi-stage build for smaller image size

FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for layer caching
COPY server/requirements.txt ./requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the laundromat package
COPY src/laundromat /app/laundromat

# Copy server code
COPY server/app.py server/inference_service.py ./

# Copy web client files
COPY web-client /app/static/client

# Copy startup script
COPY server/start.sh ./
RUN chmod +x start.sh

# Create models directory
RUN mkdir -p /app/models

# Expose ports (HTTP and HTTPS)
EXPOSE 8080 8443

# Health check - works for both HTTP and HTTPS
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -fsk https://localhost:${PORT:-8443}/health || curl -f http://localhost:${PORT:-8080}/health || exit 1

# Default command - uses start.sh which handles SSL
CMD ["./start.sh"]
