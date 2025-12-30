#!/bin/bash
# Laundromat Server Startup Script
# Handles HTTPS when SSL certificates are available

# Change to the server directory (where this script is located)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Set PYTHONPATH to include the laundromat package
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH}"

# Default config
PORT=${PORT:-8443}
HOST=${HOST:-0.0.0.0}
MODEL_PATH=${MODEL_PATH:-models/sam3.pt}
export MODEL_PATH

# Check if SSL certificates exist
if [ -f "${SSL_KEYFILE:-certs/key.pem}" ] && [ -f "${SSL_CERTFILE:-certs/cert.pem}" ]; then
    SSL_KEYFILE="${SSL_KEYFILE:-certs/key.pem}"
    SSL_CERTFILE="${SSL_CERTFILE:-certs/cert.pem}"
    echo "Starting server with HTTPS on port ${PORT}..."
    exec uvicorn app:app \
        --host "${HOST}" \
        --port "${PORT}" \
        --ssl-keyfile "${SSL_KEYFILE}" \
        --ssl-certfile "${SSL_CERTFILE}"
else
    echo "No SSL certificates found, starting HTTP on port ${PORT}..."
    exec uvicorn app:app \
        --host "${HOST}" \
        --port "${PORT}"
fi
