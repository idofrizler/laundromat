#!/bin/bash
# Laundromat Server Startup Script
# Handles HTTPS when SSL certificates are available

PORT=${PORT:-8443}
HOST=${HOST:-0.0.0.0}

# Check if SSL certificates exist
if [ -f "${SSL_KEYFILE}" ] && [ -f "${SSL_CERTFILE}" ]; then
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
