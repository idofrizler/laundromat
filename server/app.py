"""Laundromat Inference Server - FastAPI server for sock pair detection."""

import os
import logging
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from inference_service import get_inference_service, InferenceResult

log_level = os.environ.get('LOG_LEVEL', 'info').upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Laundromat Inference Server...")
    service = get_inference_service()
    
    try:
        service.load_models()
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise
    
    yield
    
    logger.info("Shutting down server...")

app = FastAPI(
    title="Laundromat Inference Server",
    description="Sock pair detection using SAM3 and ResNet18",
    version="0.2.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve web client if it exists
WEB_CLIENT_PATHS = [
    '/app/static/client',
    os.path.join(os.path.dirname(__file__), '..', 'web-client'),
]
for client_path in WEB_CLIENT_PATHS:
    if os.path.exists(client_path):
        app.mount("/client", StaticFiles(directory=client_path, html=True), name="web-client")
        logger.info(f"Serving web client from: {client_path}")
        break

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Laundromat Inference Server",
        "version": "0.2.0",
        "endpoints": {
            "POST /infer": "Run inference on a frame",
            "GET /health": "Health check",
            "GET /client": "Web client (if available)"
        }
    }

@app.get("/health")
async def health_check():
    service = get_inference_service()
    return {
        "status": "healthy",
        "models_loaded": service.is_loaded()
    }

@app.post("/infer")
async def infer(
    frame: UploadFile = File(..., description="JPEG image of the frame to process"),
    top_n_pairs: int = Query(1, ge=1, le=10, description="Maximum number of pairs to detect"),
    detection_prompt: str = Query("socks", description="Text prompt for detection"),
    exclude_basket: bool = Query(False, description="Enable basket detection and sock exclusion")
):
    if frame.content_type not in ['image/jpeg', 'image/jpg', 'image/png']:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid content type: {frame.content_type}. Expected image/jpeg or image/png"
        )
    
    try:
        frame_bytes = await frame.read()
        
        if len(frame_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty image file")
        
        logger.debug(f"Received frame: {len(frame_bytes)} bytes")
        
        service = get_inference_service()
        result = service.infer_from_jpeg(
            frame_bytes,
            top_n_pairs=top_n_pairs,
            detection_prompt=detection_prompt,
            exclude_basket=exclude_basket
        )
        
        logger.info(
            f"Inference complete: {result.total_socks_detected} socks, "
            f"{len(result.pairs_data)//2} pairs, {result.inference_time_ms:.1f}ms"
        )
        
        return asdict(result)
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.exception("Inference error")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.get("/config")
async def get_config():
    service = get_inference_service()
    return {
        "model_path": service.model_path,
        "models_loaded": service.is_loaded(),
        "supported_formats": ["image/jpeg", "image/png"],
        "max_pairs": 10
    }

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8080))
    host = os.environ.get("HOST", "0.0.0.0")
    
    ssl_keyfile = os.environ.get("SSL_KEYFILE")
    ssl_certfile = os.environ.get("SSL_CERTFILE")
    
    ssl_args = {}
    if ssl_keyfile and ssl_certfile and os.path.exists(ssl_keyfile):
        ssl_args = {
            "ssl_keyfile": ssl_keyfile,
            "ssl_certfile": ssl_certfile
        }
        logger.info(f"HTTPS enabled with certificates")
    
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=True,
        log_level=log_level.lower(),
        **ssl_args
    )
