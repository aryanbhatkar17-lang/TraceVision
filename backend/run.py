#!/usr/bin/env python3
"""
Sentinel Backend — Entrypoint
==============================
Starts the FastAPI server with all configuration documented.

Usage:
    cd backend
    python run.py

Environment Variables:
    CORS_ORIGINS          Comma-separated allowed origins (default: http://localhost:3000)
    BRIGHTNESS_THRESHOLD  Mean brightness below which frames get enhanced (default: 90.0)
    MOTION_THRESHOLD      Minimum motion score for event detection (default: 8.0)
    CHUNK_DURATION_SEC    Video chunk length in seconds (default: 60.0)
    SAMPLE_FPS            Frame sampling rate (default: 1.0)
    TARGET_WIDTH          Downscale target width (default: 640)
    TARGET_HEIGHT         Downscale target height (default: 360)
    JPEG_QUALITY          Keyframe JPEG quality 0-100 (default: 75)
    ZERO_DCE_WEIGHTS      Path to Zero-DCE model weights (default: auto-detect)
    ZERO_DCE_DEVICE       Compute device: cpu / cuda / mps (default: auto-detect)
"""

import os
import sys
import logging

# Ensure backend/ is on the path for sibling imports
_backend_dir = os.path.dirname(os.path.abspath(__file__))
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sentinel-run")

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))

logger.info("=" * 60)
logger.info("  Sentinel Video Audit Backend")
logger.info("=" * 60)
logger.info("  Host:           %s:%d", HOST, PORT)
logger.info("  CORS Origins:   %s", os.environ.get("CORS_ORIGINS", "http://localhost:3000"))
logger.info("  Brightness:     %s", os.environ.get("BRIGHTNESS_THRESHOLD", "90.0"))
logger.info("  Motion:         %s", os.environ.get("MOTION_THRESHOLD", "8.0"))
logger.info("  Chunk Duration: %ss", os.environ.get("CHUNK_DURATION_SEC", "60.0"))
logger.info("  Sample FPS:     %s", os.environ.get("SAMPLE_FPS", "1.0"))
logger.info("  Target Res:     %sx%s", os.environ.get("TARGET_WIDTH", "640"), os.environ.get("TARGET_HEIGHT", "360"))
logger.info("  JPEG Quality:   %s", os.environ.get("JPEG_QUALITY", "75"))
logger.info("  Model Weights:  %s", os.environ.get("ZERO_DCE_WEIGHTS", "auto-detect"))
logger.info("  Device:         %s", os.environ.get("ZERO_DCE_DEVICE", "auto-detect"))
logger.info("=" * 60)

if __name__ == "__main__":
    uvicorn.run("server:app", host=HOST, port=PORT, reload=True)
