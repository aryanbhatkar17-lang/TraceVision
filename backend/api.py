"""
TraceVision API — Search, Smooth & Clip Pipeline
=================================================
SIH-2026 Hackathon Project

This is the "Friend's" integration layer. It:

1. POST /api/search  — Accepts a text query + video_id, runs the existing
   VideoChunkProcessor + SemanticVideoAuditor pipeline, feeds the per-frame
   confidence scores through TemporalSmoother (teammate's algorithm), and
   returns clean, continuous ClipRange objects.

2. POST /api/clip    — Accepts a video_id + start/end timestamps, uses
   ClipCutter (FFmpeg / OpenCV fallback) to extract the MP4 sub-clip, and
   streams it back to the client.

3. GET  /api/hardware — Returns a real-time ResourceSnapshot of CPU, RAM,
   and GPU VRAM usage (enforcing the 6 GB ceiling).

All existing endpoints from server.py continue to work — they are mounted
as a sub-application on the same FastAPI instance.
"""

import os
import sys
import uuid
import json
import logging
import tempfile
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Query, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Ensure backend/ is importable regardless of CWD
# ---------------------------------------------------------------------------
_backend_dir = Path(__file__).resolve().parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

# Import teammate's temporal smoother
from smoothing import TemporalSmoother, ClipRange

# Import existing pipeline components
from server import (
    VideoChunkProcessor,
    SemanticVideoAuditor,
    MatchItem,
    AuditResponseSchema,
    format_timestamp,
    UPLOAD_DIR,
    app as server_app,
)

# Import our new infrastructure
from hardware_monitor import HardwareMonitor, add_hardware_headers_middleware
from clip_cutter import ClipCutter

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("tracevision-api")

# ---------------------------------------------------------------------------
# Application Instance
# ---------------------------------------------------------------------------
api = FastAPI(
    title="TraceVision API — Search, Smooth & Clip",
    description=(
        "Unified API that integrates the Sentinel video pipeline with "
        "temporal smoothing and FFmpeg clip extraction. "
        "Designed for the SIH-2026 hackathon with a strict 6 GB VRAM ceiling."
    ),
    version="2.0.0",
)

ALLOWED_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(",")

api.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Infrastructure Singletons
# ---------------------------------------------------------------------------
monitor = HardwareMonitor(vram_ceiling_mb=6144.0, warning_threshold=0.80)
cutter = ClipCutter(output_dir=UPLOAD_DIR / "clips")
smoother = TemporalSmoother(
    threshold=0.75,
    tolerance_window=2,
    min_clip_duration=3.0,
    frame_gap_limit=2.0,
)
processor = VideoChunkProcessor(chunk_duration_sec=60.0, sample_fps=1.0)
auditor = SemanticVideoAuditor()

# Attach hardware monitoring middleware
add_hardware_headers_middleware(api, monitor)

# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------
class SearchRequest(BaseModel):
    query: str = Field(..., description="Natural-language text search query")
    video_id: str = Field(..., description="Filename of the uploaded video")
    top_k: int = Field(default=10, ge=1, le=50, description="Max number of clip results")


class SmoothedClip(BaseModel):
    """A continuous clip range after temporal smoothing."""
    clip_id: str
    start: float
    end: float
    duration: float
    frame_count: int
    avg_confidence: float
    bridged_count: int
    start_time: str
    end_time: str


class SearchResponse(BaseModel):
    """Response from the search + smooth pipeline."""
    query: str
    video_id: str
    video_duration: float
    total_chunks: int
    raw_match_count: int
    smoothed_clips: List[SmoothedClip]
    diagnostics: dict
    # Also include raw matches for backward-compatible frontend rendering
    matches: List[dict]


class ClipRequest(BaseModel):
    video_id: str = Field(..., description="Filename of the uploaded video")
    start: float = Field(..., ge=0.0, description="Clip start time in seconds")
    end: float = Field(..., gt=0.0, description="Clip end time in seconds")


class ClipResponse(BaseModel):
    clip_id: str
    source_video: str
    start_sec: float
    end_sec: float
    duration_sec: float
    method: str
    output_path: str
    file_size_bytes: int
    download_url: str


# ---------------------------------------------------------------------------
# Core Pipeline: Search → Analyze → Smooth
# ---------------------------------------------------------------------------
async def _run_search_pipeline(
    video_id: str, query: str, top_k: int = 10
) -> dict:
    """
    Full pipeline:
      1. Locate the uploaded video
      2. Chunk it (60s segments) & extract keyframes at 1 fps
      3. Run semantic auditor on each chunk → raw MatchItems
      4. Convert match confidence into (timestamp, score) pairs
      5. Feed through TemporalSmoother → ClipRange list
      6. Return everything in a SearchResponse
    """
    # --- Step 1: Locate video ---
    video_path = UPLOAD_DIR / video_id
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video '{video_id}' not found in storage.")

    # --- VRAM safety check ---
    if not monitor.is_safe_to_proceed(required_mb=500):
        logger.warning("VRAM ceiling exceeded — refusing to process.")
        raise HTTPException(
            status_code=503,
            detail="System resources exceeded. VRAM usage is near the 6 GB ceiling. Try again later."
        )
    monitor.log_warning_if_high()

    # --- Step 2: Chunk & extract keyframes ---
    meta = processor.inspect_and_chunk(str(video_path), video_id)
    chunks = meta["chunks"]
    logger.info("Video '%s': %.1fs, %d chunk(s)", video_id, meta['duration'], len(chunks))

    # --- Step 3: Run auditor per chunk ---
    all_matches: List[MatchItem] = []
    frame_scores: List[tuple] = []

    for chunk in chunks:
        sampled = processor.sample_and_compress_chunk_keyframes(
            str(video_path), chunk.start_second, chunk.end_second
        )

        # Run semantic analysis
        chunk_matches = await auditor.analyze_chunk(chunk, sampled, query)
        all_matches.extend(chunk_matches)

        # --- Step 4: Build (timestamp, confidence) pairs ---
        # Use motion scores from sampled keyframes as a proxy for vector
        # similarity confidence (in production, this comes from the vector DB).
        for frame_info in sampled:
            ts = frame_info["timestamp_sec"]
            # Normalize motion score to [0, 1] range as a proxy confidence
            motion = frame_info.get("motion_score", 0.0)
            # Heuristic: high motion → high relevance to query
            # Clamp to [0.0, 1.0]; motion_score > 15 → confident detection
            proxy_confidence = min(1.0, max(0.0, motion / 20.0))
            frame_scores.append((ts, proxy_confidence))

        # Also inject match confidence scores at match timestamps
        for m in chunk_matches:
            mid_ts = (m.start_seconds + m.end_seconds) / 2.0
            frame_scores.append((mid_ts, m.confidence or 0.9))

    # Sort by timestamp for the smoother
    frame_scores.sort(key=lambda x: x[0])

    # --- Step 5: Temporal smoothing ---
    diagnostics = smoother.get_diagnostics(frame_scores)
    smoothed_clips = diagnostics.get("clips", [])

    # --- Step 6: Build response ---
    clip_responses = []
    for idx, c in enumerate(smoothed_clips[:top_k]):
        clip_responses.append(SmoothedClip(
            clip_id=f"smooth-{idx + 1:03d}",
            start=c["start"],
            end=c["end"],
            duration=c["duration"],
            frame_count=c["frame_count"],
            avg_confidence=c["avg_confidence"],
            bridged_count=c["bridged_count"],
            start_time=format_timestamp(c["start"]),
            end_time=format_timestamp(c["end"]),
        ))

    # Build backward-compatible matches for the existing frontend
    compat_matches = [m.model_dump() for m in all_matches]

    return SearchResponse(
        query=query,
        video_id=video_id,
        video_duration=meta["duration"],
        total_chunks=len(chunks),
        raw_match_count=len(all_matches),
        smoothed_clips=clip_responses,
        diagnostics={
            "total_frames": diagnostics.get("total_frames", 0),
            "above_threshold": diagnostics.get("above_threshold", 0),
            "bridged_frames": diagnostics.get("bridged_frames", 0),
            "dropped_frames": diagnostics.get("dropped_frames", 0),
            "smoother_config": diagnostics.get("config", {}),
        },
        matches=compat_matches,
    ).model_dump()


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@api.get("/api/health")
async def health_check():
    """Health check including hardware status."""
    snapshot = monitor.check_resources()
    return {
        "status": "healthy",
        "service": "tracevision-api",
        "version": "2.0.0",
        "storage_path": str(UPLOAD_DIR),
        "ffmpeg_available": cutter.ffmpeg_available,
        "hardware": snapshot.to_dict(),
    }


@api.post("/api/search")
async def search_video(request: SearchRequest):
    """
    Text search → vector similarity → temporal smoothing → smoothed clips.

    Accepts a natural-language query and a video_id (from prior upload).
    Returns smoothed, continuous clip ranges with confidence scores.
    """
    logger.info("Search request: query='%s', video_id='%s'", request.query, request.video_id)
    result = await _run_search_pipeline(request.video_id, request.query, request.top_k)
    return result


@api.post("/api/search/form")
async def search_video_form(
    query: str = Form(...),
    video_id: str = Form(...),
    top_k: int = Form(10),
):
    """
    Form-encoded variant of /api/search for browser/multipart clients.
    """
    logger.info("Search (form) request: query='%s', video_id='%s'", query, video_id)
    result = await _run_search_pipeline(video_id, query, top_k)
    return result


@api.post("/api/clip")
async def cut_clip(request: ClipRequest):
    """
    Cut an MP4 sub-clip from a previously uploaded video.

    Uses FFmpeg stream copy (zero VRAM) with automatic fallback to
    re-encoding or OpenCV if needed.
    """
    video_path = UPLOAD_DIR / request.video_id
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video '{request.video_id}' not found.")

    if request.start >= request.end:
        raise HTTPException(status_code=400, detail="'start' must be less than 'end'.")

    logger.info("Clip request: video='%s', %.1fs -> %.1fs", request.video_id, request.start, request.end)

    clip_id = f"clip_{uuid.uuid4().hex[:8]}"
    output_filename = f"{clip_id}_{request.start:.1f}s_{request.end:.1f}s.mp4"

    result = cutter.cut_clip(
        source_path=str(video_path),
        start_sec=request.start,
        end_sec=request.end,
        output_filename=output_filename,
    )

    if not result.success:
        raise HTTPException(
            status_code=500,
            detail=f"Clip extraction failed ({result.method}): {result.error}"
        )

    return ClipResponse(
        clip_id=clip_id,
        source_video=request.video_id,
        start_sec=result.start_sec,
        end_sec=result.end_sec,
        duration_sec=result.duration_sec,
        method=result.method,
        output_path=result.output_path,
        file_size_bytes=result.file_size_bytes,
        download_url=f"/api/clip/download/{output_filename}",
    ).model_dump()


@api.get("/api/clip/download/{filename}")
async def download_clip(filename: str):
    """Serve a previously cut clip as a downloadable MP4 file."""
    clip_path = UPLOAD_DIR / "clips" / filename
    if not clip_path.exists():
        raise HTTPException(status_code=404, detail=f"Clip '{filename}' not found.")

    return FileResponse(
        path=str(clip_path),
        media_type="video/mp4",
        filename=filename,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


@api.get("/api/hardware")
async def hardware_status():
    """Real-time hardware resource snapshot with VRAM ceiling enforcement."""
    snapshot = monitor.check_resources()
    return {
        "snapshot": snapshot.to_dict(),
        "vram_ceiling_mb": monitor.vram_ceiling_mb,
        "safe_to_proceed": monitor.is_safe_to_proceed(),
        "ffmpeg_available": cutter.ffmpeg_available,
    }


@api.post("/api/clip/batch")
async def cut_clips_batch(
    video_id: str = Form(...),
    clips_json: str = Form(..., description='JSON array of {"start": float, "end": float}'),
):
    """
    Batch clip extraction — cuts multiple sub-clips from a single video.
    Accepts a JSON array of start/end pairs.
    """
    video_path = UPLOAD_DIR / video_id
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video '{video_id}' not found.")

    try:
        clips_data = json.loads(clips_json)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in 'clips_json'.")

    results = cutter.cut_clips_batch(str(video_path), clips_data)

    return {
        "video_id": video_id,
        "total_requested": len(clips_data),
        "successful": sum(1 for r in results if r.success),
        "failed": sum(1 for r in results if not r.success),
        "clips": [r.to_dict() for r in results],
    }


# ---------------------------------------------------------------------------
# Mount existing server.py endpoints
# ---------------------------------------------------------------------------
api.mount("/legacy", server_app)

# Also proxy the original endpoints at root level for backward compatibility
# The existing server.py endpoints: /api/health, /api/upload, /api/analyze, /api/analyze/stream
# are available at /legacy/api/...
# We re-expose /api/upload and /api/analyze from the original server
from server import upload_video, analyze_video, analyze_video_stream

api.post("/api/upload")(upload_video)
api.post("/api/analyze")(analyze_video)
api.get("/api/analyze/stream")(analyze_video_stream)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    logger.info("=" * 60)
    logger.info("  TraceVision API v2.0 — Search, Smooth & Clip Pipeline")
    logger.info("=" * 60)
    logger.info("  Storage:  %s", UPLOAD_DIR)
    logger.info("  FFmpeg:   %s", "Available" if cutter.ffmpeg_available else "NOT FOUND (OpenCV fallback)")
    logger.info("  VRAM Cap: %.0f MB", monitor.vram_ceiling_mb)
    logger.info("=" * 60)

    uvicorn.run(api, host="0.0.0.0", port=8000)
