"""
TraceVision API — Search, Smooth & Clip Pipeline
=================================================
SIH-2026 Hackathon Project
"""

import os
import sys
import uuid
import json
import asyncio
import logging
import tempfile

from pathlib import Path
from typing import Optional, List
from pipeline import _run_search_pipeline
from fastapi import FastAPI, HTTPException, Query, Form, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
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
    version="2.1.0",
)

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Infrastructure Singletons
# ---------------------------------------------------------------------------
monitor = HardwareMonitor(vram_ceiling_mb=6144.0, warning_threshold=0.80)
cutter = ClipCutter(output_dir=UPLOAD_DIR / "clips")
smoother = TemporalSmoother(
    threshold=0.01,
    tolerance_window=2,
    min_clip_duration=0.0,
    frame_gap_limit=5.0,
)
processor = VideoChunkProcessor(chunk_duration_sec=60.0, sample_fps=1.0)
auditor = SemanticVideoAuditor()

add_hardware_headers_middleware(api, monitor)

# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------
class SearchRequest(BaseModel):
    query: str = Field(..., description="Natural-language text search query")
    video_id: str = Field(..., description="Filename of the uploaded video")
    top_k: int = Field(default=10, ge=1, le=50, description="Max number of clip results")


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
# Bridge: real pipeline output -> frontend-compatible response shape
# ---------------------------------------------------------------------------
def _pipeline_result_to_response(video_id: str, pipeline_result: dict) -> dict:
    diagnostics = pipeline_result.get("clips") or {}
    smoothed_clips = diagnostics.get("clips", [])

    # pipeline.py's "matches" already carries Gemini's real per-frame
    # reasoning (frontend_matches), keyed by timestamp. Use it to give each
    # smoothed clip a genuine AI justification instead of a generic label —
    # this is the evidence an officer actually needs to trust a hit.
    per_frame = pipeline_result.get("matches") or []

    matches = []
    for idx, c in enumerate(smoothed_clips):
        start_sec = c["start"]
        end_sec = c["end"]

        # If it's a single isolated frame, expand it into a 2-second snippet
        if start_sec == end_sec:
            end_sec += 2.0

        # Find the frame(s) whose timestamp falls inside this clip's range,
        # pick the highest-confidence one's reasoning as the representative
        # description.
        frames_in_range = [
            f for f in per_frame
            if start_sec - 0.5 <= f["start_seconds"] <= end_sec + 0.5
        ]
        if frames_in_range:
            best = max(frames_in_range, key=lambda f: f.get("confidence", 0))
            description = best.get("description") or "Verified continuous event."
        else:
            description = "Verified continuous event."

        matches.append({
            "id": f"match-{idx + 1:03d}",
            "start_time": format_timestamp(start_sec),
            "end_time": format_timestamp(end_sec),
            "start_seconds": round(start_sec, 2),
            "end_seconds": round(end_sec, 2),
            "category": "SECURITY",
            "description": description,
            "confidence": round(c.get("avg_confidence", 0.0), 3),
        })

    return {
        "status": "completed",
        "progress": 100,
        "message": f"Audit complete. Found {len(matches)} continuous event(s).",
        "matches": matches,
        "video_id": video_id,
    }


async def _resolve_video_path(video_id: Optional[str], file: Optional[UploadFile]) -> tuple[str, str]:
    """Shared helper: locate an already-uploaded video, or save a fresh upload."""
    if video_id:
        target_path = UPLOAD_DIR / video_id
        if not target_path.exists():
            raise HTTPException(status_code=404, detail=f"Video '{video_id}' not found.")
        return str(target_path), video_id
    elif file:
        from server import upload_video
        upload_res = await upload_video(file)
        return upload_res["path"], upload_res["video_id"]
    else:
        raise HTTPException(status_code=422, detail="Either 'video_id' or 'file' must be provided.")


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@api.get("/api/health")
async def health_check():
    snapshot = monitor.check_resources()
    return {
        "status": "healthy",
        "service": "tracevision-api",
        "version": "2.1.0",
        "storage_path": str(UPLOAD_DIR),
        "ffmpeg_available": cutter.ffmpeg_available,
        "hardware": snapshot.to_dict(),
    }


@api.post("/api/search")
async def search_video(request: SearchRequest):
    """
    THE REAL PIPELINE ENTRY POINT.
    Text search -> CLIP retrieval -> Gemini validation -> smoothed clips.
    """
    logger.info(f"[/api/search] query='{request.query}', video_id='{request.video_id}'")

    video_path = UPLOAD_DIR / request.video_id
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video '{request.video_id}' not found.")

    if not monitor.is_safe_to_proceed(required_mb=500):
        raise HTTPException(status_code=503, detail="VRAM ceiling exceeded. Try again later.")

    # Correct call: (video_path, query, smoother) — NOT top_k in the 3rd slot.
    pipeline_result = await _run_search_pipeline(str(video_path), request.query, smoother)

    response = _pipeline_result_to_response(request.video_id, pipeline_result)
    response["matches"] = response["matches"][: request.top_k]
    return response


@api.post("/api/search/form")
async def search_video_form(
    query: str = Form(...),
    video_id: str = Form(...),
    top_k: int = Form(10),
):
    logger.info(f"[/api/search/form] query='{query}', video_id='{video_id}'")

    video_path = UPLOAD_DIR / video_id
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video '{video_id}' not found.")

    pipeline_result = await _run_search_pipeline(str(video_path), query, smoother)
    response = _pipeline_result_to_response(video_id, pipeline_result)
    response["matches"] = response["matches"][:top_k]
    return response


@api.post("/api/analyze")
async def analyze_video_real(
    video_id: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    query: str = Form(...),
    duration: Optional[float] = Form(None),
    chunk_size: Optional[float] = Form(60.0),
):
    """
    THIS is almost certainly the endpoint your frontend's "Analyze" button
    is calling. It now runs the real CLIP + Gemini pipeline instead of the
    old keyword-matching mock in server.py.
    """
    target_path, resolved_video_id = await _resolve_video_path(video_id, file)
    logger.info(f"[/api/analyze] query='{query}', video_id='{resolved_video_id}'")

    if not monitor.is_safe_to_proceed(required_mb=500):
        raise HTTPException(status_code=503, detail="VRAM ceiling exceeded. Try again later.")

    pipeline_result = await _run_search_pipeline(target_path, query, smoother)
    return _pipeline_result_to_response(resolved_video_id, pipeline_result)


@api.api_route("/api/analyze/stream", methods=["GET", "POST"])
async def analyze_video_stream_real(
    video_id: str = Query(...),
    query: str = Query(...),
    chunk_size: Optional[float] = Query(60.0),
):
    """
    SSE variant, now backed by the real pipeline. Progress events are
    coarse (the pipeline itself doesn't emit granular progress), but the
    final payload is genuine CLIP+Gemini output, not the mock.
    """
    video_path = UPLOAD_DIR / video_id
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video ID '{video_id}' not found.")

    async def event_generator():
        try:
            yield f"data: {json.dumps({'status': 'extracting', 'progress': 15, 'message': 'Extracting frames at 1 fps...'})}\n\n"
            await asyncio.sleep(0.2)

            yield f"data: {json.dumps({'status': 'analyzing', 'progress': 35, 'message': 'Running CLIP retrieval...'})}\n\n"
            await asyncio.sleep(0.2)

            yield f"data: {json.dumps({'status': 'analyzing', 'progress': 65, 'message': 'Validating candidates with Gemini...'})}\n\n"

            pipeline_result = await _run_search_pipeline(str(video_path), query, smoother)

            yield f"data: {json.dumps({'status': 'aggregating', 'progress': 90, 'message': 'Smoothing verified timestamps...'})}\n\n"
            await asyncio.sleep(0.1)

            final_payload = _pipeline_result_to_response(video_id, pipeline_result)
            yield f"data: {json.dumps(final_payload)}\n\n"

        except Exception as e:
            logger.error(f"Error in analyze stream: {e}", exc_info=True)
            yield f"data: {json.dumps({'status': 'error', 'progress': 0, 'message': f'Processing error: {str(e)}'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@api.post("/api/clip")
async def cut_clip(request: ClipRequest):
    video_path = UPLOAD_DIR / request.video_id
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video '{request.video_id}' not found.")
    if request.start >= request.end:
        raise HTTPException(status_code=400, detail="'start' must be less than 'end'.")

    clip_id = f"clip_{uuid.uuid4().hex[:8]}"
    output_filename = f"{clip_id}_{request.start:.1f}s_{request.end:.1f}s.mp4"

    result = cutter.cut_clip(
        source_path=str(video_path),
        start_sec=request.start,
        end_sec=request.end,
        output_filename=output_filename,
    )
    if not result.success:
        raise HTTPException(status_code=500, detail=f"Clip extraction failed ({result.method}): {result.error}")

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
    clip_path = UPLOAD_DIR / "clips" / filename
    if not clip_path.exists():
        raise HTTPException(status_code=404, detail=f"Clip '{filename}' not found.")
    return FileResponse(
        path=str(clip_path),
        media_type="video/mp4",
        filename=filename,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@api.get("/api/hardware")
async def hardware_status():
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
    clips_json: str = Form(...),
):
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
# Mount existing server.py endpoints (kept for reference / legacy access only —
# NOT used by /api/analyze or /api/search anymore)
# ---------------------------------------------------------------------------
api.mount("/legacy", server_app)

from server import upload_video

api.post("/api/upload")(upload_video)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    logger.info("=" * 60)
    logger.info("  TraceVision API v2.1 — Real CLIP+Gemini Pipeline Wired In")
    logger.info("=" * 60)
    logger.info(f"  Storage:  {UPLOAD_DIR}")
    logger.info(f"  FFmpeg:   {'Available' if cutter.ffmpeg_available else 'NOT FOUND (OpenCV fallback)'}")
    logger.info(f"  VRAM Cap: {monitor.vram_ceiling_mb:.0f} MB")
    logger.info("=" * 60)

    uvicorn.run(api, host="0.0.0.0", port=8000)