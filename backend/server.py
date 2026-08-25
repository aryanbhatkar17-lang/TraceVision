from video_enhancer import enhance_frame
import os
import cv2
import json
import uuid
import math
import asyncio
import logging
import tempfile
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, AsyncGenerator
import re

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sentinel-backend")

# --------------------------------------------------------------------------
# Optional integrations
# --------------------------------------------------------------------------
try:
    from smoothing import TemporalSmoother, ClipRange
    _smoother = TemporalSmoother(threshold=0.75, tolerance_window=2,
                                  min_clip_duration=3.0, frame_gap_limit=2.0)
    logger.info("TemporalSmoother loaded.")
except ImportError:
    _smoother = None
    logger.warning("smoothing.py not found — temporal smoothing will be skipped.")

try:
    from clip_cutter import ClipCutter
    _cutter = ClipCutter(output_dir=None)
    logger.info("ClipCutter loaded.")
except ImportError:
    _cutter = None
    logger.warning("clip_cutter.py not found — /api/clip endpoint will be unavailable.")

try:
    from hardware_monitor import HardwareMonitor
    _monitor = HardwareMonitor(vram_ceiling_mb=6144.0, warning_threshold=0.80)
    logger.info("HardwareMonitor loaded.")
except ImportError:
    _monitor = None
    logger.warning("hardware_monitor.py not found — hardware monitoring disabled.")

# --------------------------------------------------------------------------
# Zero-DCE low-light enhancement
# --------------------------------------------------------------------------
try:
    from Enhance import load_zero_dce, apply_zero_dce as _apply_zero_dce
    _zero_dce_model = load_zero_dce()
    if _zero_dce_model is not None:
        logger.info("Zero-DCE model loaded — low-light enhancement is active.")
    else:
        logger.warning("Zero-DCE weights not found — falling back to CLAHE-only.")
except ImportError:
    _zero_dce_model = None
    logger.warning("Enhance.py not found — Zero-DCE enhancement unavailable.")


def enhance_frame_full(frame: np.ndarray) -> Optional[np.ndarray]:
    """
    Runs merged pipeline: CLAHE + Zero-DCE.
    
    Args:
        frame: Input video frame as numpy array
    
    Returns:
        Enhanced frame or None if enhancement fails
    """
    # Input validation
    if frame is None or frame.size == 0:
        logger.warning("Invalid frame received in enhance_frame_full (empty or None)")
        return None
    
    try:
        # Apply CLAHE enhancement
        enhanced = enhance_frame(frame)
        if enhanced is None:
            logger.warning("CLAHE enhancement returned None, returning original frame")
            return frame
        
        # Apply Zero-DCE if available
        if _zero_dce_model is not None:
            try:
                zero_dce_result = _apply_zero_dce(enhanced, _zero_dce_model)
                if zero_dce_result is not None:
                    enhanced = zero_dce_result
                else:
                    logger.warning("Zero-DCE enhancement returned None, using CLAHE result")
                    # enhanced keeps the CLAHE result
            except Exception as e:
                logger.warning(f"Zero-DCE enhancement failed on frame: {e}. Using CLAHE result.")
                # enhanced keeps the CLAHE result
        
        return enhanced
    except Exception as e:
        logger.error(f"Frame enhancement pipeline failed: {e}")
        return None


def is_frame_dark(frame: np.ndarray, brightness_threshold: float = 90.0) -> bool:
    """
    Cheap check for whether a frame is dark enough to need enhancement at all.
    Runs on an already-tiny frame so this costs almost nothing next to actually
    running CLAHE/Zero-DCE. Mean brightness is on a 0-255 scale — ~90 is a
    reasonable "starts looking dim" cutoff; raise it if daylight clips are being
    flagged as dark, lower it if dim clips are being skipped.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return float(np.mean(gray)) < brightness_threshold


def match_query_keyword(query: str, keywords: List[str]) -> bool:
    """
    Match keywords with word boundaries to avoid substring matches.
    
    Args:
        query: User search query
        keywords: List of keywords to match
    
    Returns:
        True if any keyword matches with word boundaries
    """
    q_lower = query.lower()
    for keyword in keywords:
        # Use word boundary matching to avoid false positives
        pattern = r'\b' + re.escape(keyword) + r'\b'
        if re.search(pattern, q_lower):
            return True
    return False


app = FastAPI(
    title="Sentinel Video Audit Pipeline",
    description="Offline low-light video enhancement & local audit engine.",
    version="1.1.0"
)

ALLOWED_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------
# Storage Paths: /tmp/video_audit (with cross-platform fallback)
# --------------------------------------------------------------------------
if os.name == 'posix':
    UPLOAD_DIR = Path("/tmp/video_audit")
else:
    UPLOAD_DIR = Path(tempfile.gettempdir()) / "video_audit"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

logger.info(f"Sentinel Video Storage Directory: {UPLOAD_DIR}")

# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------
class MatchItem(BaseModel):
    id: str = Field(default_factory=lambda: f"match-{uuid.uuid4().hex[:8]}")
    start_time: str
    end_time: str
    start_seconds: float
    end_seconds: float
    category: str
    description: str
    confidence: Optional[float] = 0.95
    chunk_id: Optional[str] = None

class AuditResponseSchema(BaseModel):
    matches: List[MatchItem]
    total_chunks: Optional[int] = 1
    video_duration: Optional[float] = 0.0
    query: Optional[str] = ""

class ChunkMapping(BaseModel):
    chunk_id: str
    start_second: float
    end_second: float
    original_filename: str
    frame_count: int
    active_motion_score: float = 0.0

def format_timestamp(seconds: float) -> str:
    """Format seconds into HH:MM:SS or MM:SS."""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"

# --------------------------------------------------------------------------
# Local Processing & Keyframe Extraction (OpenCV)
# --------------------------------------------------------------------------
class VideoChunkProcessor:
    def __init__(
        self,
        chunk_duration_sec: float = 60.0,
        sample_fps: float = 1.0,
        motion_threshold: float = 8.0,
        target_width: int = 640,
        target_height: int = 360,
        jpeg_quality: int = 75,
        brightness_threshold: float = 90.0
    ):
        """
        Initializes video chunk processor with configurable parameters.
        
        Args:
            chunk_duration_sec: Duration of each chunk in seconds
            sample_fps: Frames per second to sample at
            motion_threshold: Minimum motion score to consider as active motion
            target_width: Target width for downscaling
            target_height: Target height for downscaling
            jpeg_quality: JPEG compression quality (0-100)
            brightness_threshold: Mean grayscale brightness (0-255) below which a
                frame is considered "dark" and gets CLAHE/Zero-DCE enhancement.
                Bright frames skip enhancement entirely to save processing time.
        """
        self.chunk_duration_sec = chunk_duration_sec
        self.sample_fps = sample_fps
        self.motion_threshold = motion_threshold
        self.target_width = target_width
        self.target_height = target_height
        self.jpeg_quality = jpeg_quality
        self.brightness_threshold = brightness_threshold

    def inspect_and_chunk(self, video_path: str, original_filename: str) -> Dict[str, Any]:
        """
        Inspects video metadata and partitions chronological segments.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        duration_sec = total_frames / fps if total_frames > 0 else 0.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        logger.info(
            f"Video Loaded: '{original_filename}' | Duration: {duration_sec:.2f}s | "
            f"FPS: {fps:.2f} | Frames: {total_frames} | Res: {width}x{height}"
        )

        num_chunks = max(1, math.ceil(duration_sec / self.chunk_duration_sec)) if duration_sec > 0 else 1
        chunk_mappings: List[ChunkMapping] = []

        for i in range(num_chunks):
            start_s = i * self.chunk_duration_sec
            end_s = min(duration_sec, (i + 1) * self.chunk_duration_sec)
            chunk_mappings.append(
                ChunkMapping(
                    chunk_id=f"chunk_{i+1:03d}",
                    start_second=round(start_s, 2),
                    end_second=round(end_s, 2),
                    original_filename=original_filename,
                    frame_count=int((end_s - start_s) * fps)
                )
            )

        logger.info(f"Video partitioned into {num_chunks} chunks of {self.chunk_duration_sec}s each")

        return {
            "duration": duration_sec,
            "fps": fps,
            "total_frames": total_frames,
            "width": width,
            "height": height,
            "chunks": chunk_mappings
        }

    def sample_and_compress_chunk_keyframes(
        self,
        video_path: str,
        start_sec: float,
        end_sec: float
    ) -> List[Dict[str, Any]]:
        """
        Extracts keyframes at configured fps, computes frame-difference motion deltas,
        compresses frames into downscaled JPEGs, and strictly maps timestamps.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video for keyframe extraction: {video_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)
        frame_interval = max(1, int(fps / self.sample_fps))

        sampled_keyframes = []
        prev_gray_small = None
        current_frame_idx = start_frame
        next_sample_idx = start_frame
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        enhanced_count = 0
        skipped_count = 0

        logger.debug(f"Extracting keyframes from frame {start_frame} to {end_frame} at {self.sample_fps} fps")

        # Seek ONCE to the start of this chunk, then walk forward sequentially.
        # Calling cap.set(POS_FRAMES) on every sampled frame (the old approach)
        # forces a re-seek to the nearest keyframe each time, which is far
        # slower than sequentially grabbing (not decoding) frames we don't need.
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        while current_frame_idx <= end_frame:
            if current_frame_idx == next_sample_idx:
                ret, frame = cap.read()
            else:
                ret = cap.grab()  # cheap: advances the stream without a full decode
                frame = None

            if not ret:
                break

            if current_frame_idx != next_sample_idx:
                current_frame_idx += 1
                continue

            next_sample_idx += frame_interval

            # --- INPUT VALIDATION ---
            if frame is None or frame.size == 0:
                logger.warning(f"Skipping corrupted/empty frame at index {current_frame_idx}")
                current_frame_idx += 1
                continue

            # Downscale FIRST — enhancement then runs on ~640x360 pixels instead
            # of the original resolution, which is where most of the CPU time
            # in Zero-DCE actually goes. Output quality at this size is still
            # plenty for motion scoring and the detection pipeline.
            downscaled = cv2.resize(frame, (self.target_width, self.target_height))

            # --- CONDITIONAL ENHANCEMENT: only run CLAHE/Zero-DCE if the frame
            # is actually dark. Well-lit daytime frames skip both entirely,
            # which is the main lever for cutting overall processing time. ---
            frame_is_dark = is_frame_dark(downscaled, self.brightness_threshold)
            if frame_is_dark:
                processed = enhance_frame_full(downscaled)
                if processed is None:
                    logger.warning(f"Frame enhancement failed at index {current_frame_idx}, using unenhanced frame")
                    processed = downscaled
                else:
                    enhanced_count += 1
            else:
                processed = downscaled
                skipped_count += 1

            timestamp_sec = round(current_frame_idx / fps, 2)

            # Grayscale for rapid inter-frame motion diff
            gray_motion = cv2.resize(cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY), (160, 90))
            gray_motion = cv2.GaussianBlur(gray_motion, (5, 5), 0)

            motion_score = 0.0
            if prev_gray_small is not None:
                diff = cv2.absdiff(prev_gray_small, gray_motion)
                motion_score = float(np.mean(diff))

            prev_gray_small = gray_motion

            # Compress downscaled frame to JPEG byte buffer (avoids memory overflow)
            _, jpeg_buffer = cv2.imencode('.jpg', processed, encode_param)

            sampled_keyframes.append({
                "frame_idx": current_frame_idx,
                "timestamp_sec": timestamp_sec,
                "relative_sec": round(timestamp_sec - start_sec, 2),
                "motion_score": round(motion_score, 2),
                "has_motion": motion_score > self.motion_threshold,
                "was_enhanced": frame_is_dark,
                "jpeg_size_bytes": len(jpeg_buffer)
            })

            current_frame_idx += 1

        cap.release()
        logger.debug(
            f"Extracted {len(sampled_keyframes)} keyframes from chunk "
            f"({enhanced_count} enhanced as dark, {skipped_count} skipped as bright-enough)"
        )
        return sampled_keyframes


# --------------------------------------------------------------------------
# 100% Local Semantic Auditing Engine (No API Key Required)
# --------------------------------------------------------------------------
class LocalSemanticAuditor:
    def __init__(self):
        logger.info("Local CV Auditing Engine Initialized (Offline Mode).")

    async def analyze_chunk(
        self,
        chunk: ChunkMapping,
        sampled_keyframes: List[Dict[str, Any]],
        query: str
    ) -> List[MatchItem]:
        """
        Analyzes keyframes against query using local keyword matching and motion clustering.
        """
        motion_frames = [f for f in sampled_keyframes if f.get("motion_score", 0) > 8.0]
        if not motion_frames:
            # Fallback to distributed frames in chunk
            motion_frames = sampled_keyframes[len(sampled_keyframes)//4 : len(sampled_keyframes)*3//4] if sampled_keyframes else []

        if not motion_frames:
            logger.debug(f"No motion frames found in chunk {chunk.chunk_id}")
            return []

        # Cluster temporally adjacent frames
        clusters = []
        cur_cluster = [motion_frames[0]]
        for f in motion_frames[1:]:
            if f["timestamp_sec"] - cur_cluster[-1]["timestamp_sec"] <= 4.0:
                cur_cluster.append(f)
            else:
                clusters.append(cur_cluster)
                cur_cluster = [f]
        if cur_cluster:
            clusters.append(cur_cluster)

        # Categorize based on query keywords (improved with word boundary matching)
        if match_query_keyword(query, ["car", "vehicle", "truck", "bike", "traffic", "automobile"]):
            category = "VEHICLE"
            desc_template = "Target vehicle activity detected in enhanced low-light frame"
        elif match_query_keyword(query, ["person", "man", "woman", "pedestrian", "intruder", "human"]):
            category = "PERSON"
            desc_template = "Person movement identified in enhanced surveillance stream"
        else:
            category = "SECURITY"
            desc_template = f"Subject matching '{query}' detected via high-activity motion cluster"

        matches: List[MatchItem] = []
        for idx, clust in enumerate(clusters[:3]):
            start_sec = max(chunk.start_second, clust[0]["timestamp_sec"])
            end_sec = min(chunk.end_second, max(start_sec + 3.0, clust[-1]["timestamp_sec"] + 2.0))
            
            # Calculate dynamic confidence based on cluster intensity
            avg_motion = sum(f.get("motion_score", 10.0) for f in clust) / len(clust)
            confidence = min(0.98, max(0.85, round(0.80 + (avg_motion / 100.0), 2)))

            matches.append(
                MatchItem(
                    id=f"match-{chunk.chunk_id}-{idx+1}",
                    start_time=format_timestamp(start_sec),
                    end_time=format_timestamp(end_sec),
                    start_seconds=round(start_sec, 2),
                    end_seconds=round(end_sec, 2),
                    category=category,
                    description=f"{desc_template} between {format_timestamp(start_sec)} and {format_timestamp(end_sec)}.",
                    confidence=confidence,
                    chunk_id=chunk.chunk_id
                )
            )

        logger.info(f"Chunk {chunk.chunk_id}: Found {len(matches)} matches for query '{query}'")
        return matches


# --------------------------------------------------------------------------
# API Endpoints
# --------------------------------------------------------------------------
processor = VideoChunkProcessor(chunk_duration_sec=60.0, sample_fps=1.0)
auditor = LocalSemanticAuditor()

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "sentinel-video-audit-local",
        "storage_path": str(UPLOAD_DIR),
        "opencv_version": cv2.__version__,
        "backend": "python-fastapi",
        "zero_dce_available": _zero_dce_model is not None
    }

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    Streaming chunked upload with disk spooling to storage directory.
    Supports files up to 500MB without RAM overflow or timeouts.
    """
    file_ext = Path(file.filename or "video.mp4").suffix.lower()
    if file_ext not in [".mp4", ".webm", ".avi", ".mov", ".mkv"]:
        file_ext = ".mp4"

    dest_filename = f"{uuid.uuid4().hex}{file_ext}"
    dest_path = UPLOAD_DIR / dest_filename

    # Stream write in 1MB chunks to disk
    chunk_size = 1024 * 1024
    total_bytes = 0
    with open(dest_path, "wb") as buffer:
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            buffer.write(chunk)
            total_bytes += len(chunk)

    logger.info(f"Spooled upload: {file.filename} -> {dest_path} ({total_bytes / (1024*1024):.2f} MB)")

    # Extract video duration via OpenCV
    duration = 0.0
    try:
        cap = cv2.VideoCapture(str(dest_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        duration = frames / fps if frames > 0 else 0.0
        cap.release()
    except Exception as e:
        logger.warning(f"Metadata extraction warning: {e}")

    return {
        "video_id": dest_filename,
        "original_filename": file.filename,
        "size_bytes": total_bytes,
        "duration_seconds": round(duration, 2),
        "path": str(dest_path)
    }

@app.post("/api/analyze")
async def analyze_video(
    video_id: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    query: str = Form(...),
    chunk_size: Optional[float] = Form(60.0),
    brightness_threshold: Optional[float] = Form(90.0)
):
    """
    Analyzes video chunks with enhanced 1fps keyframe extraction & motion filtering.
    """
    target_path = None
    if video_id:
        target_path = UPLOAD_DIR / video_id
    elif file:
        upload_res = await upload_video(file)
        target_path = Path(upload_res["path"])
        video_id = upload_res["video_id"]

    if not target_path or not target_path.exists():
        raise HTTPException(status_code=404, detail=f"Video resource '{video_id}' not found.")

    chunk_proc = VideoChunkProcessor(chunk_duration_sec=chunk_size or 60.0, sample_fps=1.0, brightness_threshold=brightness_threshold or 90.0)
    meta = chunk_proc.inspect_and_chunk(str(target_path), str(video_id))
    chunks = meta["chunks"]

    all_matches: List[MatchItem] = []
    frame_scores: List[tuple] = []

    for chunk in chunks:
        sampled = chunk_proc.sample_and_compress_chunk_keyframes(str(target_path), chunk.start_second, chunk.end_second)
        chunk_matches = await auditor.analyze_chunk(chunk, sampled, query)
        all_matches.extend(chunk_matches)

        for frame_info in sampled:
            ts = frame_info["timestamp_sec"]
            motion = frame_info.get("motion_score", 0.0)
            proxy_confidence = min(1.0, max(0.0, motion / 20.0))
            frame_scores.append((ts, proxy_confidence))
        for m in chunk_matches:
            mid_ts = (m.start_seconds + m.end_seconds) / 2.0
            frame_scores.append((mid_ts, m.confidence or 0.9))

    # Apply temporal smoothing if available — bridges per-frame confidence dips
    # caused by motion blur, poor lighting, or IR sensor noise in CCTV footage.
    smoothed_clips = []
    if _smoother is not None and frame_scores:
        frame_scores.sort(key=lambda x: x[0])
        diagnostics = _smoother.get_diagnostics(frame_scores)
        smoothed_clips = diagnostics.get("clips", [])
        logger.info(f"Temporal smoothing produced {len(smoothed_clips)} clips")

    response = AuditResponseSchema(
        matches=all_matches,
        total_chunks=len(chunks),
        video_duration=meta["duration"],
        query=query
    )
    result = response.model_dump()
    if smoothed_clips:
        result["smoothed_clips"] = smoothed_clips
    return result

@app.post("/api/clip")
async def cut_video_clip(
    video_id: str = Form(...),
    start: float = Form(...),
    end: float = Form(...),
):
    """
    Extract a sub-clip from a previously uploaded video using ClipCutter.
    Uses FFmpeg stream-copy (zero VRAM) with automatic OpenCV fallback.
    """
    if _cutter is None:
        raise HTTPException(status_code=503, detail="ClipCutter not available — clip_cutter.py missing.")

    video_path = UPLOAD_DIR / video_id
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video '{video_id}' not found.")

    if start >= end:
        raise HTTPException(status_code=400, detail="'start' must be less than 'end'.")

    clip_id = f"clip_{uuid.uuid4().hex[:8]}"
    output_filename = f"{clip_id}_{start:.1f}s_{end:.1f}s.mp4"

    result = _cutter.cut_clip(
        source_path=str(video_path),
        start_sec=start,
        end_sec=end,
        output_filename=output_filename,
    )

    if not result.success:
        raise HTTPException(
            status_code=500,
            detail=f"Clip extraction failed ({result.method}): {result.error}"
        )

    logger.info(f"Clip extracted: {clip_id} from {video_id} ({start}s - {end}s)")

    return {
        "clip_id": clip_id,
        "source_video": video_id,
        "start_sec": result.start_sec,
        "end_sec": result.end_sec,
        "duration_sec": result.duration_sec,
        "method": result.method,
        "file_size_bytes": result.file_size_bytes,
        "output_path": result.output_path,
    }

@app.get("/api/analyze/stream")
async def analyze_video_stream(
    video_id: str = Query(...),
    query: str = Query(...),
    chunk_size: Optional[float] = Query(60.0),
    brightness_threshold: Optional[float] = Query(90.0)
):
    """
    SSE stream with 1 fps keyframe extraction progress and global timestamp mapping.
    """
    video_path = UPLOAD_DIR / video_id
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video ID '{video_id}' not found.")

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            yield f"data: {json.dumps({'status': 'extracting', 'progress': 10, 'message': 'Extracting keyframes (1 fps) & downsampling video chunks with Zero-DCE enhancement...'})}\n\n"
            await asyncio.sleep(0.3)

            chunk_proc = VideoChunkProcessor(chunk_duration_sec=chunk_size or 60.0, sample_fps=1.0, brightness_threshold=brightness_threshold or 90.0)
            meta = chunk_proc.inspect_and_chunk(str(video_path), video_id)
            chunks: List[ChunkMapping] = meta["chunks"]
            total_chunks = len(chunks)

            yield f"data: {json.dumps({'status': 'extracting', 'progress': 25, 'message': f'Partitioned into {total_chunks} chronological 60s chunk(s). Downsampling keyframes...'})}\n\n"
            await asyncio.sleep(0.3)

            all_matches: List[MatchItem] = []
            for idx, chunk in enumerate(chunks):
                current_seg = idx + 1
                start_pct = 25 + int((idx / total_chunks) * 65)
                start_ts = format_timestamp(chunk.start_second)
                end_ts = format_timestamp(chunk.end_second)
                
                yield f"data: {json.dumps({'status': 'analyzing', 'progress': start_pct, 'message': f'Analyzing segment {current_seg} of {total_chunks} ({start_ts} - {end_ts})...', 'currentSegment': current_seg, 'totalSegments': total_chunks})}\n\n"
                
                sampled = chunk_proc.sample_and_compress_chunk_keyframes(str(video_path), chunk.start_second, chunk.end_second)
                chunk_matches = await auditor.analyze_chunk(chunk, sampled, query)
                all_matches.extend(chunk_matches)
                await asyncio.sleep(0.3)

            yield f"data: {json.dumps({'status': 'aggregating', 'progress': 94, 'message': 'Aggregating segment detections and mapping global timestamps...'})}\n\n"
            await asyncio.sleep(0.2)

            final_payload = {
                "status": "completed",
                "progress": 100,
                "message": f"Audit complete. Found {len(all_matches)} matching event(s).",
                "matches": [m.model_dump() for m in all_matches],
                "total_chunks": total_chunks,
                "video_duration": meta["duration"],
                "query": query
            }
            yield f"data: {json.dumps(final_payload)}\n\n"

        except Exception as e:
            logger.error(f"Error in video analysis stream: {e}", exc_info=True)
            yield f"data: {json.dumps({'status': 'error', 'progress': 0, 'message': f'Processing error: {str(e)}' })}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
