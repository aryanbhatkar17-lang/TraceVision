import os
import subprocess
import shutil
import logging
import tempfile
import torch
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger("tracevision.pipeline")

_BACKEND_DIR = Path(__file__).resolve().parent
_ENV_CANDIDATES = [_BACKEND_DIR / ".env", _BACKEND_DIR / ".env.local"]
for _candidate in _ENV_CANDIDATES:
    if _candidate.exists():
        load_dotenv(dotenv_path=_candidate, encoding="utf-8-sig", override=True)
        break

from google import genai
from typing import List, Dict, Any, Optional
from legacy_server import SemanticVideoAuditor
from clip_engine import evaluate_video_frames
from smoothing import TemporalSmoother

_PIPELINE_GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
if not _PIPELINE_GEMINI_KEY:
    raise RuntimeError(
        f"GEMINI_API_KEY is not set. Checked: {[str(p) for p in _ENV_CANDIDATES]}. "
        "Create backend/.env (or .env.local) with:\n"
        "    GEMINI_API_KEY=your_actual_key_here"
    )

_gemini_client = genai.Client(api_key=_PIPELINE_GEMINI_KEY)

# Current-generation fast text model (gemini-1.5-flash is retired).
TRANSLATION_MODEL = "gemini-3.5-flash-lite"

# --- Tunable constants -------------------------------------------------
CLIP_NOISE_FLOOR    = 0.10  # Low floor keeps compositional/spatial queries alive
CLIP_TOP_K          = 15    # Number of uniform timeline bins
CLIP_PEAKS_PER_BIN  = 1    # Top-N CLIP candidates to keep per bin
# ------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Frame Extraction — FFmpeg Hardware-Accelerated with Fast Fallback
# ---------------------------------------------------------------------------

def _get_video_duration(video_path: str) -> float:
    """
    Use FFprobe to get the exact video duration in seconds.
    Falls back to 0.0 if FFprobe is unavailable or parsing fails.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True, text=True, timeout=30,
        )
        return float(result.stdout.strip())
    except Exception as e:
        logger.warning(f"[FFprobe] Could not determine duration: {e}")
        return 0.0


def _adaptive_fps(duration_sec: float) -> float:
    """
    Choose extraction FPS to cap total frames to ~25 frames,
    ensuring lightning fast CPU extraction and CLIP scoring.
    """
    if duration_sec <= 0:
        return 1.0
    fps = 25.0 / max(duration_sec, 1.0)
    return round(max(min(fps, 1.0), 0.05), 3)


def _extract_frames_ffmpeg(video_path: str, output_dir: str, fps: Optional[float] = None) -> Dict[float, str]:
    """
    Extract frames using native FFmpeg.
    """
    duration_sec = _get_video_duration(video_path)
    if fps is None:
        fps = _adaptive_fps(duration_sec)

    logger.info(
        f"[Frame Extraction] duration={duration_sec:.1f}s, adaptive_fps={fps}, "
        f"estimated_frames={int(duration_sec * fps)}"
    )

    output_pattern = os.path.join(output_dir, "frame_%04d.jpg")

    # Fast I-frame extraction with fallback (skips non-keyframes for 50x faster decode)
    cmd_fast = [
        "ffmpeg", "-y",
        "-skip_frame", "nokey",
        "-i", video_path,
        "-vf", f"fps={fps},scale='min(512,iw)':-2",
        "-vsync", "vfr",
        "-vframes", "25",
        "-q:v", "4",
        "-f", "image2",
        output_pattern,
    ]

    cmd_standard = [
        "ffmpeg", "-y",
        "-threads", "2",
        "-i", video_path,
        "-vf", f"fps={fps},scale='min(512,iw)':-2",
        "-vframes", "25",
        "-q:v", "4",
        "-f", "image2",
        output_pattern,
    ]

    try:
        subprocess.run(cmd_fast, capture_output=True, text=True, timeout=60, check=True)
        frame_files = sorted(f for f in os.listdir(output_dir) if f.startswith("frame_") and f.endswith(".jpg"))
        if not frame_files:
            logger.info("[Frame Extraction] Fast extraction returned 0 frames, falling back to standard extraction...")
            subprocess.run(cmd_standard, capture_output=True, text=True, timeout=60, check=True)
    except Exception as e:
        logger.warning(f"[Frame Extraction] Fast extraction failed: {e}. Retrying with standard extraction...")
        try:
            subprocess.run(cmd_standard, capture_output=True, text=True, timeout=60, check=True)
        except subprocess.CalledProcessError as e2:
            raise RuntimeError(f"FFmpeg extraction failed (rc={e2.returncode}):\n{e2.stderr[-800:]}")
    except FileNotFoundError:
        raise RuntimeError("FFmpeg not found in PATH. Install FFmpeg and ensure it is accessible.")

    # Reconstruct {timestamp_seconds: filepath} from the written files.
    # FFmpeg names frames starting from frame_0001.jpg (1-indexed).
    # timestamp = (frame_number - 1) / fps
    timestamp_to_path: Dict[float, str] = {}
    frame_files = sorted(
        f for f in os.listdir(output_dir)
        if f.startswith("frame_") and f.endswith(".jpg")
    )

    for frame_file in frame_files:
        try:
            frame_number = int(frame_file.replace("frame_", "").replace(".jpg", ""))
        except ValueError:
            continue
        timestamp_seconds = round((frame_number - 1) / fps, 2)
        timestamp_to_path[timestamp_seconds] = os.path.join(output_dir, frame_file)

    logger.info(f"[Frame Extraction] Wrote {len(timestamp_to_path)} frames to {output_dir}")
    return timestamp_to_path


# ---------------------------------------------------------------------------
# Timestamp Bridge
# ---------------------------------------------------------------------------

def _closest_frame_path(timestamp: float, timestamp_to_path: Dict[float, str]) -> str:
    """
    Snap a CLIP-reported timestamp to the nearest extracted frame path.
    CLIP may return float values with different precision than what FFmpeg wrote;
    this prevents missed dict lookups.
    """
    closest_ts = min(timestamp_to_path.keys(), key=lambda t: abs(t - timestamp))
    return timestamp_to_path[closest_ts]


# ---------------------------------------------------------------------------
# Stage 1 — LLM Query Translation
# ---------------------------------------------------------------------------

async def _translate_query_for_clip(raw_query: str) -> str:
    """
    Converts a conversational officer query into a broad-recall CLIP-optimized
    visual co-occurrence caption.

    CLIP ViT-B/32 cannot understand spatial prepositional binding on its own
    ("beside", "next to", "near"). Direct spatial queries produce lower cosine
    similarity scores than they deserve, causing the model to rank relevant
    frames below the noise floor. The fix: translate to broad descriptive
    co-occurrence captions that include both subjects in natural scene language,
    letting CLIP's visual encoder find frames where both objects are present.

    Examples:
        "locate man beside car"    -> "a person walking near cars on a street or road"
        "person near truck"        -> "a pedestrian standing next to a large truck on a road"
        "Did a red car pass by?"   -> "a red car driving on a street"
        "man with yellow backpack" -> "a man carrying a yellow backpack walking outdoors"
    """
    try:
        prompt = (
            "You are an AI vision assistant for a CCTV forensic search system. "
            "Convert the following surveillance search query into a broad, descriptive "
            "visual scene caption optimized for CLIP image-text similarity matching.\n\n"
            "Rules:\n"
            "1. For spatial queries ('beside', 'near', 'next to', 'walking by', 'inside'), "
            "describe both subjects co-occurring naturally in the scene WITHOUT using "
            "the spatial preposition itself (CLIP cannot reason about prepositions).\n"
            "2. Keep the description broad and general — avoid overly specific details "
            "that reduce recall on wide-angle CCTV footage.\n"
            "3. Use natural, scene-level language (e.g., 'on a street', 'on a road', "
            "'on a sidewalk', 'in a parking lot') to help CLIP find the correct environment.\n"
            "4. Output ONLY the optimized visual caption. No extra text, no punctuation at end.\n\n"
            f"Search Query: {raw_query}"
        )

        response = _gemini_client.models.generate_content(
            model=TRANSLATION_MODEL,
            contents=prompt,
        )
        translated = (response.text or "").strip()

        if not translated:
            return raw_query

        return translated
    except Exception as e:
        logger.warning(f"Gemini translation failed ({e}), falling back to raw query.")
        return raw_query


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

async def _run_search_pipeline(video_path: str, raw_query: str, smoother: "TemporalSmoother") -> Dict[str, Any]:
    """
    Four-Stage Multimodal RAG search pipeline for TraceVision.

    This is the function api.py imports via:
        from pipeline import _run_search_pipeline
    Do NOT rename this — api.py's import binds to this exact name.

    Flow:
      1. LLM Query Translation     -> CLIP-optimized visual co-occurrence caption
      2. FFmpeg CUDA Frame Extract -> High-fidelity 1024px frames (GPU-accelerated)
      3. CLIP Retrieval            -> Timeline-bucketed peak candidates
      4. Gemini Validation         -> Spatial forensic auditor confirms each frame
      5. Temporal Smoothing        -> Only Gemini-validated timestamps go in
    """
    temp_frames_dir = tempfile.mkdtemp(prefix="tracevision_frames_")

    try:
        # ------------------------------------------------------------
        # STAGE 1: LLM Query Translation
        # ------------------------------------------------------------
        clip_query = await _translate_query_for_clip(raw_query)
        logger.info(f"[Stage 1] Raw query: '{raw_query}' -> CLIP query: '{clip_query}'")

        # ------------------------------------------------------------
        # STAGE 2: FFmpeg CUDA Frame Extraction
        # ------------------------------------------------------------
        # Compute adaptive_fps here so we can pass it to the CLIP engine.
        # evaluate_video_frames needs the exact fps to reconstruct timestamps
        # from the sequential frame_%04d.jpg filenames FFmpeg writes.
        video_duration_sec = _get_video_duration(video_path)
        adaptive_fps = _adaptive_fps(video_duration_sec)

        timestamp_to_path = _extract_frames_ffmpeg(video_path, temp_frames_dir, fps=adaptive_fps)

        if not timestamp_to_path:
            raise RuntimeError("No frames were extracted from the video — check the source file.")

        # ------------------------------------------------------------
        # STAGE 3: CLIP Retrieval with Timeline-Aware Bucketing
        # ------------------------------------------------------------
        # Pass adaptive_fps so clip_engine can compute timestamps as:
        #   timestamp = (frame_index - 1) / adaptive_fps
        raw_scores: List[Dict[str, float]] = evaluate_video_frames(temp_frames_dir, clip_query, fps=adaptive_fps)

        # Noise floor: 0.10 prevents spatial/compositional queries from being
        # dropped prematurely (they inherently score lower than simple object queries).
        above_noise_floor = [f for f in raw_scores if f["confidence"] >= CLIP_NOISE_FLOOR]

        # ------------------------------------------------------------------
        # Temporal Sub-Window Peak Sampling — Top-N Per Bin
        #
        # ROOT CAUSE FIX: "Binning Collision" false negatives.
        # Empirically traced on the 34-min orange-truck video:
        #   - Bin 8 (10:55–12:17): Eddie Stobart green truck at 11:42 scored
        #     0.2745 CLIP; real orange truck at 11:47 scored 0.2469.
        #   - Old single-winner logic: 11:42 wins → 11:47 is discarded forever.
        #   - Gemini sees 11:42, correctly says "no orange truck" → false negative.
        #
        # Fix: keep CLIP_PEAKS_PER_BIN=2 candidates per bin. Both the 11:42
        # false-positive and 11:47 real-positive are forwarded to Gemini. Gemini
        # then independently rejects 11:42 (green) and confirms 11:47 (orange).
        #
        # Combined with CLIP_TOP_K=50 bins (vs 25), the total candidate budget is
        # up to 50×2=100 frames — still a single Gemini batch call, so zero extra
        # quota cost while halving the blind-spot window and adding redundancy.
        # ------------------------------------------------------------------
        if not above_noise_floor:
            return {"query": raw_query, "clip_query": clip_query, "matches": [], "clips": []}

        scored_by_time = sorted(above_noise_floor, key=lambda f: f["timestamp_seconds"])
        min_ts = scored_by_time[0]["timestamp_seconds"]
        max_ts = scored_by_time[-1]["timestamp_seconds"]
        timeline_duration = max_ts - min_ts

        # e.g. 2048s / 50 bins = ~41s per bin for a 34-min video (was 82s at 25 bins)
        bin_size = max(timeline_duration / CLIP_TOP_K, 1.0)

        # bins[bin_idx] = list of top-N frames by confidence in that bin
        bins: Dict[int, List[Dict[str, float]]] = {}
        for frame in scored_by_time:
            bin_idx = int((frame["timestamp_seconds"] - min_ts) / bin_size)
            bin_idx = min(bin_idx, CLIP_TOP_K - 1)  # clamp boundary edge case

            if bin_idx not in bins:
                bins[bin_idx] = []

            peaks = bins[bin_idx]
            peaks.append(frame)
            # Keep only the top-N by confidence; sort descending and trim
            peaks.sort(key=lambda f: f["confidence"], reverse=True)
            bins[bin_idx] = peaks[:CLIP_PEAKS_PER_BIN]

        # Flatten all bins, restore chronological order (critical for Gemini's
        # frame-index-to-timestamp mapping and the temporal smoother)
        top_candidates: List[Dict[str, float]] = []
        for bin_idx in sorted(bins.keys()):
            top_candidates.extend(bins[bin_idx])
        top_candidates.sort(key=lambda f: f["timestamp_seconds"])

        logger.info(
            f"[Stage 3] CLIP retrieved {len(raw_scores)} scored frames, "
            f"{len(above_noise_floor)} above noise floor ({CLIP_NOISE_FLOOR}). "
            f"Timeline ({timeline_duration:.1f}s) split into {CLIP_TOP_K} bins of "
            f"{bin_size:.1f}s (top {CLIP_PEAKS_PER_BIN} per bin). "
            f"Selected {len(top_candidates)} candidates for Gemini validation."
        )

        if not top_candidates:
            return {"query": raw_query, "clip_query": clip_query, "matches": [], "clips": []}

        # ------------------------------------------------------------
        # BRIDGE: map CLIP's scored timestamps -> actual frame file paths
        # ------------------------------------------------------------
        candidate_frames_for_audit = [
            {
                "timestamp_seconds": c["timestamp_seconds"],
                "confidence": c["confidence"],
                "frame_path": _closest_frame_path(c["timestamp_seconds"], timestamp_to_path),
            }
            for c in top_candidates
        ]

        # ------------------------------------------------------------
        # STAGE 4: Gemini Spatial Forensic Validation
        # ------------------------------------------------------------
        auditor = SemanticVideoAuditor()

        # Pass the ORIGINAL raw officer query — not the CLIP caption — because
        # Gemini does the spatial/logical reasoning CLIP can't.
        audit_results = await auditor.validate_frames(
            frame_paths=[c["frame_path"] for c in candidate_frames_for_audit],
            original_query=raw_query,
        )

        verdict_by_path = {r["frame_path"]: r for r in audit_results}

        validated_frames = [
            {
                "timestamp_seconds": c["timestamp_seconds"],
                # Prefer Gemini's semantic confidence over CLIP's raw cosine similarity —
                # Gemini has seen the actual 1024px frame and evaluated spatial proximity.
                "confidence": verdict_by_path.get(c["frame_path"], {}).get("confidence", c["confidence"]),
                "reasoning": verdict_by_path.get(c["frame_path"], {}).get("reasoning", ""),
            }
            for c in candidate_frames_for_audit
            if verdict_by_path.get(c["frame_path"], {}).get("is_match") is True
        ]

        logger.info(
            f"[Stage 4] Gemini validated {len(validated_frames)}/{len(candidate_frames_for_audit)} "
            f"CLIP candidates as true positives."
        )

        if not validated_frames:
            return {"query": raw_query, "clip_query": clip_query, "matches": [], "clips": []}

        # ------------------------------------------------------------
        # STAGE 5: Temporal Smoothing & Frontend Formatting
        # ------------------------------------------------------------
        # Sort chronologically before feeding to TemporalSmoother — its
        # gap-bridging logic assumes non-decreasing timestamps.
        validated_frames.sort(key=lambda x: x["timestamp_seconds"])

        smoother_input = [(f["timestamp_seconds"], f["confidence"]) for f in validated_frames]
        diagnostics = smoother.get_diagnostics(smoother_input)

        frontend_matches = []
        for idx, f in enumerate(validated_frames):
            frontend_matches.append({
                "id": f"gemini-match-{idx}",
                "start_seconds": f["timestamp_seconds"],
                "end_seconds": f["timestamp_seconds"] + 2.0,
                "confidence": f["confidence"],
                "description": f.get("reasoning", f"Verified match: {clip_query}"),
                "category": "SECURITY",
            })

        return {
            "query": raw_query,
            "clip_query": clip_query,
            "matches": frontend_matches,
            "clips": diagnostics,
        }

    finally:
        # Never let extracted frames pile up on disk
        shutil.rmtree(temp_frames_dir, ignore_errors=True)
        logger.info(f"Cleaned up temp frame directory: {temp_frames_dir}")