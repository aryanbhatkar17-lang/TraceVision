import os
import cv2
import shutil
import logging
import tempfile
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
from typing import List, Dict, Any
from server import SemanticVideoAuditor
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
FRAME_SAMPLE_RATE_FPS = 1
CLIP_NOISE_FLOOR = 0.10          # Lowered to 0.10 to allow night/low-contrast footage through
CLIP_TOP_K = 25                  # matches gemini-3.5-flash-lite's generous quota
# ------------------------------------------------------------------------


def _extract_frames_opencv(video_path: str, output_dir: str, fps: int = None) -> Dict[float, str]:
    """
    Extracts frames using OpenCV and saves them as JPEGs.
    Returns a mapping of {timestamp_seconds: frame_file_path} so we can
    bridge CLIP's (timestamp, confidence) output back to actual image files
    for Gemini later.

    If fps is not given, it's chosen adaptively from video duration: short
    clips (<=30s) sample at 2fps so quick, brief actions (a hand reaching
    into a bag, a fast pass-by) aren't missed entirely between 1fps gaps;
    longer footage falls back to 1fps to keep frame/API counts reasonable.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    if fps is None:
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        duration_sec = (total_frames / native_fps) if native_fps > 0 else 0
        fps = 2 if duration_sec <= 30 else FRAME_SAMPLE_RATE_FPS

    frame_interval = max(int(round(native_fps / fps)), 1)

    timestamp_to_path: Dict[float, str] = {}
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            timestamp_seconds = round(frame_idx / native_fps, 2)
            filename = f"frame_{timestamp_seconds:.2f}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            timestamp_to_path[timestamp_seconds] = filepath

        frame_idx += 1

    cap.release()
    return timestamp_to_path


def _closest_frame_path(timestamp: float, timestamp_to_path: Dict[float, str]) -> str:
    """
    evaluate_video_frames may not return timestamps with identical float
    precision to what OpenCV wrote to disk, so we snap to the nearest
    known extracted frame rather than assuming an exact dict hit.
    """
    closest_ts = min(timestamp_to_path.keys(), key=lambda t: abs(t - timestamp))
    return timestamp_to_path[closest_ts]


async def _translate_query_for_clip(raw_query: str) -> str:
    """
    Stage 1: LLM Query Translation using Gemini.
    Converts a conversational officer query into a CLIP-optimized visual caption.

    Example:
        "locate a person with a stroller" -> "a baby stroller on a sidewalk"
        "Did a red car pass by?"          -> "a red car driving on a street"
    """
    try:
        prompt = (
            "You are an AI assistant for a CCTV surveillance search system. "
            "Convert the following natural language search query into a concise, "
            "visual scene description optimized for CLIP image-text matching. "
            "Output ONLY the optimized visual description/caption, with no extra text or punctuation.\n\n"
            f"Query: {raw_query}"
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


async def _run_search_pipeline(video_path: str, raw_query: str, smoother: "TemporalSmoother") -> Dict[str, Any]:
    """
    Two-Stage Multimodal RAG search pipeline for TraceVision.

    This is the function api.py imports via:
        from pipeline import _run_search_pipeline
    Do NOT rename this — api.py's import binds to this exact name.

    Flow:
      1. LLM Query Translation   -> CLIP-optimized visual caption
      2. CLIP Retrieval          -> Top-K raw candidate frames (fast, noisy)
      3. LLM Validation          -> Gemini (SemanticVideoAuditor) confirms
                                     which candidates are real matches
      4. Temporal Smoothing      -> Only Gemini-validated timestamps go in
    """
    temp_frames_dir = tempfile.mkdtemp(prefix="tracevision_frames_")

    try:
        # ------------------------------------------------------------
        # STAGE 1: LLM Query Translation
        # ------------------------------------------------------------
        clip_query = await _translate_query_for_clip(raw_query)
        logger.info(f"[Stage 1] Raw query: '{raw_query}' -> CLIP query: '{clip_query}'")

        # ------------------------------------------------------------
        # STAGE 2: CLIP Retrieval (Top-K, not a hard threshold)
        # ------------------------------------------------------------
        timestamp_to_path = _extract_frames_opencv(video_path, temp_frames_dir)

        if not timestamp_to_path:
            raise RuntimeError("No frames were extracted from the video — check the source file.")

        # evaluate_video_frames returns: [{"timestamp_seconds": float, "confidence": float}]
        raw_scores: List[Dict[str, float]] = evaluate_video_frames(temp_frames_dir, clip_query)

        # Apply only the noise floor, then rank and slice to Top-K.
        # This is the key fix: we stop treating a single scalar threshold
        # as a pass/fail gate and instead treat CLIP purely as a retriever.
        above_noise_floor = [f for f in raw_scores if f["confidence"] >= CLIP_NOISE_FLOOR]
        top_candidates = sorted(above_noise_floor, key=lambda f: f["confidence"], reverse=True)[:CLIP_TOP_K]

        logger.info(
            f"[Stage 2] CLIP retrieved {len(raw_scores)} scored frames, "
            f"{len(above_noise_floor)} above noise floor ({CLIP_NOISE_FLOOR}), "
            f"top {len(top_candidates)} passed to Gemini for validation."
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
        # STAGE 3: LLM Validation via SemanticVideoAuditor (Gemini)
        # ------------------------------------------------------------
        auditor = SemanticVideoAuditor()

        # We pass the ORIGINAL raw officer query here — not the CLIP
        # caption — because Gemini does the spatial/logical reasoning
        # CLIP can't ("with a stroller" implies a person AND a stroller
        # together, not just either object present in frame).
        audit_results = await auditor.validate_frames(
            frame_paths=[c["frame_path"] for c in candidate_frames_for_audit],
            original_query=raw_query,
        )
        # Expected shape from SemanticVideoAuditor.validate_frames:
        # [{"frame_path": str, "is_match": bool, "reasoning": str}, ...]

        verdict_by_path = {r["frame_path"]: r for r in audit_results}

        validated_frames = [
            {
                "timestamp_seconds": c["timestamp_seconds"],
                "confidence": c["confidence"],
                "reasoning": verdict_by_path.get(c["frame_path"], {}).get("reasoning", ""),
            }
            for c in candidate_frames_for_audit
            if verdict_by_path.get(c["frame_path"], {}).get("is_match") is True
        ]

        logger.info(
            f"[Stage 3] Gemini validated {len(validated_frames)}/{len(candidate_frames_for_audit)} "
            f"CLIP candidates as true positives."
        )

        if not validated_frames:
            return {"query": raw_query, "clip_query": clip_query, "matches": [], "clips": []}

        # ------------------------------------------------------------
        # STAGE 4: Temporal Smoothing & Frontend Formatting
        # ------------------------------------------------------------
        # CRITICAL FIX: sort chronologically. validated_frames comes out
        # of the is_match filter in confidence-rank order (whatever order
        # asyncio.gather resolved them), NOT time order. TemporalSmoother's
        # gap-bridging logic assumes non-decreasing timestamps — feeding it
        # an out-of-order sequence corrupts clustering (e.g. splits a real
        # continuous 10s-16s event into garbage instead of one clip).
        validated_frames.sort(key=lambda x: x["timestamp_seconds"])

        # The smoother strictly expects a list of tuples: [(timestamp, confidence)]
        smoother_input = [(f["timestamp_seconds"], f["confidence"]) for f in validated_frames]
        diagnostics = smoother.get_diagnostics(smoother_input)

        # Format the output strictly to match the AuditMatch interface in dashboard.tsx
        frontend_matches = []
        for idx, f in enumerate(validated_frames):
            frontend_matches.append({
                "id": f"gemini-match-{idx}",
                "start_seconds": f["timestamp_seconds"],
                "end_seconds": f["timestamp_seconds"] + 2.0,  # Assuming a standard 2s snippet length
                "confidence": f["confidence"],
                "description": f.get("reasoning", f"Verified match: {clip_query}"),
                "category": "SECURITY"
            })

        return {
            "query": raw_query,
            "clip_query": clip_query,
            "matches": frontend_matches,
            "clips": diagnostics,
        }

    finally:
        # ------------------------------------------------------------
        # Cleanup: never let extracted frames pile up on disk
        # ------------------------------------------------------------
        shutil.rmtree(temp_frames_dir, ignore_errors=True)
        logger.info(f"Cleaned up temp frame directory: {temp_frames_dir}")