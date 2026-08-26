"""
Zero-DCE Low-Light Image Enhancement
=====================================
Provides per-frame enhancement using a deep curve estimation network.

Model path resolution order:
  1. ZERO_DCE_WEIGHTS env var
  2. Epoch99.pth in this file's directory
  3. Epoch99.pth in current working directory (legacy fallback)

Device resolution order:
  1. ZERO_DCE_DEVICE env var (cpu / cuda / cuda:0 / mps)
  2. Auto-detect: CUDA → MPS → CPU
"""

import os
import logging
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import torch
from model import EnhanceNetNoPool

logger = logging.getLogger("zero-dce")

# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def _resolve_device() -> torch.device:
    """Pick the best available compute device."""
    env_device = os.environ.get("ZERO_DCE_DEVICE", "").strip().lower()
    if env_device:
        return torch.device(env_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_model_path() -> Optional[Path]:
    """Resolve the model weights file path."""
    # 1. Explicit env var
    env_path = os.environ.get("ZERO_DCE_WEIGHTS", "").strip()
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p
        logger.warning("ZERO_DCE_WEIGHTS set to %s but file not found", env_path)

    # 2. Same directory as this file
    local = Path(__file__).resolve().parent / "Epoch99.pth"
    if local.exists():
        return local

    # 3. Legacy: current working directory
    cwd = Path.cwd() / "Epoch99.pth"
    if cwd.exists():
        logger.info("Using legacy model path (cwd): %s", cwd)
        return cwd

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_zero_dce() -> Optional[torch.nn.Module]:
    """
    Loads and initializes the Zero-DCE model with pre-trained weights.

    Returns:
        Loaded model on the resolved device, or None if loading fails.
    """
    device = _resolve_device()
    model_path = _resolve_model_path()

    if model_path is None:
        logger.error(
            "Zero-DCE weights not found. Set ZERO_DCE_WEIGHTS env var or "
            "place Epoch99.pth next to backend/enhance.py"
        )
        return None

    try:
        model = EnhanceNetNoPool()
        state_dict = torch.load(str(model_path), map_location="cpu")
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        logger.info("Zero-DCE model loaded: path=%s device=%s", model_path, device)
        return model
    except Exception as e:
        logger.error("Failed to load Zero-DCE model from %s: %s", model_path, e)
        return None


def apply_zero_dce(
    image_input: Union[str, np.ndarray],
    model: Optional[torch.nn.Module] = None,
) -> Optional[np.ndarray]:
    """
    Enhances a single image or video frame using Zero-DCE.

    Args:
        image_input: Either a file path (str) or loaded OpenCV frame (numpy array).
        model: Pre-loaded model. If None, will attempt to load.

    Returns:
        Enhanced frame as numpy array (BGR, uint8) or None if enhancement fails.
    """
    if image_input is None:
        logger.warning("apply_zero_dce: received None input")
        return None

    try:
        if model is None:
            model = load_zero_dce()
            if model is None:
                return None

        # Load from path if needed
        if isinstance(image_input, str):
            frame = cv2.imread(image_input)
            if frame is None:
                logger.error("Failed to read image: %s", image_input)
                return None
        else:
            frame = image_input

        if frame is None or frame.size == 0:
            logger.warning("apply_zero_dce: empty or invalid frame")
            return None

        # Determine device from model parameters
        device = next(model.parameters()).device

        # Prepare tensor
        data = frame.astype(np.float32) / 255.0
        tensor = torch.from_numpy(data).float().permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            _, enhanced_tensor, _ = model(tensor)

        enhanced_np = enhanced_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        enhanced_np = np.clip(enhanced_np * 255.0, 0, 255).astype(np.uint8)
        return enhanced_np

    except Exception as e:
        logger.error("Error during Zero-DCE inference: %s", e)
        return None


def enhance_video(input_video_path: str, output_video_path: str) -> bool:
    """
    Processes a video file frame-by-frame through Zero-DCE and outputs
    an enhanced video. Intended for standalone testing only -- the server
    uses apply_zero_dce() per-frame instead.
    """
    try:
        model = load_zero_dce()
        if model is None:
            logger.error("Cannot enhance video: model not loaded")
            return False

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            logger.error("Failed to open input video: %s", input_video_path)
            return False

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info("Input video: %dx%d @ %dfps, %d frames", width, height, fps, total_frames)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        if not out.isOpened():
            logger.error("Failed to create output video writer: %s", output_video_path)
            cap.release()
            return False

        frame_count = 0
        failed_frames = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame is None or frame.size == 0:
                logger.warning("Skipping empty/corrupted frame at index %d", frame_count)
                failed_frames += 1
                frame_count += 1
                continue

            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                logger.info("Processing frame %d/%d (%.1f%%)", frame_count, total_frames, progress)

            try:
                enhanced = apply_zero_dce(frame, model)
                if enhanced is None:
                    logger.warning("Enhancement failed for frame %d, using original", frame_count)
                    enhanced = frame
            except Exception as e:
                logger.warning("Exception at frame %d: %s", frame_count, e)
                enhanced = frame

            out.write(enhanced)
            frame_count += 1

        cap.release()
        out.release()

        logger.info(
            "Enhanced video saved to %s (%d frames, %d failed)",
            output_video_path, frame_count, failed_frames,
        )
        return True

    except Exception as e:
        logger.error("Fatal error in enhance_video: %s", e)
        return False
