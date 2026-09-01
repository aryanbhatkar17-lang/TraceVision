"""
clip_engine.py
================
Loads CLIP ViT-B/32 once as a singleton and exposes evaluate_video_frames(),
which scores every extracted JPEG in a directory against a text query using
cosine similarity between CLIP's image and text embeddings.

Frame filenames are produced by FFmpeg's image2 muxer in the form:
  "frame_%04d.jpg"  (1-indexed, e.g. frame_0001.jpg, frame_0024.jpg)

Timestamp is reconstructed as:
  timestamp_seconds = (frame_index - 1) / fps

where `fps` is the adaptive extraction rate passed in from pipeline.py.
"""

import os
import re
import gc
import glob
import logging
from typing import List, Dict

import torch
from PIL import Image

# Force PyTorch to use 1 thread to minimize RAM arena fragmentation on 512MB instances
torch.set_num_threads(1)

logger = logging.getLogger("tracevision.clip")

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_BACKEND = None  # "openai" or "transformers"
_MODEL = None
_PREPROCESS = None

# Matches frame_%04d.jpg filenames produced by FFmpeg's image2 muxer.
# Captures the 1-based integer frame index (e.g. "0001" from "frame_0001.jpg").
_FRAME_FILENAME_RE = re.compile(r"frame_(\d+)\.jpg$")


def _load_model():
    """Loads ViT-B/32 exactly once per process. Supports both openai-clip
    and huggingface transformers."""
    global _MODEL, _PREPROCESS, _BACKEND
    if _MODEL is not None:
        return _BACKEND, _MODEL, _PREPROCESS

    # Try OpenAI's clip package first
    try:
        import clip
        logger.info(f"Loading OpenAI CLIP ViT-B/32 on device={_DEVICE} ...")
        _MODEL, _PREPROCESS = clip.load("ViT-B/32", device=_DEVICE)
        _MODEL.eval()
        _BACKEND = "openai"
        logger.info("OpenAI CLIP ViT-B/32 loaded.")
        return _BACKEND, _MODEL, _PREPROCESS
    except ImportError:
        pass

    # Fallback to HuggingFace transformers CLIP (already in requirements.txt)
    try:
        from transformers import CLIPProcessor, CLIPModel
        logger.info(f"Loading HuggingFace CLIP (openai/clip-vit-base-patch32) on device={_DEVICE} ...")
        _MODEL = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(_DEVICE)
        _PREPROCESS = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _MODEL.eval()
        _BACKEND = "transformers"
        logger.info("HuggingFace CLIP ViT-B/32 loaded.")
        return _BACKEND, _MODEL, _PREPROCESS
    except Exception as e:
        logger.error(f"Failed to load CLIP via both openai-clip and transformers: {e}")
        raise RuntimeError(
            "Neither 'clip' nor 'transformers' CLIP is available. "
            "Please install transformers (pip install transformers) or clip."
        )


def _timestamp_from_filename(path: str, fps: float) -> float:
    """
    Converts an FFmpeg frame_%04d.jpg filename back to a video timestamp.

    FFmpeg's image2 muxer produces 1-indexed filenames:
      frame_0001.jpg -> frame_index=1 -> timestamp = (1-1)/fps = 0.0s
      frame_0002.jpg -> frame_index=2 -> timestamp = (2-1)/fps = 1.0s  (at 1fps)
      frame_0025.jpg -> frame_index=25 -> timestamp = (25-1)/fps = 24.0s (at 1fps)

    Args:
        path: Absolute path to the frame file.
        fps:  The extraction FPS passed in from pipeline.py's _adaptive_fps().

    Raises:
        ValueError: If the filename doesn't match the expected pattern.
    """
    match = _FRAME_FILENAME_RE.search(os.path.basename(path))
    if not match:
        raise ValueError(
            f"Frame filename doesn't match FFmpeg frame_%04d.jpg pattern: {path}"
        )
    frame_index = int(match.group(1))
    return round((frame_index - 1) / fps, 2)


def evaluate_video_frames(frames_dir: str, query: str, fps: float = 1.0, batch_size: int = 8) -> List[Dict[str, float]]:
    """
    Scores every frame in frames_dir against `query` using CLIP cosine similarity.
    Uses small batches and explicit garbage collection to stay well below 512MB RAM.
    """
    backend, model, preprocess = _load_model()

    frame_paths = sorted(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))
    if not frame_paths:
        logger.warning(f"No frames found in {frames_dir}")
        return []

    results: List[Dict[str, float]] = []

    if backend == "openai":
        import clip
        text_tokens = clip.tokenize([query]).to(_DEVICE)
        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        for i in range(0, len(frame_paths), batch_size):
            batch_paths = frame_paths[i:i + batch_size]
            images, valid_paths = [], []

            for p in batch_paths:
                try:
                    images.append(preprocess(Image.open(p).convert("RGB")))
                    valid_paths.append(p)
                except Exception as e:
                    logger.warning(f"Skipping unreadable frame {p}: {e}")

            if not images:
                continue

            image_batch = torch.stack(images).to(_DEVICE)
            with torch.no_grad():
                image_features = model.encode_image(image_batch)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                similarity = (image_features @ text_features.T).squeeze(-1)

            for path, sim in zip(valid_paths, similarity.tolist()):
                try:
                    ts = _timestamp_from_filename(path, fps)
                except ValueError as e:
                    logger.warning(f"Skipping frame with unparseable filename: {e}")
                    continue
                results.append({
                    "timestamp_seconds": ts,
                    "confidence": float(sim),
                })
            
            del images, image_batch, image_features
            gc.collect()
    else:
        # HuggingFace transformers backend
        with torch.no_grad():
            text_inputs = preprocess(text=[query], return_tensors="pt", padding=True).to(_DEVICE)
            text_features = model.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        for i in range(0, len(frame_paths), batch_size):
            batch_paths = frame_paths[i:i + batch_size]
            pil_images, valid_paths = [], []

            for p in batch_paths:
                try:
                    pil_images.append(Image.open(p).convert("RGB"))
                    valid_paths.append(p)
                except Exception as e:
                    logger.warning(f"Skipping unreadable frame {p}: {e}")

            if not pil_images:
                continue

            with torch.no_grad():
                image_inputs = preprocess(images=pil_images, return_tensors="pt", padding=True).to(_DEVICE)
                image_features = model.get_image_features(**image_inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                similarity = (image_features @ text_features.T).squeeze(-1)

            for path, sim in zip(valid_paths, similarity.tolist()):
                try:
                    ts = _timestamp_from_filename(path, fps)
                except ValueError as e:
                    logger.warning(f"Skipping frame with unparseable filename: {e}")
                    continue
                results.append({
                    "timestamp_seconds": ts,
                    "confidence": float(sim),
                })

            del pil_images, image_inputs, image_features
            gc.collect()

    results.sort(key=lambda r: r["timestamp_seconds"])
    return results