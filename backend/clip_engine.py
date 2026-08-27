"""
clip_engine.py
================
Loads CLIP ViT-B/32 once as a singleton and exposes evaluate_video_frames(),
which scores every extracted JPEG in a directory against a text query using
cosine similarity between CLIP's image and text embeddings.

Frame filenames must match pipeline.py's _extract_frames_opencv() output:
"frame_<timestamp_seconds:.2f>.jpg" e.g. "frame_12.00.jpg" -> 12.00
"""

import os
import re
import glob
import logging
from typing import List, Dict

import torch
import clip  # pip install git+https://github.com/openai/CLIP.git
from PIL import Image

logger = logging.getLogger("tracevision.clip")

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_MODEL = None
_PREPROCESS = None
_FRAME_FILENAME_RE = re.compile(r"frame_([0-9]+\.[0-9]+)\.jpg$")


def _load_model():
    """Loads ViT-B/32 exactly once per process — reloading per request
    would be slow and waste VRAM under your 6GB ceiling."""
    global _MODEL, _PREPROCESS
    if _MODEL is None:
        logger.info(f"Loading CLIP ViT-B/32 on device={_DEVICE} ...")
        _MODEL, _PREPROCESS = clip.load("ViT-B/32", device=_DEVICE)
        _MODEL.eval()
        logger.info("CLIP ViT-B/32 loaded.")
    return _MODEL, _PREPROCESS


def _timestamp_from_filename(path: str) -> float:
    match = _FRAME_FILENAME_RE.search(os.path.basename(path))
    if not match:
        raise ValueError(f"Frame filename doesn't match expected pattern: {path}")
    return float(match.group(1))


def evaluate_video_frames(frames_dir: str, query: str, batch_size: int = 32) -> List[Dict[str, float]]:
    """
    Scores every frame in frames_dir against `query` using CLIP cosine similarity.
    Returns: [{"timestamp_seconds": float, "confidence": float}, ...]
    Confidence is raw CLIP cosine similarity (typically ~0.20-0.35 for real
    matches on natural photos) — this matches the numeric range you were
    already observing, so pipeline.py's noise floor / top-k logic doesn't
    need retuning.
    """
    model, preprocess = _load_model()

    frame_paths = sorted(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))
    if not frame_paths:
        logger.warning(f"No frames found in {frames_dir}")
        return []

    text_tokens = clip.tokenize([query]).to(_DEVICE)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    results: List[Dict[str, float]] = []

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
            results.append({
                "timestamp_seconds": _timestamp_from_filename(path),
                "confidence": float(sim),
            })

    results.sort(key=lambda r: r["timestamp_seconds"])
    return results