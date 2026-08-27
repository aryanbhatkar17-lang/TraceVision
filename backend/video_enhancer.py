"""
CLAHE-based Night-Vision Enhancement Pipeline
=============================================
Lightweight, CPU-only frame enhancement for low-light CCTV footage.

Pipeline:
  1. Gamma correction - lifts dark shadows naturally
  2. CLAHE on L-channel - sharpens edges for better AI detection
  3. Mild Gaussian denoise - removes sensor grain
"""

from typing import Optional

import cv2
import numpy as np

# Tunable constants (could be moved to env vars if needed)
_GAMMA: float = 1.5
_CLAHE_CLIP_LIMIT: float = 3.5
_CLAHE_TILE_SIZE: int = 8


def enhance_frame(frame: np.ndarray) -> Optional[np.ndarray]:
    """
    Enhance a single BGR video frame for low-light CCTV conditions.

    Args:
        frame: Input BGR frame (uint8, H x W x 3). Must not be None or empty.

    Returns:
        Enhanced BGR frame (uint8) or None if input is invalid.
    """
    if frame is None or frame.size == 0:
        return None

    try:
        # Step 1: Gamma correction - lifts the darkest shadows naturally
        inv_gamma = 1.0 / _GAMMA
        table = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
            dtype=np.uint8,
        )
        brightened = cv2.LUT(frame, table)

        # Step 2: CLAHE on the L channel - sharpens edges for AI detection
        lab = cv2.cvtColor(brightened, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=_CLAHE_CLIP_LIMIT,
            tileGridSize=(_CLAHE_TILE_SIZE, _CLAHE_TILE_SIZE),
        )
        enhanced_l = clahe.apply(l_channel)

        merged_lab = cv2.merge((enhanced_l, a_channel, b_channel))
        high_contrast = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

        # Step 3: Mild Gaussian denoise - removes sensor grain (3x3 is very fast)
        final = cv2.GaussianBlur(high_contrast, (3, 3), 0)

        return final

    except Exception:
        # In case of unexpected OpenCV errors, return the original frame
        return frame