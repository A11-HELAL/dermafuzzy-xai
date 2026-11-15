"""Room 3: Grad-CAM Saliency Verification.

Provides spatial attention verification to ground deep neural network predictions:
- Clamps raw activation maps with ReLU (np.maximum(H, 0.0)).
- Bilinear upsampling of arbitrary 2D heatmaps (e.g. 7x7 MobileNetV2 -> 224x224).
- Calculates on-lesion attention overlap proportion alpha in [0.0, 1.0].
- Computes verified deep confidence D_verified = np.clip(p_cnn * alpha, 0.0, 1.0).

Governed by Constitution Principles I, II, and IV. Zero top-level heavy ML framework imports.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


def compute_attention_overlap(
    heatmap: Any,
    mask: Any,
    eps: float = 1e-6,
) -> float:
    """Calculate spatial attention overlap alpha in [0.0, 1.0].

    Parameters
    ----------
    heatmap : Any
        2D floating-point Grad-CAM activation map. If None, returns 0.0.
    mask : Any
        Binary lesion mask in {0, 1}, 2D array.
    eps : float, default=1e-6
        Denominator stabilizer.

    Returns
    -------
    float
        Attention overlap ratio alpha in [0.0, 1.0].

    Raises
    ------
    ValueError
        If heatmap or mask dimensions or values are invalid.
    """
    if heatmap is None:
        return 0.0

    if not isinstance(heatmap, np.ndarray):
        raise ValueError(f"Heatmap must be a numpy.ndarray, got {type(heatmap).__name__}")

    if heatmap.ndim != 2:
        raise ValueError(f"Heatmap must be a 2D array, got shape {heatmap.shape}")

    if not np.all(np.isfinite(heatmap)):
        raise ValueError("Heatmap contains non-finite (NaN or Inf) values")

    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise ValueError("Mask must be a 2D numpy.ndarray")

    # Apply ReLU: clamp negative values to 0.0
    hm = np.maximum(heatmap.astype(np.float64), 0.0)

    # Check if total activation is negligible
    total_activation = float(hm.sum())
    if total_activation <= eps:
        return 0.0

    # Resize if spatial resolution differs from mask
    mh, mw = mask.shape[:2]
    hh, hw = hm.shape[:2]
    if (hw, hh) != (mw, mh):
        hm = cv2.resize(hm, (mw, mh), interpolation=cv2.INTER_LINEAR)
        total_activation = float(hm.sum())
        if total_activation <= eps:
            return 0.0

    # Calculate attention overlap on lesion (mask > 0)
    lesion_activation = float(np.sum(hm[mask > 0]))
    alpha = lesion_activation / (total_activation + eps)

    return float(np.clip(alpha, 0.0, 1.0))


def compute_verified_confidence(
    p_cnn: Any,
    alpha: Any,
) -> float:
    """Compute attention-verified deep confidence D_verified = np.clip(p_cnn * alpha, 0.0, 1.0).

    Parameters
    ----------
    p_cnn : Any
        Raw deep learning classifier prediction in [0.0, 1.0].
    alpha : Any
        Spatial attention overlap ratio in [0.0, 1.0].

    Returns
    -------
    float
        Verified confidence score D_verified in [0.0, 1.0].

    Raises
    ------
    ValueError
        If p_cnn or alpha are boolean, non-finite, or outside [0.0, 1.0].
    """
    if isinstance(p_cnn, bool) or not isinstance(p_cnn, (int, float)):
        raise ValueError(f"p_cnn must be a float, got {type(p_cnn).__name__}")
    f_p_cnn = float(p_cnn)
    if not math.isfinite(f_p_cnn):
        raise ValueError(f"p_cnn must be a finite float, got {f_p_cnn}")
    if not (0.0 <= f_p_cnn <= 1.0):
        raise ValueError(f"p_cnn must be within [0.0, 1.0], got {f_p_cnn}")

    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ValueError(f"alpha must be a float, got {type(alpha).__name__}")
    f_alpha = float(alpha)
    if not math.isfinite(f_alpha):
        raise ValueError(f"alpha must be a finite float, got {f_alpha}")
    if not (0.0 <= f_alpha <= 1.0):
        raise ValueError(f"alpha must be within [0.0, 1.0], got {f_alpha}")

    d_verified = f_p_cnn * f_alpha
    return float(np.clip(d_verified, 0.0, 1.0))
