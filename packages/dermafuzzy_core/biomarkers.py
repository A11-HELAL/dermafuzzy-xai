"""Room 3: Explainable ABCD Clinical Biomarker Extraction.

Provides pure, deterministic, normalized clinical biomarker calculations:
- Asymmetry (A): Centroid and principal inertia axis affine rotation alignment,
  vertical and horizontal XOR folding, normalized to [0.0, 1.0].
- Fuzzy Border Irregularity (B_fuzzy): Hybrid combination of polygon-smoothed geometric
  circularity deficit (B_geo) and perilesional rim gradient fuzziness (B_grad).
- Multi-Channel Color Variation (C): Pixel dispersion across CIE-LAB (L, a, b) and
  HSV (S, V) channels strictly within the lesion interior.
- Attention-Verified Deep Confidence (D_verified): Spatial verification gating raw CNN
  confidence by Grad-CAM attention overlap (alpha).

Governed by Constitution Principles I, II, and IV. Zero top-level heavy ML framework imports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from dermafuzzy_core.config import DermaConfig, load_config
from dermafuzzy_core.gradcam import compute_attention_overlap, compute_verified_confidence


@dataclass(frozen=True, slots=True)
class BiomarkerResult:
    """Encapsulates explainable ABCD clinical features and Grad-CAM saliency metrics."""

    A: float
    B_fuzzy: float
    C: float
    D_verified: float
    B_geo: float
    B_grad: float
    alpha: float

    def as_vector(self) -> tuple[float, float, float, float]:
        """Return the standard 4-biomarker feature vector (A, B_fuzzy, C, D_verified)."""
        return (self.A, self.B_fuzzy, self.C, self.D_verified)

    def as_dict(self) -> dict[str, float]:
        """Return all computed biomarker features and diagnostic sub-scores as a dictionary."""
        return {
            "A": self.A,
            "B_fuzzy": self.B_fuzzy,
            "C": self.C,
            "D_verified": self.D_verified,
            "B_geo": self.B_geo,
            "B_grad": self.B_grad,
            "alpha": self.alpha,
        }


def _validate_mask(mask: Any) -> np.ndarray:
    """Validate that input is a 2D NumPy array representing a binary mask."""
    if not isinstance(mask, np.ndarray):
        raise ValueError(f"Mask must be a numpy.ndarray, got {type(mask).__name__}")
    if mask.ndim != 2:
        raise ValueError(f"Mask must be a 2D array, got shape {mask.shape}")
    if mask.shape[0] <= 0 or mask.shape[1] <= 0:
        raise ValueError(f"Mask dimensions must be positive, got shape {mask.shape}")
    if not np.all(np.isfinite(mask)):
        raise ValueError("Mask contains non-finite (NaN or Inf) values")
    return mask


def _validate_rgb(clean_rgb: Any) -> np.ndarray:
    """Validate that input is a 3D uint8 RGB image."""
    if not isinstance(clean_rgb, np.ndarray):
        raise ValueError(f"clean_rgb must be a numpy.ndarray, got {type(clean_rgb).__name__}")
    if clean_rgb.ndim != 3 or clean_rgb.shape[2] != 3:
        raise ValueError(f"clean_rgb must be a 3D array (H, W, 3), got shape {clean_rgb.shape}")
    if clean_rgb.dtype != np.uint8:
        raise ValueError(f"clean_rgb must have dtype uint8, got {clean_rgb.dtype}")
    if clean_rgb.shape[0] <= 0 or clean_rgb.shape[1] <= 0:
        raise ValueError(f"clean_rgb dimensions must be positive, got shape {clean_rgb.shape}")
    return clean_rgb


def compute_asymmetry(
    mask: Any,
    eps: float = 1e-6,
) -> float:
    """Compute rotation-invariant asymmetry score A in [0.0, 1.0].

    Parameters
    ----------
    mask : Any
        Binary lesion mask, 2D array in {0, 1}.
    eps : float, default=1e-6
        Denominator stabilizer.

    Returns
    -------
    float
        Asymmetry score A in [0.0, 1.0]. Ideal circle <= 0.05; L-shape >= 0.35.
    """
    m = _validate_mask(mask)
    mask_u8 = (m > 0).astype(np.uint8)

    M = cv2.moments(mask_u8)
    m00 = float(M["m00"])
    if m00 <= eps:
        return 0.0

    cx = float(M["m10"]) / (m00 + eps)
    cy = float(M["m01"]) / (m00 + eps)

    mu20 = float(M["mu20"])
    mu02 = float(M["mu02"])
    mu11 = float(M["mu11"])

    # Principal inertia axis angle
    theta = 0.5 * math.atan2(2.0 * mu11, mu20 - mu02)
    theta_deg = math.degrees(theta)

    h, w = mask_u8.shape[:2]
    target_cx = (w - 1) / 2.0
    target_cy = (h - 1) / 2.0

    # Affine transformation: rotate by theta_deg around (cx, cy) and shift to image center
    rot_mat = cv2.getRotationMatrix2D((cx, cy), theta_deg, 1.0)
    rot_mat[0, 2] += (target_cx - cx)
    rot_mat[1, 2] += (target_cy - cy)

    aligned = cv2.warpAffine(
        mask_u8,
        rot_mat,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    aligned_bin = (aligned > 0)
    area = float(aligned_bin.sum())
    if area <= eps:
        return 0.0

    # Fold across horizontal and vertical axes via XOR (divide by 2 to count single-sided mismatched pairs)
    xor_h = float(np.logical_xor(aligned_bin, np.fliplr(aligned_bin)).sum()) / 2.0
    xor_v = float(np.logical_xor(aligned_bin, np.flipud(aligned_bin)).sum()) / 2.0

    asym = (xor_h + xor_v) / (area + eps)
    return float(np.clip(asym, 0.0, 1.0))


def compute_border_fuzziness(
    clean_rgb: Any,
    mask: Any,
    boundary_ring: Any,
    config: DermaConfig | None = None,
) -> tuple[float, float, float]:
    """Compute hybrid border irregularity B_fuzzy, geometric score B_geo, and gradient score B_grad.

    Parameters
    ----------
    clean_rgb : Any
        Cleaned RGB image, shape (H, W, 3), dtype uint8.
    mask : Any
        Binary lesion mask in {0, 1}.
    boundary_ring : Any
        Binary boundary band in {0, 1}.
    config : DermaConfig | None, optional
        Loaded configuration containing b_fuzzy_weights.

    Returns
    -------
    tuple[float, float, float]
        (B_fuzzy, B_geo, B_grad), all strictly clipped to [0.0, 1.0].
    """
    rgb = _validate_rgb(clean_rgb)
    m = _validate_mask(mask)
    ring = _validate_mask(boundary_ring)

    if config is None:
        config = load_config()

    eps = float(config.preprocessing.eps)
    mask_u8 = (m > 0).astype(np.uint8)

    # --------------------------------------------------------------------------
    # 1. Geometric Jaggedness (B_geo)
    # --------------------------------------------------------------------------
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        B_geo = 0.0
    else:
        c = max(contours, key=cv2.contourArea)
        raw_p = float(cv2.arcLength(c, True))
        # Polygon approximation smoothing to remove grid staircase discretization
        approx_c = cv2.approxPolyDP(c, 0.004 * raw_p, True)
        p = float(cv2.arcLength(approx_c, True))
        area = float(cv2.contourArea(approx_c))
        if area <= eps or p <= eps:
            area = float(mask_u8.sum())

        circ = (4.0 * math.pi * area) / (p * p + eps)
        B_geo = float(np.clip(1.0 - circ, 0.0, 1.0))

    # --------------------------------------------------------------------------
    # 2. Perilesional Rim Gradient Fuzziness (B_grad)
    # --------------------------------------------------------------------------
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0].astype(np.float64) / 255.0

    gx = cv2.Sobel(L, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(L, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.hypot(gx, gy)

    ring_pixels = (ring > 0)
    if not np.any(ring_pixels):
        B_grad = 0.0
    else:
        ring_vals = grad_mag[ring_pixels]
        mean_val = float(np.mean(ring_vals))
        std_val = float(np.std(ring_vals))
        if mean_val <= eps:
            B_grad = 1.0
        else:
            # High peak relative to mean indicates a sharp step edge (B_grad -> 0.0);
            # Uniform diffuse gradient across the ring indicates a blurry rim (B_grad -> 1.0).
            rel_sharpness = std_val / (mean_val + eps)
            B_grad = float(np.clip(1.0 - rel_sharpness, 0.0, 1.0))

    # --------------------------------------------------------------------------
    # 3. Fuse with Configured Weights
    # --------------------------------------------------------------------------
    w_geo = float(config.biomarkers.b_fuzzy_weights["w_geo"])
    w_grad = float(config.biomarkers.b_fuzzy_weights["w_grad"])
    B_fuzzy = float(np.clip(w_geo * B_geo + w_grad * B_grad, 0.0, 1.0))

    return B_fuzzy, B_geo, B_grad


def compute_color_variation(
    clean_rgb: Any,
    mask: Any,
    eps: float = 1e-6,
) -> float:
    """Compute multi-channel color variation score C in [0.0, 1.0].

    Parameters
    ----------
    clean_rgb : Any
        Cleaned RGB image, shape (H, W, 3), dtype uint8.
    mask : Any
        Binary lesion mask in {0, 1}.
    eps : float, default=1e-6
        Denominator stabilizer.

    Returns
    -------
    float
        Color variation score C in [0.0, 1.0]. Uniform patch <= 0.03; multi-pigment >= 0.45.
    """
    rgb = _validate_rgb(clean_rgb)
    m = _validate_mask(mask)

    lesion_pixels = (m > 0)
    if not np.any(lesion_pixels):
        return 0.0

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    # 5 complementary channels: Luminance (L), Opponent A (a), Opponent B (b), Saturation (S), Value (V)
    channels = [
        lab[:, :, 0],
        lab[:, :, 1],
        lab[:, :, 2],
        hsv[:, :, 1],
        hsv[:, :, 2],
    ]

    dispersions: list[float] = []
    for ch in channels:
        vals = ch[lesion_pixels].astype(np.float64) / 255.0
        std_val = float(np.std(vals))
        p90 = float(np.percentile(vals, 90))
        p10 = float(np.percentile(vals, 10))
        idr_val = p90 - p10
        # Combine standard deviation with robust inter-decile spread
        dispersions.append(0.5 * (std_val * 2.0 + idr_val))

    # Scale so solid colors evaluate to ~0.0 while 4-pigment quadrants evaluate to >= 0.45
    c_score = 1.35 * float(np.mean(dispersions))
    return float(np.clip(c_score, 0.0, 1.0))


def extract_abcd_biomarkers(
    clean_rgb: Any,
    mask: Any,
    boundary_ring: Any,
    config: DermaConfig | None = None,
    p_cnn: float = 0.0,
    heatmap: Any = None,
) -> BiomarkerResult:
    """Orchestrate end-to-end ABCD feature extraction and attention verification.

    Parameters
    ----------
    clean_rgb : Any
        Cleaned RGB image, shape (H, W, 3), dtype uint8.
    mask : Any
        Binary lesion mask in {0, 1}.
    boundary_ring : Any
        Binary boundary band in {0, 1}.
    config : DermaConfig | None, optional
        Loaded configuration.
    p_cnn : float, default=0.0
        Raw deep learning classifier prediction in [0.0, 1.0].
    heatmap : Any, optional
        2D Grad-CAM saliency activation map.

    Returns
    -------
    BiomarkerResult
        Immutable dataclass holding all computed scores and diagnostic sub-metrics.
    """
    if config is None:
        config = load_config()

    eps = float(config.preprocessing.eps)

    A = compute_asymmetry(mask, eps=eps)
    B_fuzzy, B_geo, B_grad = compute_border_fuzziness(clean_rgb, mask, boundary_ring, config=config)
    C = compute_color_variation(clean_rgb, mask, eps=eps)
    alpha = compute_attention_overlap(heatmap, mask, eps=eps)
    D_verified = compute_verified_confidence(p_cnn, alpha)

    return BiomarkerResult(
        A=A,
        B_fuzzy=B_fuzzy,
        C=C,
        D_verified=D_verified,
        B_geo=B_geo,
        B_grad=B_grad,
        alpha=alpha,
    )
