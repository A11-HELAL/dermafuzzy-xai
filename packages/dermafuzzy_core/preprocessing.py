"""Room 2: OpenCV Preprocessing and Perilesional Boundary Ring Isolation.

Provides 100% pure, deterministic, write-protected dermoscopic preprocessing:
- Hair artifact removal via Grayscale Morphological Black-Hat and Telea inpainting.
- Illumination normalization via CIE-LAB CLAHE strictly on the L channel.
- Lesion segmentation via Gaussian blur, inverse Otsu thresholding, border vignetting suppression,
  and Largest Connected Component (LCC) extraction.
- Deterministic centered elliptical fallback mask preventing zero-division on degenerate inputs.
- Morphological perilesional boundary ring derivation (R_edge).
- Deep array write-protection (arr.flags.writeable = False).

Governed by Constitution Principles I, II, and IV. Zero top-level heavy ML framework imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from dermafuzzy_core.config import DermaConfig, load_config


@dataclass(frozen=True, slots=True)
class PreprocessingResult:
    """Encapsulates preprocessed image buffers and lesion segmentation masks."""

    clean_rgb: np.ndarray
    hair_mask: np.ndarray
    lesion_mask: np.ndarray
    boundary_ring: np.ndarray
    area_ratio: float
    fallback_used: bool


def _validate_rgb_image(img_rgb: Any) -> np.ndarray:
    """Validate that input is a valid 3D uint8 RGB NumPy array."""
    if not isinstance(img_rgb, np.ndarray):
        raise ValueError(f"Input image must be a numpy.ndarray, got {type(img_rgb).__name__}")

    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError(f"Input image must be a 3-channel 3D array (H, W, 3), got shape {img_rgb.shape}")

    if img_rgb.dtype != np.uint8:
        raise ValueError(f"Input image must have dtype uint8, got {img_rgb.dtype}")

    if img_rgb.shape[0] <= 0 or img_rgb.shape[1] <= 0:
        raise ValueError(f"Input image dimensions must be positive, got shape {img_rgb.shape}")

    return img_rgb


def remove_hair(
    img_rgb: np.ndarray,
    kernel_size: int = 17,
    inpaint_radius: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Isolate dark hair occlusions via Black-Hat morphology and restore texture via Telea inpainting.

    Parameters
    ----------
    img_rgb : np.ndarray
        RGB image of shape (H, W, 3) and dtype uint8.
    kernel_size : int, default=17
        Diameter of elliptical structuring element. Must be positive odd integer.
    inpaint_radius : int, default=3
        Neighborhood radius for Telea inpainting.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (inpainted_rgb, hair_mask) where hair_mask is uint8 in {0, 255}.
    """
    _validate_rgb_image(img_rgb)
    if isinstance(kernel_size, bool) or not isinstance(kernel_size, int) or kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
    if isinstance(inpaint_radius, bool) or not isinstance(inpaint_radius, int) or inpaint_radius <= 0:
        raise ValueError(f"inpaint_radius must be a positive integer, got {inpaint_radius}")

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # Dynamic thresholding: if max blackhat response is very low, uniform skin with no hairs
    if float(blackhat.max()) < 12.0:
        hair_mask = np.zeros(gray.shape, dtype=np.uint8)
        inpainted_rgb = img_rgb.copy()
    else:
        # Isolate top dark fiber responses
        _, hair_thresh = cv2.threshold(blackhat, 14, 255, cv2.THRESH_BINARY)
        # Suppress single-pixel noise with mild opening
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        hair_mask = cv2.morphologyEx(hair_thresh, cv2.MORPH_OPEN, k3)

        if int(hair_mask.sum()) > 0:
            inpainted_rgb = cv2.inpaint(img_rgb, hair_mask, inpaint_radius, cv2.INPAINT_TELEA)
        else:
            inpainted_rgb = img_rgb.copy()

    return inpainted_rgb, hair_mask


def apply_lab_clahe(
    img_rgb: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply CLAHE strictly to the L channel in CIE-LAB space, keeping a and b bit-identical.

    Parameters
    ----------
    img_rgb : np.ndarray
        RGB image of shape (H, W, 3), dtype uint8.
    clip_limit : float, default=2.0
        Contrast limit for adaptive histogram equalization.
    tile_grid : tuple[int, int], default=(8, 8)
        Tile grid dimensions (w, h).

    Returns
    -------
    np.ndarray
        Illumination-equalized RGB image, shape (H, W, 3), dtype uint8.
    """
    _validate_rgb_image(img_rgb)
    if isinstance(clip_limit, bool) or not isinstance(clip_limit, (int, float)) or clip_limit <= 0.0:
        raise ValueError(f"clip_limit must be a positive float, got {clip_limit}")

    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(int(tile_grid[0]), int(tile_grid[1])),
    )
    # Strictly operate on L channel (index 0)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])

    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def extract_lesion_mask_and_ring(
    clean_rgb: np.ndarray,
    config: DermaConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """Segment primary lesion mask M and derive perilesional boundary ring R_edge.

    Parameters
    ----------
    clean_rgb : np.ndarray
        Cleaned RGB image, shape (H, W, 3), dtype uint8.
    config : DermaConfig | None, optional
        Loaded configuration.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, float, bool]
        (lesion_mask, boundary_ring, area_ratio, fallback_used)
        Both masks are uint8 in {0, 1}.
    """
    _validate_rgb_image(clean_rgb)
    if config is None:
        config = load_config()

    h, w = clean_rgb.shape[:2]

    # Convert to LAB to extract luminance
    lab = cv2.cvtColor(clean_rgb, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0]

    # Step 1: Gaussian smoothing
    g_k = config.preprocessing.gaussian_blur_kernel
    blurred = cv2.GaussianBlur(L, (g_k, g_k), 0)

    # Step 2: Inverse Otsu thresholding (lesion is typically darker than skin)
    _, binary = cv2.threshold(blurred, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Step 3: Border-vignetting suppression
    # Sample outer 2% perimeter
    bw = max(1, int(0.02 * min(h, w)))
    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[:bw, :] = True
    border_mask[-bw:, :] = True
    border_mask[:, :bw] = True
    border_mask[:, -bw:] = True

    border_fg_ratio = float(binary[border_mask].mean())
    center_fg_ratio = float(binary[~border_mask].mean())
    if border_fg_ratio > 0.50 and center_fg_ratio < border_fg_ratio:
        # Optical vignetting artifact detected; clear border frame
        binary[border_mask] = 0

    # Step 4: Largest Connected Component (LCC)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        mask = np.zeros((h, w), dtype=np.uint8)
    else:
        # Find largest foreground label (label > 0)
        largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = (labels == largest_label).astype(np.uint8)

    # Fill internal holes inside the primary lesion mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(mask, contours, -1, 1, thickness=-1)

    area_ratio = float(mask.sum()) / float(h * w)
    min_bound, max_bound = config.preprocessing.mask_area_bounds
    fallback_used = False

    # Step 5: Fallback check
    if area_ratio < min_bound or area_ratio > max_bound:
        # Replace with deterministic centered ellipse
        mask = np.zeros((h, w), dtype=np.uint8)
        axes = (int(0.35 * w), int(0.35 * h))
        cv2.ellipse(mask, (w // 2, h // 2), axes, 0, 0, 360, 1, -1)
        area_ratio = float(mask.sum()) / float(h * w)
        fallback_used = True

    # Step 6: Perilesional Boundary Ring (R_edge = Dilate - Erode)
    r_k = config.preprocessing.boundary_ring_kernel
    k_ring = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r_k, r_k))
    m_outer = cv2.dilate(mask, k_ring)
    m_inner = cv2.erode(mask, k_ring)
    boundary_ring = (m_outer - m_inner).astype(np.uint8)

    return mask, boundary_ring, area_ratio, fallback_used


def preprocess_image(
    img_rgb: np.ndarray,
    config: DermaConfig | None = None,
) -> PreprocessingResult:
    """Preprocess a raw dermoscopic image end-to-end with deep array immutability.

    Parameters
    ----------
    img_rgb : np.ndarray
        Input RGB image, shape (H, W, 3) and dtype uint8.
    config : DermaConfig | None, optional
        Loaded configuration. If None, loads root configuration.

    Returns
    -------
    PreprocessingResult
        Immutable dataclass holding read-only array buffers and segmentation state.

    Raises
    ------
    ValueError
        If input array is invalid, wrong shape, or non-finite.
    """
    img = _validate_rgb_image(img_rgb)
    if config is None:
        config = load_config()

    # Step 0: Deterministic resize if dimensions mismatch config
    target_w, target_h = config.dataset.image_size
    h, w = img.shape[:2]
    if (w, h) != (target_w, target_h):
        interp = cv2.INTER_AREA if (w > target_w or h > target_h) else cv2.INTER_LINEAR
        img = cv2.resize(img, (target_w, target_h), interpolation=interp)

    # Step 1: Hair inpainting
    inpainted_rgb, hair_mask = remove_hair(
        img,
        kernel_size=config.preprocessing.blackhat_kernel_size,
        inpaint_radius=config.preprocessing.inpaint_radius,
    )

    # Step 2: CIE-LAB CLAHE illumination normalization
    clean_rgb = apply_lab_clahe(
        inpainted_rgb,
        clip_limit=config.preprocessing.clahe_clip_limit,
        tile_grid=config.preprocessing.clahe_tile_grid,
    )

    # Step 3 & 4: Lesion mask, fallback gate, and boundary ring
    lesion_mask, boundary_ring, area_ratio, fallback_used = extract_lesion_mask_and_ring(
        clean_rgb,
        config=config,
    )

    # Step 5: Lock all output arrays as read-only
    clean_rgb.flags.writeable = False
    hair_mask.flags.writeable = False
    lesion_mask.flags.writeable = False
    boundary_ring.flags.writeable = False

    return PreprocessingResult(
        clean_rgb=clean_rgb,
        hair_mask=hair_mask,
        lesion_mask=lesion_mask,
        boundary_ring=boundary_ring,
        area_ratio=area_ratio,
        fallback_used=fallback_used,
    )
