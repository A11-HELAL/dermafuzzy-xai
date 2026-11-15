"""Synthetic Dermoscopic Phantom Generators for DermaFuzzy-XAI Testing.

Provides deterministic geometric phantoms, artifact overlays, and saliency maps
for benchmarking Room 2 (preprocessing) and Room 3 (biomarkers & gradcam).
"""

from __future__ import annotations

import math
from typing import Tuple

import cv2
import numpy as np

# Standard realistic skin and pigment colors (RGB uint8)
SKIN_TONE = (220, 190, 170)
LESION_DARK_BROWN = (70, 45, 30)


def make_circle_phantom(
    size: Tuple[int, int] = (224, 224),
    radius: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate an ideal circular lesion phantom.

    Returns
    -------
    img_rgb : np.ndarray
        RGB image of shape (H, W, 3), dtype uint8.
    mask : np.ndarray
        Binary mask of shape (H, W), dtype uint8 in {0, 1}.
    """
    h, w = size
    cx, cy = w // 2, h // 2

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), radius, 1, -1)

    img_rgb = np.full((h, w, 3), SKIN_TONE, dtype=np.uint8)
    img_rgb[mask == 1] = LESION_DARK_BROWN

    return img_rgb, mask


def make_l_shape_phantom(
    size: Tuple[int, int] = (224, 224),
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a highly asymmetric L-shaped lesion phantom.

    Returns
    -------
    img_rgb : np.ndarray
        RGB image of shape (H, W, 3), dtype uint8.
    mask : np.ndarray
        Binary mask of shape (H, W), dtype uint8 in {0, 1}.
    """
    h, w = size
    mask = np.zeros((h, w), dtype=np.uint8)

    # Vertical bar of L
    mask[60:164, 80:110] = 1
    # Horizontal bar of L
    mask[134:164, 110:160] = 1

    img_rgb = np.full((h, w, 3), SKIN_TONE, dtype=np.uint8)
    img_rgb[mask == 1] = LESION_DARK_BROWN

    return img_rgb, mask


def make_star_phantom(
    size: Tuple[int, int] = (224, 224),
    num_points: int = 7,
    r_inner: int = 25,
    r_outer: int = 65,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a jagged star contour with severe circularity deficit (B_geo >= 0.50).

    Returns
    -------
    img_rgb : np.ndarray
        RGB image of shape (H, W, 3), dtype uint8.
    mask : np.ndarray
        Binary mask of shape (H, W), dtype uint8 in {0, 1}.
    """
    h, w = size
    cx, cy = w // 2, h // 2

    vertices = []
    total_steps = num_points * 2
    for i in range(total_steps):
        angle = i * math.pi / num_points
        r = r_outer if (i % 2 == 0) else r_inner
        x = int(cx + r * math.cos(angle))
        y = int(cy + r * math.sin(angle))
        vertices.append([x, y])

    pts = np.array(vertices, dtype=np.int32).reshape((-1, 1, 2))

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)

    img_rgb = np.full((h, w, 3), SKIN_TONE, dtype=np.uint8)
    img_rgb[mask == 1] = LESION_DARK_BROWN

    return img_rgb, mask


def make_hair_overlay(
    img_rgb: np.ndarray,
    num_strands: int = 6,
    seed: int = 42,
) -> np.ndarray:
    """Overlay thin, dark, curved hair strands onto an image.

    Returns
    -------
    np.ndarray
        RGB image with dark hair fibers, dtype uint8.
    """
    img_hair = img_rgb.copy()
    h, w = img_hair.shape[:2]
    rng = np.random.default_rng(seed)

    hair_color = (25, 20, 15)  # Very dark hair fiber

    for _ in range(num_strands):
        # Generate quadratic bezier curve
        p0 = (int(rng.uniform(0, w * 0.3)), int(rng.uniform(0, h)))
        p2 = (int(rng.uniform(w * 0.7, w)), int(rng.uniform(0, h)))
        p1 = (int(rng.uniform(w * 0.3, w * 0.7)), int(rng.uniform(0, h)))

        curve_pts = []
        for t in np.linspace(0.0, 1.0, 50):
            x = int((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0])
            y = int((1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1])
            curve_pts.append((x, y))

        for j in range(len(curve_pts) - 1):
            cv2.line(img_hair, curve_pts[j], curve_pts[j + 1], hair_color, thickness=2, lineType=cv2.LINE_AA)

    return img_hair


def make_sharp_step_phantom(
    size: Tuple[int, int] = (224, 224),
    radius: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a crisp step-edge lesion phantom with sharp perilesional transition.

    Returns
    -------
    img_rgb : np.ndarray
        RGB image with crisp boundary step.
    mask : np.ndarray
        Binary mask {0, 1}.
    boundary_ring : np.ndarray
        Boundary ring {0, 1} derived via 5x5 kernel dilation - erosion.
    """
    img_rgb, mask = make_circle_phantom(size, radius)
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(mask, k5)
    eroded = cv2.erode(mask, k5)
    boundary_ring = (dilated - eroded).astype(np.uint8)

    return img_rgb, mask, boundary_ring


def make_blurred_rim_phantom(
    size: Tuple[int, int] = (224, 224),
    radius: int = 50,
    sigma: float = 7.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a lesion phantom with a Gaussian-feathered, blurry perilesional rim.

    Returns
    -------
    img_rgb : np.ndarray
        RGB image with feathered blurry boundary transition.
    mask : np.ndarray
        Binary mask {0, 1}.
    boundary_ring : np.ndarray
        Boundary ring {0, 1}.
    """
    h, w = size
    _, mask = make_circle_phantom(size, radius)

    # Smooth the mask boundary heavily using Gaussian blur
    mask_float = mask.astype(np.float64)
    feathered = cv2.GaussianBlur(mask_float, (0, 0), sigmaX=sigma, sigmaY=sigma)

    skin = np.array(SKIN_TONE, dtype=np.float64)
    lesion = np.array(LESION_DARK_BROWN, dtype=np.float64)

    # Linear blend
    blended = (1.0 - feathered[:, :, None]) * skin + feathered[:, :, None] * lesion
    img_rgb = np.clip(blended, 0, 255).astype(np.uint8)

    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(mask, k5)
    eroded = cv2.erode(mask, k5)
    boundary_ring = (dilated - eroded).astype(np.uint8)

    return img_rgb, mask, boundary_ring


def make_uniform_phantom(
    size: Tuple[int, int] = (224, 224),
    radius: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a uniform single-color lesion phantom (C <= 0.03)."""
    return make_circle_phantom(size, radius)


def make_multi_pigment_phantom(
    size: Tuple[int, int] = (224, 224),
    radius: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a multi-pigment quadrant lesion (brown, black, red, blue-gray) (C >= 0.45).

    Returns
    -------
    img_rgb : np.ndarray
        Multi-colored lesion on skin background.
    mask : np.ndarray
        Binary lesion mask.
    """
    h, w = size
    cx, cy = w // 2, h // 2

    _, mask = make_circle_phantom(size, radius)
    img_rgb = np.full((h, w, 3), SKIN_TONE, dtype=np.uint8)

    # 4 distinct dermatological colors
    color_top_left = (70, 40, 20)      # Dark brown
    color_top_right = (15, 15, 15)     # Deep black
    color_bottom_left = (175, 45, 45)  # Erythematous red
    color_bottom_right = (75, 95, 125) # Slate blue-gray

    for y in range(h):
        for x in range(w):
            if mask[y, x] == 1:
                if y < cy and x < cx:
                    img_rgb[y, x] = color_top_left
                elif y < cy and x >= cx:
                    img_rgb[y, x] = color_top_right
                elif y >= cy and x < cx:
                    img_rgb[y, x] = color_bottom_left
                else:
                    img_rgb[y, x] = color_bottom_right

    return img_rgb, mask


def make_gradcam_heatmap(
    size: Tuple[int, int] = (224, 224),
    on_lesion: bool = True,
    low_res_shape: Tuple[int, int] | None = None,
) -> np.ndarray:
    """Generate a synthetic 2D Grad-CAM heatmap.

    Parameters
    ----------
    size : Tuple[int, int]
        Target spatial size (H, W).
    on_lesion : bool
        If True, activation is centered on lesion; if False, activation is on the periphery.
    low_res_shape : Tuple[int, int] | None
        If provided (e.g. (7, 7)), generates heatmap at low resolution.

    Returns
    -------
    np.ndarray
        2D float32 heatmap with values in [0.0, 1.0].
    """
    target_shape = low_res_shape if low_res_shape is not None else size
    th, tw = target_shape

    hm = np.zeros((th, tw), dtype=np.float32)

    if on_lesion:
        # Centered Gaussian blob concentrated over lesion
        cx, cy = (tw - 1) / 2.0, (th - 1) / 2.0
        sigma = min(tw, th) * 0.08  # tightly concentrated on lesion interior
    else:
        # Peripheral corner artifact (e.g. ruler marking)
        cx, cy = tw * 0.08, th * 0.08
        sigma = min(tw, th) * 0.06

    y_grid, x_grid = np.ogrid[:th, :tw]
    dist_sq = (x_grid - cx) ** 2 + (y_grid - cy) ** 2
    hm = np.exp(-dist_sq / (2.0 * sigma ** 2)).astype(np.float32)

    # Normalize to [0.0, 1.0]
    max_val = float(hm.max())
    if max_val > 0.0:
        hm /= max_val

    return hm
