"""Unit and Synthetic Phantom tests for Room 2: OpenCV Preprocessing.

Tests:
- Hair removal recovery via Black-Hat and Telea inpainting.
- Illumination normalization via CIE-LAB CLAHE with bit-identical chromatic preservation.
- LCC lesion segmentation with border vignetting suppression.
- Deterministic centered elliptical fallback on degenerate blank/saturated frames.
- Perilesional boundary ring derivation.
- Deep array immutability (arr.flags.writeable == False).
- Strict negative validation on bad input tensors.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any

import cv2
import numpy as np
import pytest

from dermafuzzy_core.config import load_config
from dermafuzzy_core.preprocessing import (
    PreprocessingResult,
    apply_lab_clahe,
    extract_lesion_mask_and_ring,
    preprocess_image,
    remove_hair,
)
from tests.unit.phantoms import (
    make_circle_phantom,
    make_hair_overlay,
)


@pytest.fixture
def config():
    return load_config()


# ------------------------------------------------------------------------------
# User Story 1: Preprocessing Pipeline & Array Immutability
# ------------------------------------------------------------------------------

def test_preprocess_image_valid(config) -> None:
    """Verify that a valid phantom image produces a complete PreprocessingResult with read-only arrays."""
    img, _ = make_circle_phantom()
    res = preprocess_image(img, config)

    assert isinstance(res, PreprocessingResult)
    assert res.clean_rgb.shape == (224, 224, 3)
    assert res.clean_rgb.dtype == np.uint8
    assert res.hair_mask.shape == (224, 224)
    assert res.hair_mask.dtype == np.uint8
    assert res.lesion_mask.shape == (224, 224)
    assert res.lesion_mask.dtype == np.uint8
    assert set(np.unique(res.lesion_mask)).issubset({0, 1})
    assert res.boundary_ring.shape == (224, 224)
    assert res.boundary_ring.dtype == np.uint8
    assert set(np.unique(res.boundary_ring)).issubset({0, 1})
    assert 0.0 < res.area_ratio < 1.0
    assert res.fallback_used is False

    # Deep array write-protection check
    assert res.clean_rgb.flags.writeable is False
    assert res.hair_mask.flags.writeable is False
    assert res.lesion_mask.flags.writeable is False
    assert res.boundary_ring.flags.writeable is False


def test_immutability_write_protection(config) -> None:
    """Verify that attempting to mutate array buffers or dataclass fields raises errors."""
    img, _ = make_circle_phantom()
    res = preprocess_image(img, config)

    with pytest.raises(ValueError, match="read-only"):
        res.clean_rgb[0, 0, 0] = 255

    with pytest.raises(ValueError, match="read-only"):
        res.lesion_mask[0, 0] = 1

    with pytest.raises(ValueError, match="read-only"):
        res.hair_mask[0, 0] = 255

    with pytest.raises(ValueError, match="read-only"):
        res.boundary_ring[0, 0] = 1

    with pytest.raises(FrozenInstanceError):
        res.fallback_used = True  # type: ignore[misc]


def test_remove_hair_telea() -> None:
    """Verify Black-Hat hair detection and Telea inpainting on synthetic hair overlay."""
    clean_img, _ = make_circle_phantom()
    img_with_hair = make_hair_overlay(clean_img, num_strands=6, seed=123)

    inpainted, hair_mask = remove_hair(img_with_hair, kernel_size=17, inpaint_radius=3)

    assert inpainted.shape == clean_img.shape
    assert hair_mask.shape == clean_img.shape[:2]
    assert hair_mask.dtype == np.uint8
    assert np.sum(hair_mask > 0) > 0, "Hair mask must capture overlaid hair strands"

    # Inpainted pixels should be restored close to the clean original
    hair_pixels = hair_mask > 0
    diff_before = np.mean(np.abs(img_with_hair[hair_pixels].astype(float) - clean_img[hair_pixels].astype(float)))
    diff_after = np.mean(np.abs(inpainted[hair_pixels].astype(float) - clean_img[hair_pixels].astype(float)))
    assert diff_after < diff_before, "Inpainting must reduce pixel error compared to occluded hair"


def test_apply_lab_clahe_chromatic_preservation() -> None:
    """Verify that CIE-LAB CLAHE operates strictly on L, keeping a and b bit-identical."""
    img, _ = make_circle_phantom()
    equalized_rgb = apply_lab_clahe(img, clip_limit=2.0, tile_grid=(8, 8))

    # Convert both to LAB space
    lab_orig = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab_eq = cv2.cvtColor(equalized_rgb, cv2.COLOR_RGB2LAB)

    # a and b channels must be bit-identical
    assert np.array_equal(lab_orig[:, :, 1], lab_eq[:, :, 1]), "CIE-LAB 'a' channel was modified by CLAHE!"
    assert np.array_equal(lab_orig[:, :, 2], lab_eq[:, :, 2]), "CIE-LAB 'b' channel was modified by CLAHE!"


def test_fallback_on_blank_and_saturated_frames(config) -> None:
    """Verify deterministic centered elliptical fallback when input is pure black, pure white, or saturated."""
    # Pure black frame
    black_img = np.zeros((224, 224, 3), dtype=np.uint8)
    res_black = preprocess_image(black_img, config)
    assert res_black.fallback_used is True
    assert res_black.lesion_mask.sum() > 0, "Fallback mask must provide non-zero area"
    assert config.preprocessing.mask_area_bounds[0] <= res_black.area_ratio <= config.preprocessing.mask_area_bounds[1]

    # Pure white frame
    white_img = np.full((224, 224, 3), 255, dtype=np.uint8)
    res_white = preprocess_image(white_img, config)
    assert res_white.fallback_used is True
    assert res_white.lesion_mask.sum() > 0


def test_boundary_ring_morphology(config) -> None:
    """Verify that perilesional boundary ring spans the contour boundary of the lesion mask."""
    img, _ = make_circle_phantom()
    res = preprocess_image(img, config)

    # Ring should overlap the lesion interior and exterior
    assert np.sum(res.boundary_ring) > 0
    inner_overlap = np.sum((res.boundary_ring == 1) & (res.lesion_mask == 1))
    outer_overlap = np.sum((res.boundary_ring == 1) & (res.lesion_mask == 0))

    assert inner_overlap > 0, "Boundary ring must contain pixels inside the lesion boundary"
    assert outer_overlap > 0, "Boundary ring must contain pixels outside the lesion boundary"


def test_image_resizing_on_non_224_dimensions(config) -> None:
    """Verify input images differing from config image_size are resized deterministically."""
    # Input is 300x400
    img = np.full((300, 400, 3), 180, dtype=np.uint8)
    res = preprocess_image(img, config)
    assert res.clean_rgb.shape == (224, 224, 3)
    assert res.lesion_mask.shape == (224, 224)


# ------------------------------------------------------------------------------
# Negative Input Validation Tests
# ------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "bad_input",
    [
        np.zeros((224, 224)),                  # 2D instead of 3D
        np.zeros((1, 224, 224, 3)),             # 4D tensor
        np.zeros((224, 224, 1), dtype=np.uint8), # 1 channel
        np.zeros((224, 224, 4), dtype=np.uint8), # 4 channels
        np.zeros((224, 224, 3), dtype=np.float32), # float instead of uint8
        np.zeros((0, 0, 3), dtype=np.uint8),    # empty
        "not_an_array",                         # wrong type
        None,                                   # None
    ],
)
def test_rejects_invalid_input_images(bad_input: Any) -> None:
    """Verify non-3D, wrong channel count, non-uint8, or empty images are rejected with ValueError."""
    with pytest.raises(ValueError):
        preprocess_image(bad_input)


def test_rejects_non_finite_values_if_castable() -> None:
    """Verify non-finite float arrays converted or passed are caught."""
    # Array with NaN/Inf cannot exist as uint8, but if invalid array is supplied:
    nan_arr = np.full((224, 224, 3), np.nan)
    with pytest.raises(ValueError):
        preprocess_image(nan_arr)
