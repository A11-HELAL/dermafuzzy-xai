"""Unit and Synthetic Phantom tests for Room 3: Grad-CAM Saliency Verification.

Tests:
- On-lesion attention overlap (alpha >= 0.85).
- Off-lesion peripheral ruler artifact penalization (alpha <= 0.15).
- Automatic low-resolution (7x7 -> 224x224) bilinear upsampling.
- Zero and negative heatmap handling (alpha == 0.0, D_verified == 0.0).
- Negative tests: bool, NaN, Inf, and out-of-bounds input rejection.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from dermafuzzy_core.gradcam import (
    compute_attention_overlap,
    compute_verified_confidence,
)
from tests.unit.phantoms import (
    make_circle_phantom,
    make_gradcam_heatmap,
)


@pytest.fixture
def circle_mask() -> np.ndarray:
    _, mask = make_circle_phantom((224, 224), radius=50)
    return mask


def test_on_lesion_attention_overlap(circle_mask: np.ndarray) -> None:
    """Verify on-lesion centered heatmap yields alpha >= 0.85 and D_verified >= 0.75."""
    hm = make_gradcam_heatmap((224, 224), on_lesion=True)
    alpha = compute_attention_overlap(hm, circle_mask)
    d_verified = compute_verified_confidence(0.90, alpha)

    assert alpha >= 0.85, f"Expected on-lesion alpha >= 0.85, got {alpha}"
    assert d_verified >= 0.75, f"Expected D_verified >= 0.75, got {d_verified}"


def test_off_lesion_ruler_artifact_penalization(circle_mask: np.ndarray) -> None:
    """Verify off-lesion ruler artifact heatmap yields alpha <= 0.15 and D_verified <= 0.15."""
    hm_ruler = make_gradcam_heatmap((224, 224), on_lesion=False)
    alpha = compute_attention_overlap(hm_ruler, circle_mask)
    d_verified = compute_verified_confidence(0.90, alpha)

    assert alpha <= 0.15, f"Expected off-lesion alpha <= 0.15, got {alpha}"
    assert d_verified <= 0.15, f"Expected penalized D_verified <= 0.15, got {d_verified}"


def test_low_res_7x7_upsampling(circle_mask: np.ndarray) -> None:
    """Verify 7x7 low-res activation map is automatically upsampled to 224x224 and scores correctly."""
    hm_7x7 = make_gradcam_heatmap((224, 224), on_lesion=True, low_res_shape=(7, 7))
    alpha = compute_attention_overlap(hm_7x7, circle_mask)
    d_verified = compute_verified_confidence(0.85, alpha)

    assert alpha >= 0.80, f"Expected upsampled 7x7 alpha >= 0.80, got {alpha}"
    assert d_verified >= 0.68


def test_zero_and_negative_activation_heatmaps(circle_mask: np.ndarray) -> None:
    """Verify zero, None, or all-negative heatmaps return alpha == 0.0 and D_verified == 0.0."""
    zero_hm = np.zeros((224, 224), dtype=np.float32)
    assert compute_attention_overlap(zero_hm, circle_mask) == 0.0

    neg_hm = np.full((224, 224), -1.5, dtype=np.float32)
    assert compute_attention_overlap(neg_hm, circle_mask) == 0.0

    assert compute_attention_overlap(None, circle_mask) == 0.0

    assert compute_verified_confidence(0.95, 0.0) == 0.0


# ------------------------------------------------------------------------------
# Negative Input Validation Tests
# ------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "bad_heatmap",
    [
        np.zeros((224, 224, 3)),            # 3D instead of 2D
        np.zeros((224,)),                   # 1D instead of 2D
        np.full((224, 224), np.nan),         # NaN values
        np.full((224, 224), np.inf),         # Inf values
        "not_an_array",                     # string
        123,                                # integer
    ],
)
def test_rejects_invalid_heatmaps(circle_mask: np.ndarray, bad_heatmap: Any) -> None:
    """Verify invalid dimension or non-finite heatmaps raise ValueError."""
    with pytest.raises(ValueError):
        compute_attention_overlap(bad_heatmap, circle_mask)


@pytest.mark.parametrize(
    ("p_cnn", "alpha"),
    [
        (True, 0.5),
        (0.5, False),
        (float("nan"), 0.5),
        (0.5, float("nan")),
        (float("inf"), 0.5),
        (0.5, float("-inf")),
        (-0.1, 0.5),
        (1.1, 0.5),
        (0.5, -0.05),
        (0.5, 1.05),
    ],
)
def test_rejects_invalid_p_cnn_or_alpha(p_cnn: Any, alpha: Any) -> None:
    """Verify bool, NaN, Inf, or out-of-bounds inputs to compute_verified_confidence raise ValueError."""
    with pytest.raises(ValueError):
        compute_verified_confidence(p_cnn, alpha)
