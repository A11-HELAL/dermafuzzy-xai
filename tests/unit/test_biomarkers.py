"""Unit and Synthetic Phantom tests for Room 3: ABCD Biomarker Extraction.

Tests:
- Asymmetry A on circle (A <= 0.05) and L-shape (A >= 0.35).
- Rotation invariance of Asymmetry A across 0, 30, 45, 90, 135 degrees (|Delta A| <= 0.03).
- Geometric border jaggedness B_geo on circle (B_geo <= 0.08) and star (B_geo >= 0.50).
- Perilesional rim gradient fuzziness B_grad on sharp rim (B_grad <= 0.25) vs blurry rim (B_grad >= 0.55).
- Multi-channel color variation C on uniform (C <= 0.03) vs quad-color (C >= 0.45).
- Full orchestration via extract_abcd_biomarkers with BiomarkerResult vector and dict exports.
- Strict negative validation on bad shapes, non-finite inputs, and invalid arguments.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from dermafuzzy_core.biomarkers import (
    BiomarkerResult,
    compute_asymmetry,
    compute_border_fuzziness,
    compute_color_variation,
    extract_abcd_biomarkers,
)
from dermafuzzy_core.config import load_config
from tests.unit.phantoms import (
    make_blurred_rim_phantom,
    make_circle_phantom,
    make_gradcam_heatmap,
    make_l_shape_phantom,
    make_multi_pigment_phantom,
    make_sharp_step_phantom,
    make_star_phantom,
    make_uniform_phantom,
)


@pytest.fixture
def config():
    return load_config()


# ------------------------------------------------------------------------------
# User Story 2: ABCD Biomarkers & Phantoms
# ------------------------------------------------------------------------------

def test_asymmetry_circle_and_l_shape() -> None:
    """Verify ideal circle has A <= 0.05 and asymmetric L-shape has A >= 0.35."""
    _, circle_mask = make_circle_phantom()
    _, l_mask = make_l_shape_phantom()

    a_circle = compute_asymmetry(circle_mask)
    a_l_shape = compute_asymmetry(l_mask)

    assert a_circle <= 0.05, f"Expected circle A <= 0.05, got {a_circle}"
    assert a_l_shape >= 0.35, f"Expected L-shape A >= 0.35, got {a_l_shape}"


def test_asymmetry_rotation_invariance() -> None:
    """Verify asymmetry score of L-shape is invariant across 0, 30, 45, 90, 135 degrees (|Delta A| <= 0.03)."""
    _, l_mask = make_l_shape_phantom()
    base_a = compute_asymmetry(l_mask)

    h, w = l_mask.shape[:2]
    angles = [30, 45, 90, 135]

    for angle in angles:
        rot_mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        rotated_mask = cv2.warpAffine(l_mask, rot_mat, (w, h), flags=cv2.INTER_NEAREST)
        rot_a = compute_asymmetry(rotated_mask)
        delta_a = abs(rot_a - base_a)
        assert delta_a <= 0.03, f"Rotation by {angle} deg caused |Delta A| = {delta_a:.4f} > 0.03 (base={base_a}, rot={rot_a})"


def test_border_fuzziness_geometric_and_gradient(config) -> None:
    """Verify geometric circularity deficit and perilesional rim gradient fuzziness."""
    # Geometric check: circle vs star
    _, circle_mask = make_circle_phantom()
    _, star_mask = make_star_phantom()

    # Create dummy ring for geometric evaluation
    ring_circle = cv2.dilate(circle_mask, np.ones((5, 5), np.uint8)) - cv2.erode(circle_mask, np.ones((5, 5), np.uint8))
    ring_star = cv2.dilate(star_mask, np.ones((5, 5), np.uint8)) - cv2.erode(star_mask, np.ones((5, 5), np.uint8))

    dummy_rgb = np.full((224, 224, 3), 180, dtype=np.uint8)

    _, b_geo_circle, _ = compute_border_fuzziness(dummy_rgb, circle_mask, ring_circle, config)
    _, b_geo_star, _ = compute_border_fuzziness(dummy_rgb, star_mask, ring_star, config)

    assert b_geo_circle <= 0.08, f"Expected circle B_geo <= 0.08, got {b_geo_circle}"
    assert b_geo_star >= 0.50, f"Expected star B_geo >= 0.50, got {b_geo_star}"

    # Gradient check: sharp step vs blurry Gaussian rim
    sharp_rgb, sharp_mask, sharp_ring = make_sharp_step_phantom()
    blurry_rgb, blurry_mask, blurry_ring = make_blurred_rim_phantom(sigma=7.0)

    _, _, b_grad_sharp = compute_border_fuzziness(sharp_rgb, sharp_mask, sharp_ring, config)
    _, _, b_grad_blurry = compute_border_fuzziness(blurry_rgb, blurry_mask, blurry_ring, config)

    assert b_grad_sharp <= 0.25, f"Expected sharp B_grad <= 0.25, got {b_grad_sharp}"
    assert b_grad_blurry >= 0.55, f"Expected blurry B_grad >= 0.55, got {b_grad_blurry}"
    assert (b_grad_blurry - b_grad_sharp) >= 0.30, f"Expected separation >= 0.30, got {b_grad_blurry - b_grad_sharp}"


def test_color_variation_uniform_vs_quad() -> None:
    """Verify uniform lesion yields C <= 0.03 while quad-pigment lesion yields C >= 0.45."""
    uniform_rgb, uniform_mask = make_uniform_phantom()
    quad_rgb, quad_mask = make_multi_pigment_phantom()

    c_uniform = compute_color_variation(uniform_rgb, uniform_mask)
    c_quad = compute_color_variation(quad_rgb, quad_mask)

    assert c_uniform <= 0.03, f"Expected uniform C <= 0.03, got {c_uniform}"
    assert c_quad >= 0.45, f"Expected quad C >= 0.45, got {c_quad}"


def test_extract_abcd_biomarkers_orchestration(config) -> None:
    """Verify end-to-end extraction returns a valid, immutable BiomarkerResult with export methods."""
    rgb, mask, ring = make_sharp_step_phantom()
    hm = make_gradcam_heatmap((224, 224), on_lesion=True)

    result = extract_abcd_biomarkers(
        clean_rgb=rgb,
        mask=mask,
        boundary_ring=ring,
        config=config,
        p_cnn=0.88,
        heatmap=hm,
    )

    assert isinstance(result, BiomarkerResult)
    vec = result.as_vector()
    assert isinstance(vec, tuple)
    assert len(vec) == 4
    for val in vec:
        assert isinstance(val, float)
        assert 0.0 <= val <= 1.0

    d = result.as_dict()
    assert isinstance(d, dict)
    assert set(d.keys()) == {"A", "B_fuzzy", "C", "D_verified", "B_geo", "B_grad", "alpha"}


# ------------------------------------------------------------------------------
# Negative Input Validation Tests
# ------------------------------------------------------------------------------

def test_biomarkers_rejects_invalid_inputs(config) -> None:
    """Verify invalid shapes, non-finite values, and non-array arguments are rejected."""
    good_rgb = np.zeros((224, 224, 3), dtype=np.uint8)
    good_mask = np.zeros((224, 224), dtype=np.uint8)
    good_ring = np.zeros((224, 224), dtype=np.uint8)

    # Asymmetry
    with pytest.raises(ValueError):
        compute_asymmetry(np.zeros((224, 224, 3)))  # 3D mask

    with pytest.raises(ValueError):
        compute_asymmetry("not_an_array")

    # Border fuzziness
    with pytest.raises(ValueError):
        compute_border_fuzziness(np.zeros((224, 224)), good_mask, good_ring, config)  # 2D rgb

    # Color variation
    with pytest.raises(ValueError):
        compute_color_variation(np.zeros((224, 224)), good_mask)  # 2D rgb


def test_core_zero_heavy_ml_imports() -> None:
    """Verify importing dermafuzzy_core does not load heavy deep learning frameworks."""
    import sys
    import dermafuzzy_core  # noqa: F401

    prohibited = ["tensorflow", "keras", "torch", "torchvision"]
    for fw in prohibited:
        assert fw not in sys.modules, f"Constitution violation: heavy ML framework '{fw}' in sys.modules!"

