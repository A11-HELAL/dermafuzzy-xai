"""Unit and Synthetic Phantom tests for Room 3: ABCD Biomarker Extraction.

Tests:
- BiomarkerResult: frozen immutability, slots, vector/dict serialization.
- Asymmetry A: ideal circle (A <= 0.05), symmetric square (A <= 0.05), asymmetric L-shape (A >= 0.35).
- Asymmetry A: rotation invariance across 0, 30, 45, 90, 135 degrees (|Delta A| <= 0.03).
- Asymmetry A: degenerate blank and tiny (< 5 pixels) masks safely returning 0.0.
- Geometric border jaggedness B_geo: circle (B_geo <= 0.08) vs star (B_geo >= 0.50).
- Perilesional rim gradient fuzziness B_grad: sharp rim (B_grad <= 0.25) vs blurry rim (B_grad >= 0.55).
- Border fuzziness: degenerate blank mask and blank ring handling, custom weight fusion.
- Multi-channel color variation C: uniform (C <= 0.03) vs quad-color (C >= 0.45), degenerate blank mask.
- Full orchestration: extract_abcd_biomarkers with BiomarkerResult, p_cnn/heatmap variations.
- Strict negative validation on bad shapes, dtypes, non-finite inputs, and invalid arguments.
"""

from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError
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
from dermafuzzy_core.config import DermaConfig, load_config
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
def config() -> DermaConfig:
    return load_config()


# ------------------------------------------------------------------------------
# Entity Tests: BiomarkerResult Dataclass
# ------------------------------------------------------------------------------

class TestBiomarkerResultDataclass:
    """Test BiomarkerResult frozen immutability, slots, and export contracts."""

    def test_immutability_and_slots(self):
        result = BiomarkerResult(
            A=0.10,
            B_fuzzy=0.20,
            C=0.30,
            D_verified=0.40,
            B_geo=0.15,
            B_grad=0.25,
            alpha=0.85,
        )

        # Immutability check
        with pytest.raises(FrozenInstanceError):
            result.A = 0.50  # type: ignore

        # Slotted check (no __dict__)
        assert hasattr(result, "__slots__")
        assert not hasattr(result, "__dict__")

        # Vector export
        vec = result.as_vector()
        assert isinstance(vec, tuple)
        assert len(vec) == 4
        assert vec == (0.10, 0.20, 0.30, 0.40)

        # Dictionary export
        d = result.as_dict()
        assert isinstance(d, dict)
        assert d == {
            "A": 0.10,
            "B_fuzzy": 0.20,
            "C": 0.30,
            "D_verified": 0.40,
            "B_geo": 0.15,
            "B_grad": 0.25,
            "alpha": 0.85,
        }


# ------------------------------------------------------------------------------
# Asymmetry Tests
# ------------------------------------------------------------------------------

class TestAsymmetryCalculation:
    """Test compute_asymmetry with geometric phantoms and boundary edge cases."""

    def test_asymmetry_circle_and_l_shape(self):
        """Verify ideal circle has A <= 0.05 and asymmetric L-shape has A >= 0.35."""
        _, circle_mask = make_circle_phantom()
        _, l_mask = make_l_shape_phantom()

        a_circle = compute_asymmetry(circle_mask)
        a_l_shape = compute_asymmetry(l_mask)

        assert a_circle <= 0.05, f"Expected circle A <= 0.05, got {a_circle}"
        assert a_l_shape >= 0.35, f"Expected L-shape A >= 0.35, got {a_l_shape}"

    def test_asymmetry_symmetric_square(self):
        """Verify centered symmetric square has low asymmetry A <= 0.05."""
        mask = np.zeros((224, 224), dtype=np.uint8)
        mask[62:162, 62:162] = 255
        a_square = compute_asymmetry(mask)
        assert a_square <= 0.05, f"Expected symmetric square A <= 0.05, got {a_square}"

    def test_asymmetry_rotation_invariance(self):
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

    def test_asymmetry_blank_and_tiny_masks(self):
        """Verify degenerate blank and tiny (< 5 pixels) masks return 0.0 without crash."""
        blank_mask = np.zeros((224, 224), dtype=np.uint8)
        assert compute_asymmetry(blank_mask) == 0.0

        tiny_mask = np.zeros((224, 224), dtype=np.uint8)
        tiny_mask[112, 112] = 255
        tiny_mask[112, 113] = 255
        assert compute_asymmetry(tiny_mask) == 0.0


# ------------------------------------------------------------------------------
# Border Fuzziness Tests
# ------------------------------------------------------------------------------

class TestBorderFuzzinessCalculation:
    """Test compute_border_fuzziness with phantoms and boundary edge cases."""

    def test_border_fuzziness_geometric_and_gradient(self, config: DermaConfig):
        """Verify geometric circularity deficit and perilesional rim gradient fuzziness."""
        # Geometric check: circle vs star
        _, circle_mask = make_circle_phantom()
        _, star_mask = make_star_phantom()

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

    def test_border_fuzziness_degenerate_blank_mask_and_ring(self, config: DermaConfig):
        """Verify blank mask and empty boundary ring handle gracefully without division by zero."""
        dummy_rgb = np.full((224, 224, 3), 180, dtype=np.uint8)
        blank_mask = np.zeros((224, 224), dtype=np.uint8)
        blank_ring = np.zeros((224, 224), dtype=np.uint8)

        b_fuzz, b_geo, b_grad = compute_border_fuzziness(dummy_rgb, blank_mask, blank_ring, config)
        assert b_fuzz == 0.0
        assert b_geo == 0.0
        assert b_grad == 0.0

        # Valid mask but empty ring: b_grad falls back to 0.0
        _, mask = make_circle_phantom()
        b_fuzz, b_geo, b_grad = compute_border_fuzziness(dummy_rgb, mask, blank_ring, config)
        assert b_geo <= 0.08
        assert b_grad == 0.0
        assert 0.0 <= b_fuzz <= 1.0

    def test_border_fuzziness_zero_gradient_yields_diffuse(self, config: DermaConfig):
        """Verify that a uniform flat image across the ring evaluates to B_grad = 1.0 (diffuse)."""
        black_rgb = np.zeros((224, 224, 3), dtype=np.uint8)
        _, mask = make_circle_phantom()
        ring = cv2.dilate(mask, np.ones((5, 5), np.uint8)) - cv2.erode(mask, np.ones((5, 5), np.uint8))

        _, _, b_grad = compute_border_fuzziness(black_rgb, mask, ring, config)
        assert b_grad == 1.0


# ------------------------------------------------------------------------------
# Color Variation Tests
# ------------------------------------------------------------------------------

class TestColorVariationCalculation:
    """Test compute_color_variation with phantoms and edge cases."""

    def test_color_variation_uniform_vs_quad(self):
        """Verify uniform lesion yields C <= 0.03 while quad-pigment lesion yields C >= 0.45."""
        uniform_rgb, uniform_mask = make_uniform_phantom()
        quad_rgb, quad_mask = make_multi_pigment_phantom()

        c_uniform = compute_color_variation(uniform_rgb, uniform_mask)
        c_quad = compute_color_variation(quad_rgb, quad_mask)

        assert c_uniform <= 0.03, f"Expected uniform C <= 0.03, got {c_uniform}"
        assert c_quad >= 0.45, f"Expected quad C >= 0.45, got {c_quad}"

    def test_color_variation_blank_mask(self):
        """Verify empty mask yields C = 0.0."""
        uniform_rgb, _ = make_uniform_phantom()
        blank_mask = np.zeros((224, 224), dtype=np.uint8)
        assert compute_color_variation(uniform_rgb, blank_mask) == 0.0


# ------------------------------------------------------------------------------
# Orchestration & Negative Validation Tests
# ------------------------------------------------------------------------------

class TestBiomarkersOrchestrationAndValidation:
    """Test extract_abcd_biomarkers and comprehensive negative input validation."""

    def test_extract_abcd_biomarkers_orchestration(self, config: DermaConfig):
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
        assert len(vec) == 4
        for val in vec:
            assert isinstance(val, float)
            assert 0.0 <= val <= 1.0

        d = result.as_dict()
        assert set(d.keys()) == {"A", "B_fuzzy", "C", "D_verified", "B_geo", "B_grad", "alpha"}

    def test_extract_abcd_biomarkers_none_heatmap_and_zero_p_cnn(self, config: DermaConfig):
        """Verify None heatmap and zero p_cnn result in alpha=0.0 and D_verified=0.0."""
        rgb, mask, ring = make_sharp_step_phantom()

        res = extract_abcd_biomarkers(
            clean_rgb=rgb,
            mask=mask,
            boundary_ring=ring,
            config=config,
            p_cnn=0.0,
            heatmap=None,
        )
        assert res.alpha == 0.0
        assert res.D_verified == 0.0

    @pytest.mark.parametrize(
        "bad_mask",
        [
            np.zeros((224, 224, 3)),              # 3D
            np.zeros((224,)),                     # 1D
            np.full((224, 224), np.nan),          # NaN
            np.full((224, 224), np.inf),          # Inf
            "not_an_array",                       # String
            123,                                  # Integer
        ],
    )
    def test_compute_asymmetry_rejects_invalid_mask(self, bad_mask: Any):
        with pytest.raises(ValueError):
            compute_asymmetry(bad_mask)

    @pytest.mark.parametrize(
        "bad_rgb",
        [
            np.zeros((224, 224)),                 # 2D
            np.zeros((224, 224, 4), dtype=np.uint8), # 4 channels
            np.zeros((224, 224, 3), dtype=np.float32), # float instead of uint8
            "not_an_array",
        ],
    )
    def test_compute_color_variation_rejects_invalid_rgb(self, bad_rgb: Any):
        mask = np.zeros((224, 224), dtype=np.uint8)
        with pytest.raises(ValueError):
            compute_color_variation(bad_rgb, mask)

    @pytest.mark.parametrize(
        "bad_p_cnn",
        [
            True,
            False,
            float("nan"),
            float("inf"),
            -0.1,
            1.5,
        ],
    )
    def test_extract_abcd_biomarkers_rejects_invalid_p_cnn(self, config: DermaConfig, bad_p_cnn: Any):
        rgb, mask, ring = make_sharp_step_phantom()
        with pytest.raises(ValueError):
            extract_abcd_biomarkers(
                clean_rgb=rgb,
                mask=mask,
                boundary_ring=ring,
                config=config,
                p_cnn=bad_p_cnn,
            )

    def test_core_zero_heavy_ml_imports(self):
        """Verify importing dermafuzzy_core does not load heavy deep learning frameworks."""
        import sys
        import dermafuzzy_core  # noqa: F401

        prohibited = ["tensorflow", "keras", "torch", "torchvision"]
        for fw in prohibited:
            assert fw not in sys.modules, f"Constitution violation: heavy ML framework '{fw}' in sys.modules!"
