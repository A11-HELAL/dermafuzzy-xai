"""Room 6: Metamorphic Testing Suite for DermaFuzzy-XAI.

Implements 5 clinical Metamorphic Relations (MR1–MR5) evaluating the end-to-end pipeline:
Room 2 (Preprocessing) -> Room 3 (Biomarkers & Grad-CAM) -> Room 4 (Mamdani FIS).

Metamorphic Relations:
- MR1: Rotation Invariance (scores invariant within |Delta| <= 2.0% across 0, 90, 180, 270 deg).
- MR2: Peripheral Ruler Artifact Immunity (off-lesion ruler artifact preserves Low Risk).
- MR3: Monotonic Biomarker Risk Response (increasing any biomarker never decreases risk_score).
- MR4: Partition Completeness Grid Sweep (625 combinations across [0, 1]^4 execute with zero NaN).
- MR5: Deterministic Calibration Serialization (JSON round-trip bit-identity).
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from dermafuzzy_core.config import CANONICAL_FEATURE_NAMES, load_config
from dermafuzzy_core.fis_engine import (
    FISCalibration,
    calibrate_fis_percentiles,
    diagnose_image,
    evaluate_mamdani_fis,
    load_calibration_json,
    save_calibration_json,
)
from tests.unit.phantoms import (
    make_circle_phantom,
    make_gradcam_heatmap,
    make_l_shape_phantom,
    make_multi_pigment_phantom,
)


@pytest.fixture
def default_calibration() -> FISCalibration:
    """Return standard clinical quartiles calibration."""
    cutoffs = {feat: (0.25, 0.50, 0.75) for feat in CANONICAL_FEATURE_NAMES}
    return FISCalibration(
        cutoffs=cutoffs,
        n_samples=1000,
        generated_at="2026-11-15T12:00:00+00:00",
        config_hash="0" * 64,
    )


# ------------------------------------------------------------------------------
# MR1: Rotation Invariance
# ------------------------------------------------------------------------------

def test_mr1_rotation_invariance(default_calibration: FISCalibration) -> None:
    """MR1: Rotating input image and heatmap by 90, 180, 270 deg preserves risk within 2.0%."""
    # Test on a circle phantom with high deep probability
    img_rgb, mask = make_circle_phantom(size=(224, 224), radius=50)
    heatmap = make_gradcam_heatmap(size=(224, 224), on_lesion=True)
    p_cnn = 0.85

    # Base orientation (0 deg)
    _, _, pred_0 = diagnose_image(img_rgb, p_cnn, heatmap, default_calibration)

    rotation_codes = [
        (cv2.ROTATE_90_CLOCKWISE, "90 deg"),
        (cv2.ROTATE_180, "180 deg"),
        (cv2.ROTATE_90_COUNTERCLOCKWISE, "270 deg"),
    ]

    for rot_code, label in rotation_codes:
        rot_img = cv2.rotate(img_rgb, rot_code)
        rot_heatmap = cv2.rotate(heatmap, rot_code)

        _, _, pred_rot = diagnose_image(rot_img, p_cnn, rot_heatmap, default_calibration)

        delta = abs(pred_rot.risk_score - pred_0.risk_score)
        assert delta <= 2.0, (
            f"MR1 violation at {label}: risk_score shifted by {delta:.3f}% "
            f"(base={pred_0.risk_score:.2f}%, rot={pred_rot.risk_score:.2f}%)"
        )
        assert pred_rot.risk_category == pred_0.risk_category, (
            f"MR1 category flip at {label}: base={pred_0.risk_category}, rot={pred_rot.risk_category}"
        )


# ------------------------------------------------------------------------------
# MR2: Peripheral Ruler Artifact Immunity
# ------------------------------------------------------------------------------

def test_mr2_peripheral_ruler_artifact_immunity(default_calibration: FISCalibration) -> None:
    """MR2: A benign nevus with high CNN probability corrupted by an off-lesion ruler classifies as Low Risk."""
    # Benign circular lesion
    img_rgb, mask = make_circle_phantom(size=(224, 224), radius=45)

    # CNN predicts high probability (e.g. 0.92) because of peripheral ruler,
    # but Grad-CAM attention focuses entirely on the off-lesion ruler artifact
    p_cnn_spurious = 0.92
    ruler_heatmap = make_gradcam_heatmap(size=(224, 224), on_lesion=False)

    _, bio_res, pred = diagnose_image(
        img_rgb=img_rgb,
        p_cnn=p_cnn_spurious,
        heatmap=ruler_heatmap,
        calibration=default_calibration,
    )

    # Attention overlap should be low because heatmap is on the ruler outside lesion
    assert bio_res.alpha <= 0.15, f"Expected off-lesion alpha <= 0.15, got {bio_res.alpha}"
    assert bio_res.D_verified <= 0.15, f"Expected verified confidence <= 0.15, got {bio_res.D_verified}"

    # FIS MUST suppress the spurious deep CNN confidence and classify as Low Risk via Rule 5
    assert pred.risk_score < 35.0, (
        f"MR2 violation: Spurious CNN probability breached artifact guard! "
        f"Score={pred.risk_score:.2f}% (must be < 35.0%)"
    )
    assert pred.risk_category == "Low"
    assert pred.dominant_rule_id == "R5"


# ------------------------------------------------------------------------------
# MR3: Monotonic Biomarker Risk Response
# ------------------------------------------------------------------------------

def test_mr3_monotonic_biomarker_risk_response(default_calibration: FISCalibration) -> None:
    """MR3: Increasing any biomarker monotonically while holding others constant NEVER decreases risk_score."""
    # Test across multiple baseline configurations
    base_vectors = [
        [0.10, 0.10, 0.10, 0.10], # Low baseline
        [0.40, 0.40, 0.40, 0.40], # Med-Low baseline
        [0.50, 0.50, 0.50, 0.50], # Pure Med baseline
        [0.70, 0.70, 0.70, 0.70], # High baseline
        [0.20, 0.60, 0.30, 0.80], # Mixed baseline 1
        [0.80, 0.20, 0.70, 0.30], # Mixed baseline 2
        [0.10, 0.80, 0.80, 0.10], # Mixed baseline 3
    ]

    sweep_values = np.linspace(0.05, 0.95, 19)

    for base_idx, base_vec in enumerate(base_vectors):
        for coord in range(4):
            feat_name = CANONICAL_FEATURE_NAMES[coord]
            scores: list[float] = []

            for val in sweep_values:
                vec = base_vec.copy()
                vec[coord] = float(val)
                pred = evaluate_mamdani_fis(vec, default_calibration)
                scores.append(pred.risk_score)

            diffs = np.diff(scores)
            min_diff = float(np.min(diffs))

            assert min_diff >= -1e-9, (
                f"MR3 Monotonicity violation in base vector {base_idx} along feature '{feat_name}': "
                f"min diff = {min_diff:.6e}. Scores: {scores}"
            )


# ------------------------------------------------------------------------------
# MR4: Partition Completeness Grid Sweep
# ------------------------------------------------------------------------------

def test_mr4_partition_completeness_grid_sweep(default_calibration: FISCalibration) -> None:
    """MR4: Full 5x5x5x5 hypercube sweep (625 points) executes with finite values and zero NaN."""
    grid_coords = [0.0, 0.25, 0.50, 0.75, 1.00]
    combinations = list(itertools.product(grid_coords, repeat=4))
    assert len(combinations) == 625

    for vec in combinations:
        pred = evaluate_mamdani_fis(vec, default_calibration)

        assert math.isfinite(pred.risk_score), f"MR4 violation: non-finite risk_score for {vec}"
        assert 0.0 <= pred.risk_score <= 100.0, f"MR4 violation: out-of-bounds score {pred.risk_score} for {vec}"
        assert pred.risk_category in {"Low", "Medium", "High"}, f"MR4 violation: invalid category {pred.risk_category}"
        assert pred.dominant_rule_id in {"R1", "R2", "R3", "R4", "R5"}, f"MR4 violation: invalid rule {pred.dominant_rule_id}"

        # Rule firings must all be finite and within [0, 1]
        for r_id, firing in pred.rule_firings.items():
            assert math.isfinite(firing)
            assert 0.0 <= firing <= 1.0


# ------------------------------------------------------------------------------
# MR5: Deterministic Calibration Serialization
# ------------------------------------------------------------------------------

def test_mr5_deterministic_calibration_serialization(tmp_path: Path) -> None:
    """MR5: Calibrating on empirical training data, serializing to JSON, and reloading preserves bit-identity."""
    np.random.seed(12345)
    train_data = np.random.uniform(0.05, 0.95, size=(120, 4))
    calib_orig = calibrate_fis_percentiles(train_data)

    calib_file = tmp_path / "artifacts" / "test_fis_calibration.json"
    save_calibration_json(calib_orig, calib_file)
    calib_reloaded = load_calibration_json(calib_file)

    # Test batch of 20 random evaluation vectors
    test_batch = np.random.uniform(0.0, 1.0, size=(20, 4))
    for v in test_batch:
        p_orig = evaluate_mamdani_fis(v, calib_orig)
        p_reloaded = evaluate_mamdani_fis(v, calib_reloaded)

        assert p_orig.risk_score == p_reloaded.risk_score, (
            f"MR5 violation: risk_score mismatch ({p_orig.risk_score} != {p_reloaded.risk_score})"
        )
        assert p_orig.risk_category == p_reloaded.risk_category
        assert p_orig.dominant_rule_id == p_reloaded.dominant_rule_id
        assert p_orig.dominant_rule_text == p_reloaded.dominant_rule_text
        assert dict(p_orig.rule_firings) == dict(p_reloaded.rule_firings)
