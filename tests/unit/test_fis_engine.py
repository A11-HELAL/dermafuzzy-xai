"""Unit tests for dermafuzzy_core.fis_engine (Room 4).

Covers:
- US1: Empirical percentile calibration, degenerate separation guard (delta = 1e-4),
       JSON persistence, schema validation, and MappingProxyType deep immutability.
- US2: Piecewise-linear fuzzification, Partition of Unity (sum == 1.0 across 10,000 points),
       Mamdani clinical rule evaluation, Centroid defuzzification benchmarks,
       dominant rule selection, and end-to-end diagnose_image pipeline.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest

from dermafuzzy_core.biomarkers import extract_abcd_biomarkers
from dermafuzzy_core.config import CANONICAL_FEATURE_NAMES, load_config
from dermafuzzy_core.fis_engine import (
    CLINICAL_RULES,
    ClinicalRule,
    FISCalibration,
    FISPrediction,
    FuzzyMembershipSet,
    calibrate_fis_percentiles,
    compute_memberships,
    diagnose_image,
    evaluate_mamdani_fis,
    load_calibration_json,
    save_calibration_json,
)
from dermafuzzy_core.preprocessing import preprocess_image
from tests.unit.phantoms import make_circle_phantom, make_gradcam_heatmap


@pytest.fixture
def standard_calibration() -> FISCalibration:
    """Return an immutable calibration with standard quartiles (0.25, 0.50, 0.75)."""
    cutoffs = {feat: (0.25, 0.50, 0.75) for feat in CANONICAL_FEATURE_NAMES}
    return FISCalibration(
        cutoffs=cutoffs,
        n_samples=1000,
        generated_at="2026-11-15T12:00:00+00:00",
        config_hash="0" * 64,
    )


# ------------------------------------------------------------------------------
# User Story 1: Percentile Calibration & Secure Serialization
# ------------------------------------------------------------------------------

def test_calibrate_fis_percentiles_valid() -> None:
    """Verify calibration computes strictly ordered percentiles on valid empirical data."""
    np.random.seed(42)
    train_matrix = np.random.beta(2, 5, size=(100, 4))
    calib = calibrate_fis_percentiles(train_matrix)

    assert isinstance(calib, FISCalibration)
    assert calib.n_samples == 100
    assert len(calib.config_hash) == 64

    for feat in CANONICAL_FEATURE_NAMES:
        p25, p50, p75 = calib.get_cutoffs(feat)
        assert 0.0 < p25 < p50 < p75 < 1.0
        assert (p50 - p25) >= 1e-4 - 1e-9
        assert (p75 - p50) >= 1e-4 - 1e-9


def test_degenerate_separation_guard() -> None:
    """Verify degenerate constant columns are separated by delta = 1e-4 without crashing."""
    # Col 0: all identical 0.5; Col 1: all identical 0.0; Col 2: all identical 1.0; Col 3: normal
    train_matrix = np.zeros((20, 4), dtype=np.float64)
    train_matrix[:, 0] = 0.5
    train_matrix[:, 1] = 0.0
    train_matrix[:, 2] = 1.0
    train_matrix[:, 3] = np.linspace(0.1, 0.9, 20)

    calib = calibrate_fis_percentiles(train_matrix)

    for feat in CANONICAL_FEATURE_NAMES:
        p25, p50, p75 = calib.get_cutoffs(feat)
        assert 0.0 < p25 < p50 < p75 < 1.0
        assert (p50 - p25) >= 1e-4 - 1e-9
        assert (p75 - p50) >= 1e-4 - 1e-9


def test_calibration_immutability_mapping_proxy(standard_calibration: FISCalibration) -> None:
    """Verify calibration cutoffs dictionary cannot be mutated at runtime."""
    assert isinstance(standard_calibration.cutoffs, MappingProxyType)

    with pytest.raises(TypeError):
        standard_calibration.cutoffs["A"] = (0.2, 0.4, 0.6)  # type: ignore[index]


def test_calibration_json_roundtrip(tmp_path: Path, standard_calibration: FISCalibration) -> None:
    """Verify calibration serializes to JSON and reloads with bit-identical values."""
    save_path = tmp_path / "artifacts" / "test_calibration.json"
    written = save_calibration_json(standard_calibration, save_path)
    assert written == save_path
    assert save_path.is_file()

    reloaded = load_calibration_json(save_path)
    assert isinstance(reloaded, FISCalibration)
    assert reloaded.n_samples == standard_calibration.n_samples
    assert reloaded.config_hash == standard_calibration.config_hash

    for feat in CANONICAL_FEATURE_NAMES:
        assert reloaded.get_cutoffs(feat) == standard_calibration.get_cutoffs(feat)


def test_calibration_rejects_corrupted_json(tmp_path: Path) -> None:
    """Verify schema-violating or malformed JSON files are rejected."""
    bad_path = tmp_path / "corrupted.json"

    # Malformed JSON syntax
    bad_path.write_text("{bad json", encoding="utf-8")
    with pytest.raises(ValueError, match="Failed to parse calibration JSON"):
        load_calibration_json(bad_path)

    # Missing keys
    bad_path.write_text('{"version": "1.0.0"}', encoding="utf-8")
    with pytest.raises(ValueError, match="missing required keys"):
        load_calibration_json(bad_path)

    # Inverted cutoffs
    invalid_data = {
        "version": "1.0.0",
        "generated_at": "2026-11-15T00:00:00Z",
        "n_samples": 100,
        "feature_names": list(CANONICAL_FEATURE_NAMES),
        "cutoffs": {feat: [0.75, 0.50, 0.25] for feat in CANONICAL_FEATURE_NAMES},
        "config_hash": "0" * 64,
    }
    with pytest.raises(ValueError, match="Strict monotonicity violated"):
        FISCalibration.from_dict(invalid_data)


@pytest.mark.parametrize(
    "bad_matrix",
    [
        "not_an_array",
        None,
        True,
        np.ones((3, 4)),          # N < 4
        np.ones((10, 3)),         # 3 cols instead of 4
        np.ones((10, 4, 1)),      # 3D
        np.full((10, 4), np.nan), # NaN
        np.full((10, 4), np.inf), # Inf
        np.full((10, 4), -0.1),   # < 0.0
        np.full((10, 4), 1.1),    # > 1.0
    ],
)
def test_calibration_rejects_invalid_inputs(bad_matrix: Any) -> None:
    """Verify invalid training matrices are rejected immediately."""
    with pytest.raises(ValueError):
        calibrate_fis_percentiles(bad_matrix)


# ------------------------------------------------------------------------------
# User Story 2: Mamdani Fuzzification & Clinical Rule Engine
# ------------------------------------------------------------------------------

def test_compute_memberships_partition_of_unity() -> None:
    """Verify analytical Partition of Unity (sum == 1.0) holds across 10,000 points."""
    cutoffs = (0.25, 0.50, 0.75)
    test_points = np.linspace(0.0, 1.0, 10000)

    for x in test_points:
        m = compute_memberships(x, cutoffs)
        assert 0.0 <= m.low <= 1.0
        assert 0.0 <= m.medium <= 1.0
        assert 0.0 <= m.high <= 1.0
        tot = m.low + m.medium + m.high
        assert math.isclose(tot, 1.0, abs_tol=1e-6)

    # Edge saturation
    m_zero = compute_memberships(0.0, cutoffs)
    assert m_zero.low == 1.0 and m_zero.medium == 0.0 and m_zero.high == 0.0

    m_one = compute_memberships(1.0, cutoffs)
    assert m_one.low == 0.0 and m_one.medium == 0.0 and m_one.high == 1.0


def test_evaluate_mamdani_pure_low(standard_calibration: FISCalibration) -> None:
    """Verify pure Low input scores produce Low Risk category and activate Rule 5."""
    pred = evaluate_mamdani_fis((0.05, 0.05, 0.05, 0.05), standard_calibration)
    assert pred.risk_score < 25.0
    assert pred.risk_category == "Low"
    assert pred.dominant_rule_id == "R5"
    assert pred.rule_firings["R5"] == 1.0


def test_evaluate_mamdani_pure_medium(standard_calibration: FISCalibration) -> None:
    """Verify pure Medium input scores (at P50) produce Medium Risk category and activate Rule 4."""
    pred = evaluate_mamdani_fis((0.50, 0.50, 0.50, 0.50), standard_calibration)
    assert 35.0 <= pred.risk_score < 65.0
    assert pred.risk_category == "Medium"
    assert pred.dominant_rule_id == "R4"
    assert pred.rule_firings["R4"] == 1.0


def test_evaluate_mamdani_worked_study_example(standard_calibration: FISCalibration) -> None:
    """Verify the study guide worked reference case (0.68, 0.74, 0.81, 0.80) yields [80%, 88%] High Risk."""
    pred = evaluate_mamdani_fis((0.68, 0.74, 0.81, 0.80), standard_calibration)
    assert 80.0 <= pred.risk_score <= 88.0
    assert pred.risk_category == "High"
    assert pred.dominant_rule_id in {"R1", "R2"}
    assert "Malignant" in pred.dominant_rule_text


def test_evaluate_mamdani_artifact_protection(standard_calibration: FISCalibration) -> None:
    """Verify a benign nevus corrupted by an off-lesion ruler artifact classifies as Low Risk via Rule 5."""
    # A=0.08, B=0.10, C=0.05, p_cnn=0.92, alpha=0.12 -> D_verified=0.1104
    pred = evaluate_mamdani_fis((0.08, 0.10, 0.05, 0.1104), standard_calibration)
    assert pred.risk_score < 35.0
    assert pred.risk_category == "Low"
    assert pred.dominant_rule_id == "R5"
    assert pred.rule_firings["R5"] == 1.0


def test_prediction_immutability_mapping_proxy(standard_calibration: FISCalibration) -> None:
    """Verify prediction firings and memberships cannot be mutated at runtime."""
    pred = evaluate_mamdani_fis((0.50, 0.50, 0.50, 0.50), standard_calibration)

    assert isinstance(pred.rule_firings, MappingProxyType)
    assert isinstance(pred.memberships, MappingProxyType)
    assert isinstance(pred.memberships["A"], MappingProxyType)

    with pytest.raises(TypeError):
        pred.rule_firings["R1"] = 0.99  # type: ignore[index]

    with pytest.raises(TypeError):
        pred.memberships["A"]["low"] = 0.5  # type: ignore[index]


@pytest.mark.parametrize(
    "bad_biomarkers",
    [
        (0.5, 0.5, 0.5),          # 3 elements
        (0.5, 0.5, 0.5, 0.5, 0.5), # 5 elements
        (True, 0.5, 0.5, 0.5),    # bool
        (np.nan, 0.5, 0.5, 0.5),  # NaN
        (np.inf, 0.5, 0.5, 0.5),  # Inf
        (-0.1, 0.5, 0.5, 0.5),    # < 0.0
        (1.1, 0.5, 0.5, 0.5),     # > 1.0
        "not_a_tuple",
    ],
)
def test_evaluate_mamdani_rejects_invalid_inputs(
    bad_biomarkers: Any, standard_calibration: FISCalibration
) -> None:
    """Verify invalid biomarker inputs are rejected."""
    with pytest.raises(ValueError):
        evaluate_mamdani_fis(bad_biomarkers, standard_calibration)


def test_diagnose_image_orchestration(standard_calibration: FISCalibration) -> None:
    """Verify end-to-end pipeline execution from raw image to FISPrediction."""
    rgb, mask = make_circle_phantom(size=(224, 224), radius=50)
    heatmap = make_gradcam_heatmap(size=(224, 224), on_lesion=True)
    p_cnn = 0.85

    prep_res, bio_res, pred = diagnose_image(
        img_rgb=rgb,
        p_cnn=p_cnn,
        heatmap=heatmap,
        calibration=standard_calibration,
    )

    assert prep_res.clean_rgb.shape == (224, 224, 3)
    assert bio_res.A <= 0.05
    assert bio_res.B_fuzzy <= 0.20
    assert isinstance(pred, FISPrediction)
    assert 0.0 <= pred.risk_score <= 100.0
