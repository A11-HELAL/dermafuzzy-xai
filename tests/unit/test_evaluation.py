"""Unit tests for Room 5 Clinical Evaluation & 3-Tier Ablation Engine.

Covers:
- Immutable ClinicalMetrics and AblationReport dataclasses
- Input validation (rejection of bool, NaN, Inf, dimension mismatch, invalid thresholds)
- Confusion matrix counts and clinical rate formulations with epsilon stabilizers
- Exact pure NumPy tied-score trapezoidal AUC-ROC matching Mann-Whitney U fractional ranks
- Crisp ABCD Baseline 2 thresholding at P50
- 3-tier Ablation Study execution and demonstration of DermaFuzzy-XAI artifact resistance
- Markdown and CSV report formatters
"""

import math
from datetime import datetime
import numpy as np
import pytest
from scipy.stats import mannwhitneyu

from dermafuzzy_core.biomarkers import BiomarkerResult
from dermafuzzy_core.fis_engine import FISCalibration
from dermafuzzy_core.evaluation import (
    ClinicalMetrics,
    AblationReport,
    compute_clinical_metrics,
    evaluate_crisp_abcd_baseline,
    run_ablation_study,
)


def _make_dummy_calibration() -> FISCalibration:
    """Helper to create standard quartile calibration."""
    cutoffs = {
        "A": (0.25, 0.50, 0.75),
        "B_fuzzy": (0.25, 0.50, 0.75),
        "C": (0.25, 0.50, 0.75),
        "D_verified": (0.25, 0.50, 0.75),
    }
    return FISCalibration(
        cutoffs=cutoffs,
        n_samples=100,
        generated_at="2026-11-15T00:00:00+00:00",
        config_hash="0" * 64,
    )


class TestClinicalMetricsDataclass:
    """Test ClinicalMetrics creation, immutability, and validation."""

    def test_valid_creation_and_immutability(self):
        metrics = ClinicalMetrics(
            tp=45,
            tn=48,
            fp=2,
            fn=5,
            sensitivity=0.90,
            specificity=0.96,
            precision=0.9574,
            accuracy=0.93,
            f1_score=0.9278,
            auc_roc=0.985,
            threshold=0.50,
        )
        assert metrics.tp == 45
        assert metrics.tn == 48
        assert metrics.sensitivity == 0.90

        # Verify frozen immutability
        with pytest.raises(Exception):
            metrics.tp = 50  # type: ignore

    def test_reject_bool_and_non_int_counts(self):
        with pytest.raises(ValueError, match="must be non-negative integer"):
            ClinicalMetrics(
                tp=True,  # type: ignore
                tn=10,
                fp=0,
                fn=0,
                sensitivity=1.0,
                specificity=1.0,
                precision=1.0,
                accuracy=1.0,
                f1_score=1.0,
                auc_roc=1.0,
            )

        with pytest.raises(ValueError, match="must be non-negative integer"):
            ClinicalMetrics(
                tp=-1,
                tn=10,
                fp=0,
                fn=0,
                sensitivity=1.0,
                specificity=1.0,
                precision=1.0,
                accuracy=1.0,
                f1_score=1.0,
                auc_roc=1.0,
            )

    def test_reject_out_of_bounds_rates(self):
        with pytest.raises(ValueError, match="must be in"):
            ClinicalMetrics(
                tp=10,
                tn=10,
                fp=0,
                fn=0,
                sensitivity=1.5,  # Invalid rate > 1.0
                specificity=1.0,
                precision=1.0,
                accuracy=1.0,
                f1_score=1.0,
                auc_roc=1.0,
            )

    def test_to_dict_keys(self):
        metrics = ClinicalMetrics(
            tp=10,
            tn=10,
            fp=0,
            fn=0,
            sensitivity=1.0,
            specificity=1.0,
            precision=1.0,
            accuracy=1.0,
            f1_score=1.0,
            auc_roc=1.0,
        )
        d = metrics.to_dict()
        assert d["tp"] == 10
        assert d["accuracy"] == 1.0


class TestComputeClinicalMetrics:
    """Test compute_clinical_metrics function."""

    def test_perfect_classification(self):
        y_true = np.array([1, 1, 0, 0])
        y_prob = np.array([0.9, 0.8, 0.1, 0.2])
        m = compute_clinical_metrics(y_true, y_prob, threshold=0.50)

        assert m.tp == 2
        assert m.tn == 2
        assert m.fp == 0
        assert m.fn == 0
        assert math.isclose(m.sensitivity, 1.0)
        assert math.isclose(m.specificity, 1.0)
        assert math.isclose(m.precision, 1.0)
        assert math.isclose(m.accuracy, 1.0)
        assert math.isclose(m.f1_score, 1.0)
        assert math.isclose(m.auc_roc, 1.0)

    def test_percentage_scaled_probabilities(self):
        # Scale [0.0, 100.0] should be auto-normalized to [0.0, 1.0]
        y_true = [1, 0]
        y_prob = [90.0, 10.0]
        m = compute_clinical_metrics(y_true, y_prob, threshold=0.50)
        assert m.tp == 1
        assert m.tn == 1

    def test_zero_division_single_class(self):
        # All positive labels: specificity denominator (TN + FP) = 0
        y_true = np.array([1, 1, 1])
        y_prob = np.array([0.8, 0.9, 0.7])
        m = compute_clinical_metrics(y_true, y_prob)
        assert m.sensitivity == 1.0
        assert m.specificity == 0.0  # Safe zero fallback
        assert m.auc_roc == 0.50     # Undefined ROC yields chance 0.50

        # All negative labels: sensitivity denominator (TP + FN) = 0
        y_true_neg = np.array([0, 0, 0])
        y_prob_neg = np.array([0.1, 0.2, 0.3])
        m_neg = compute_clinical_metrics(y_true_neg, y_prob_neg)
        assert m_neg.sensitivity == 0.0
        assert m_neg.specificity == 1.0
        assert m_neg.auc_roc == 0.50

    def test_constant_prediction_auc(self):
        # All identical predictions -> AUC must be 0.50
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([0.5, 0.5, 0.5, 0.5])
        m = compute_clinical_metrics(y_true, y_prob)
        assert m.auc_roc == 0.50

    def test_exact_tied_score_auc_roc_vs_mann_whitney(self):
        """AUC-ROC must match exact Mann-Whitney U fractional ranks on heavy ties."""
        y_true = np.array([0, 0, 1, 0, 1, 1, 0, 1, 0, 1])
        y_prob = np.array([0.1, 0.4, 0.4, 0.4, 0.6, 0.7, 0.7, 0.7, 0.9, 0.9])

        m = compute_clinical_metrics(y_true, y_prob)

        # Scipy reference
        pos = y_prob[y_true == 1]
        neg = y_prob[y_true == 0]
        u_stat, _ = mannwhitneyu(pos, neg, method="exact")
        expected_auc = u_stat / (len(pos) * len(neg))

        assert math.isclose(m.auc_roc, expected_auc, abs_tol=1e-6)

    def test_input_validation_rejections(self):
        # Reject bool input
        with pytest.raises(ValueError, match="Boolean input"):
            compute_clinical_metrics(True, [0.5])  # type: ignore

        # Reject shape mismatch
        with pytest.raises(ValueError, match="Length mismatch"):
            compute_clinical_metrics([1, 0], [0.5])

        # Reject empty arrays
        with pytest.raises(ValueError, match="cannot be empty"):
            compute_clinical_metrics([], [])

        # Reject NaN or Inf
        with pytest.raises(ValueError, match="NaN or Inf"):
            compute_clinical_metrics([1, 0], [np.nan, 0.5])

        with pytest.raises(ValueError, match="NaN or Inf"):
            compute_clinical_metrics([1, 0], [np.inf, 0.5])

        # Reject labels outside {0, 1}
        with pytest.raises(ValueError, match="binary"):
            compute_clinical_metrics([1, 2], [0.5, 0.5])

        # Reject invalid threshold
        with pytest.raises(ValueError, match="threshold"):
            compute_clinical_metrics([1, 0], [0.5, 0.5], threshold=1.5)


class TestCrispABCDBaseline:
    """Test evaluate_crisp_abcd_baseline."""

    def test_step_indicator_evaluation(self):
        calib = _make_dummy_calibration()
        # All below P50 (0.50) -> 0.0
        # All above P50 (0.50) -> 1.0
        # 2 above, 2 below -> 0.50
        batch = np.array([
            [0.1, 0.2, 0.3, 0.4],   # 0 triggered -> 0.0
            [0.6, 0.7, 0.8, 0.9],   # 4 triggered -> 1.0
            [0.6, 0.7, 0.2, 0.3],   # 2 triggered -> 0.5
        ])

        scores = evaluate_crisp_abcd_baseline(batch, calib)
        assert np.allclose(scores, [0.0, 1.0, 0.5])

    def test_biomarker_result_sequence_input(self):
        calib = _make_dummy_calibration()
        b1 = BiomarkerResult(A=0.1, B_fuzzy=0.2, C=0.3, D_verified=0.4, alpha=0.5, B_geo=0.1, B_grad=0.2)
        b2 = BiomarkerResult(A=0.8, B_fuzzy=0.8, C=0.8, D_verified=0.8, alpha=0.9, B_geo=0.8, B_grad=0.8)

        scores = evaluate_crisp_abcd_baseline([b1, b2], calib)
        assert np.allclose(scores, [0.0, 1.0])

    def test_reject_wrong_shape(self):
        calib = _make_dummy_calibration()
        with pytest.raises(ValueError, match="shape"):
            evaluate_crisp_abcd_baseline(np.zeros((3, 3)), calib)


class TestAblationStudy:
    """Test run_ablation_study and artifact reporting."""

    def test_run_ablation_study_3_tiers(self):
        calib = _make_dummy_calibration()

        # Cohort of 10 samples: 5 Benign with ruler artifact (CNN says high, FIS suppresses)
        # 5 True Melanoma (CNN and FIS both high)
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        # Pure CNN is tricked by ruler on the 5 benign cases
        p_cnn = np.array([0.85, 0.90, 0.78, 0.82, 0.88, 0.92, 0.95, 0.89, 0.94, 0.91])

        # Biomarkers:
        # Benign cases have low A, B, C and low alpha (attention on ruler), so D_verified is suppressed
        # Melanoma cases have high A, B, C, high alpha
        biomarkers = []
        for i in range(5):
            # Benign with ruler: A, B, C low (< 0.20), alpha low (0.15) -> D_verified = 0.85 * 0.15 = 0.1275
            biomarkers.append([0.15, 0.18, 0.12, 0.12])
        for i in range(5):
            # Melanoma: A, B, C high (> 0.70), alpha high (0.85) -> D_verified = 0.90 * 0.85 = 0.765
            biomarkers.append([0.75, 0.80, 0.72, 0.78])

        report = run_ablation_study(y_true, p_cnn, biomarkers, calib)

        assert isinstance(report, AblationReport)
        assert report.sample_count == 10
        assert "1_pure_mobilenetv2_blackbox" in report.tiers
        assert "2_crisp_abcd_fixed_thresholds" in report.tiers
        assert "3_proposed_dermafuzzy_xai" in report.tiers

        tier1 = report.tiers["1_pure_mobilenetv2_blackbox"]
        tier3 = report.tiers["3_proposed_dermafuzzy_xai"]

        # In Tier 1, Pure CNN flags all 5 benign cases as malignant -> Specificity = 0.0
        assert tier1.fp == 5
        assert tier1.specificity == 0.0

        # In Tier 3, DermaFuzzy-XAI suppresses all 5 benign cases -> Specificity = 1.0!
        assert tier3.fp == 0
        assert tier3.specificity == 1.0
        assert tier3.sensitivity == 1.0
        assert tier3.accuracy == 1.0

        # Verify export formatting
        md = report.to_markdown_table()
        assert "Tier 1: Pure MobileNetV2 Black-Box" in md or "1_pure_mobilenetv2_blackbox" in md
        assert "Tier 3: Proposed DermaFuzzy-XAI" in md or "3_proposed_dermafuzzy_xai" in md
        assert "Specificity" in md
        assert "AUC-ROC" in md

        csv_str = report.to_csv()
        assert "tier,tp,tn,fp,fn" in csv_str
        assert "3_proposed_dermafuzzy_xai" in csv_str
