"""Unit tests for Q1 empirical artifact perturbation study and statistical analysis.

Covers (zero TensorFlow required):
- Mask-aware peripheral artifact overlays: overlay_ruler, overlay_ink, overlay_gel
- Hermite S-Curve saliency trust gate, off-lesion masking, resize parity, and CSV round-trip
- CLI `--selftest` execution without TensorFlow or dataset
- Exact tie-aware ROC-AUC, step-function PR-AUC, DeLong ROC test, McNemar exact test,
  stratified bootstrap 95% CIs, Bayes prevalence-adjusted PPV/NPV, and operating points
- Edge-case rejection of bool/np.bool_, NaN, Inf, empty arrays, and single-class inputs
- Regression test for [0, 100] risk_* -> [0, 1] normalization at threshold 0.50
- Constitution Principle I compliance (zero TensorFlow / PyTorch module leakage)
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from scipy import stats as scipy_stats

from dermafuzzy_core.config import CANONICAL_FEATURE_NAMES, load_config
from dermafuzzy_core.evaluation import compute_clinical_metrics
from dermafuzzy_core.fis_engine import FISCalibration
from dermafuzzy_core.preprocessing import preprocess_image
from services.colab_trainer.artifact_perturbation_study import (
    ARTIFACT_CONDITIONS,
    CSV_FIELDNAMES,
    CaseEvaluationInput,
    _resize_to_config,
    apply_artifact_condition,
    compute_hermite_verified_confidence,
    compute_mask_iou,
    evaluate_case_variants,
    neutralize_off_lesion_pixels,
    overlay_gel,
    overlay_ink,
    overlay_ruler,
    read_cases_csv,
    write_cases_csv,
)
from services.colab_trainer.stats_analysis import (
    CONDITIONS,
    PAIRED_COMPARISONS,
    TIER_COLUMNS,
    analyze_study_tables,
    compute_pr_auc,
    compute_prevalence_table,
    compute_roc_auc_exact,
    delong_roc_test,
    load_cases_table,
    mcnemar_exact_test,
    prevalence_adjusted_ppv_npv,
    run_stats_analysis,
    stratified_bootstrap_ci,
)


def _make_skin_phantom(size: int = 128, radius: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """Create read-only uint8 RGB skin image with centered brown lesion and binary mask."""
    rgb = np.full((size, size, 3), (205, 170, 150), dtype=np.uint8)
    y_grid, x_grid = np.ogrid[:size, :size]
    center = size // 2
    mask = ((x_grid - center) ** 2 + (y_grid - center) ** 2 <= radius ** 2).astype(np.uint8)
    rgb[mask > 0] = (75, 45, 35)
    rgb.flags.writeable = False
    mask.flags.writeable = False
    return rgb, mask


def _make_standard_calibration() -> FISCalibration:
    """Create quartile calibration matching CANONICAL_FEATURE_NAMES."""
    return FISCalibration(
        cutoffs={feat: (0.25, 0.50, 0.75) for feat in CANONICAL_FEATURE_NAMES},
        n_samples=100,
        generated_at="2026-11-26T00:00:00+00:00",
        config_hash="0" * 64,
    )


class TestArtifactOverlays:
    """Unit tests for mask-aware peripheral artifact overlays."""

    def test_ruler_selects_lowest_overlap_edge_and_is_deterministic(self) -> None:
        rgb, _ = _make_skin_phantom(size=128, radius=20)
        asym_mask = np.zeros((128, 128), dtype=np.uint8)
        asym_mask[:16, :] = 1
        asym_mask[-16:, :] = 1
        asym_mask[:, :16] = 1
        orig_copy = rgb.copy()

        out_a = overlay_ruler(rgb, asym_mask, np.random.default_rng(123))
        out_b = overlay_ruler(rgb, asym_mask, np.random.default_rng(123))

        assert np.array_equal(rgb, orig_copy)
        assert out_a.shape == rgb.shape and out_a.dtype == np.uint8
        assert np.array_equal(out_a, out_b)
        assert np.array_equal(out_a[:, :16], rgb[:, :16])
        assert np.any(out_a[:, -10:] != rgb[:, -10:])

    def test_ink_maintains_6px_clearance_and_determinism(self) -> None:
        rgb, mask = _make_skin_phantom(size=140, radius=24)
        orig_copy = rgb.copy()

        out_a = overlay_ink(rgb, mask, np.random.default_rng(42))
        out_b = overlay_ink(rgb, mask, np.random.default_rng(42))

        assert np.array_equal(rgb, orig_copy)
        assert out_a.shape == rgb.shape and out_a.dtype == np.uint8
        assert np.array_equal(out_a, out_b)

        modified = np.any(out_a != rgb, axis=2)
        assert np.any(modified)
        assert not np.any(modified & (mask > 0))

        k6 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        dilated = cv2.dilate(mask, k6) > 0
        assert not np.any(modified & dilated), "Ink entered 6px margin outside lesion mask"

        max_loc = np.unravel_index(
            np.argmax(np.sum(np.abs(out_a.astype(int) - rgb.astype(int)), axis=2)), (140, 140)
        )
        px = out_a[max_loc]
        assert int(px[2]) > int(px[1])

    def test_full_frame_lesion_triggers_corner_fallback_overlays(self) -> None:
        rgb, _ = _make_skin_phantom(size=96, radius=20)
        full_mask = np.ones((96, 96), dtype=np.uint8)

        ink_out = overlay_ink(rgb, full_mask, np.random.default_rng(7))
        gel_out = overlay_gel(rgb, full_mask, np.random.default_rng(7))

        assert ink_out.shape == rgb.shape and ink_out.dtype == np.uint8
        assert gel_out.shape == rgb.shape and gel_out.dtype == np.uint8
        assert not np.array_equal(ink_out, rgb)
        assert not np.array_equal(gel_out, rgb)

    def test_gel_blends_peripheral_skin_with_alpha_035(self) -> None:
        rgb, mask = _make_skin_phantom(size=128, radius=24)
        orig_copy = rgb.copy()

        out_a = overlay_gel(rgb, mask, np.random.default_rng(99))
        out_b = overlay_gel(rgb, mask, np.random.default_rng(99))

        assert np.array_equal(rgb, orig_copy)
        assert out_a.shape == rgb.shape and out_a.dtype == np.uint8
        assert np.array_equal(out_a, out_b)

        modified = np.any(out_a != rgb, axis=2)
        assert np.any(modified)
        assert not np.any(modified & (mask > 0))

        expected = np.clip(
            np.round(0.65 * rgb[modified].astype(np.float32) + 0.35 * 255.0), 0, 255
        ).astype(np.uint8)
        assert np.all(np.abs(out_a[modified].astype(np.int16) - expected.astype(np.int16)) <= 1)

    @pytest.mark.parametrize(
        ("rgb_fn", "mask_fn", "rng_val", "cond"),
        [
            (lambda r, m: "not_array", lambda r, m: m, np.random.default_rng(1), "ruler"),
            (lambda r, m: r.astype(np.float32), lambda r, m: m, np.random.default_rng(1), "ink"),
            (lambda r, m: r, lambda r, m: np.zeros((32, 32), dtype=np.uint8), np.random.default_rng(1), "gel"),
            (lambda r, m: r, lambda r, m: np.where(m > 0, np.nan, 0.0), np.random.default_rng(1), "gel"),
            (lambda r, m: r, lambda r, m: m, 42, "ruler"),
            (lambda r, m: r, lambda r, m: m, np.random.default_rng(1), "vignette"),
        ],
    )
    def test_overlays_reject_malformed_inputs(
        self,
        rgb_fn,
        mask_fn,
        rng_val,
        cond: str,
    ) -> None:
        rgb, mask = _make_skin_phantom(size=64, radius=12)
        with pytest.raises(ValueError):
            apply_artifact_condition(rgb_fn(rgb, mask), mask_fn(rgb, mask), cond, rng_val)


class TestArtifactVariantsAndCSV:
    """Tests for Hermite gate, resize parity, off-lesion masking, CSV round-trip, and --selftest."""

    def test_hermite_gate_saturates_at_tau_bounds_and_is_monotonic(self) -> None:
        assert math.isclose(compute_hermite_verified_confidence(0.90, 0.0), 0.0)
        assert math.isclose(compute_hermite_verified_confidence(0.90, 0.10), 0.0)
        assert math.isclose(compute_hermite_verified_confidence(0.90, 0.24), 0.45, abs_tol=1e-9)
        assert math.isclose(compute_hermite_verified_confidence(0.90, 0.38), 0.90)
        assert math.isclose(compute_hermite_verified_confidence(0.90, 0.95), 0.90)

        vals = [compute_hermite_verified_confidence(0.8, a) for a in np.linspace(0.0, 1.0, 21)]
        assert all(vals[i] <= vals[i + 1] + 1e-12 for i in range(len(vals) - 1))

        with pytest.raises(ValueError):
            compute_hermite_verified_confidence(True, 0.5)
        with pytest.raises(ValueError):
            compute_hermite_verified_confidence(np.bool_(True), 0.5)
        with pytest.raises(ValueError):
            compute_hermite_verified_confidence(0.5, float("nan"))

    def test_resize_matches_preprocess_image_baseline_identically(self) -> None:
        cfg = load_config()
        raw_ham_rgb = np.random.default_rng(5).integers(0, 255, size=(450, 600, 3), dtype=np.uint8)
        prep_direct = preprocess_image(raw_ham_rgb, cfg)
        prep_via_study = preprocess_image(_resize_to_config(raw_ham_rgb, cfg), cfg)
        assert np.array_equal(prep_direct.clean_rgb, prep_via_study.clean_rgb)
        assert np.array_equal(prep_direct.lesion_mask, prep_via_study.lesion_mask)
        assert np.array_equal(prep_direct.boundary_ring, prep_via_study.boundary_ring)

    def test_neutralize_off_lesion_pixels_sets_bg_to_128_without_mutation(self) -> None:
        rgb, mask = _make_skin_phantom(size=64, radius=14)
        orig = rgb.copy()
        masked_rgb = neutralize_off_lesion_pixels(rgb, mask)
        assert np.array_equal(rgb, orig)
        assert np.all(masked_rgb[mask == 0] == 128)
        assert np.array_equal(masked_rgb[mask > 0], rgb[mask > 0])

    def test_evaluate_variants_and_csv_roundtrip(self, tmp_path: Path) -> None:
        cfg = load_config()
        calib = _make_standard_calibration()
        rgb, _ = _make_skin_phantom(size=cfg.dataset.image_size[0], radius=48)
        prep_clean = preprocess_image(rgb, cfg)
        heatmap = np.zeros((32, 32), dtype=np.float32)
        heatmap[10:22, 10:22] = 1.0

        rows = []
        for cond in ARTIFACT_CONDITIONS:
            corrupted = apply_artifact_condition(rgb, prep_clean.lesion_mask, cond, np.random.default_rng(42))
            prep_cond = prep_clean if cond == "clean" else preprocess_image(corrupted, cfg)
            res = evaluate_case_variants(
                CaseEvaluationInput(
                    image_id="ISIC_0001",
                    label=1,
                    condition=cond,
                    p_cnn=0.88,
                    p_masked=0.82,
                    heatmap=heatmap,
                    prep_clean=prep_clean,
                    prep_cond=prep_cond,
                ),
                calibration=calib,
                config=cfg,
            )
            rows.append(res)
            if cond == "clean":
                assert math.isclose(res.mask_iou_vs_clean, 1.0)
            assert set(res.to_csv_row().keys()) == set(CSV_FIELDNAMES)

        csv_out = write_cases_csv(rows, tmp_path / "artifact_study_cases.csv")
        loaded = read_cases_csv(csv_out)
        assert len(loaded) == len(ARTIFACT_CONDITIONS)
        assert math.isclose(loaded[0]["risk_sqrt"], rows[0].risk_sqrt, abs_tol=1e-6)
        assert math.isclose(compute_mask_iou(prep_clean.lesion_mask, prep_clean.lesion_mask), 1.0)

    def test_cli_selftest_exits_zero_offline(self) -> None:
        proc = subprocess.run(
            [sys.executable, "services/colab_trainer/artifact_perturbation_study.py", "--selftest"],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, f"Selftest failed: {proc.stderr or proc.stdout}"
        assert "SELFTEST PASSED" in proc.stdout


class TestStatisticalAnalysis:
    """Tests for pure NumPy/SciPy DeLong, McNemar, PR-AUC, bootstrap, Bayes PPV/NPV, and edge rejections."""

    def test_roc_auc_matches_core_evaluation_with_ties(self) -> None:
        y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        scores = np.array([0.9, 0.2, 0.5, 0.5, 0.5, 0.5, 0.8, 0.1, 0.7, 0.3])
        assert math.isclose(compute_roc_auc_exact(y, scores), compute_clinical_metrics(y, scores).auc_roc, abs_tol=1e-12)

    def test_compute_pr_auc_handles_perfect_constant_and_ties(self) -> None:
        y = np.array([1, 1, 0, 0])
        assert math.isclose(compute_pr_auc(y, [0.9, 0.8, 0.2, 0.1]), 1.0, abs_tol=1e-12)
        assert math.isclose(compute_pr_auc(y, [0.5, 0.5, 0.5, 0.5]), 0.5, abs_tol=1e-12)
        assert math.isclose(compute_pr_auc([1, 0, 1, 0], [0.9, 0.5, 0.5, 0.1]), 5.0 / 6.0, abs_tol=1e-12)

    def test_delong_identical_separated_and_zero_variance_guard(self) -> None:
        rng = np.random.default_rng(7)
        y = np.array([0] * 40 + [1] * 40)
        scores_strong = np.concatenate([rng.uniform(0.05, 0.45, 40), rng.uniform(0.55, 0.95, 40)])
        scores_weak = np.concatenate([rng.uniform(0.25, 0.70, 40), rng.uniform(0.30, 0.75, 40)])

        res_same = delong_roc_test(y, scores_strong, scores_strong)
        assert math.isclose(res_same["auc_diff"], 0.0, abs_tol=1e-15)
        assert math.isclose(res_same["z_stat"], 0.0, abs_tol=1e-15)
        assert math.isclose(res_same["p_value"], 1.0, abs_tol=1e-15)

        res_sep = delong_roc_test(y, scores_strong, scores_weak)
        res_swap = delong_roc_test(y, scores_weak, scores_strong)
        assert res_sep["auc_a"] > res_sep["auc_b"] and res_sep["p_value"] < 0.05
        assert math.isclose(res_sep["p_value"], res_swap["p_value"], abs_tol=1e-12)
        assert math.isclose(res_sep["z_stat"], -res_swap["z_stat"], abs_tol=1e-12)

        y_small = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        res_guard = delong_roc_test(
            y_small,
            np.array([0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9]),
            np.array([0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1]),
        )
        assert math.isfinite(res_guard["z_stat"]) and res_guard["p_value"] < 0.001

        with pytest.raises(ValueError, match="at least 2 positive"):
            delong_roc_test([0, 0, 1], [0.1, 0.2, 0.9], [0.2, 0.3, 0.8])

    def test_mcnemar_exact_matches_binomial_for_asymmetric_discordant_cells(self) -> None:
        y = np.array([1] * 25 + [0] * 25)
        res_id = mcnemar_exact_test(y, y.astype(float), y.astype(float), threshold=0.50)
        assert res_id["b_a_correct_b_wrong"] == 0 and res_id["c_a_wrong_b_correct"] == 0
        assert math.isclose(res_id["p_value"], 1.0)

        pred_a, pred_b = y.copy(), y.copy()
        pred_a[:2] = 1 - pred_a[:2]
        pred_b[2:20] = 1 - pred_b[2:20]

        res_asym = mcnemar_exact_test(y, pred_a.astype(float), pred_b.astype(float), threshold=0.50)
        expected_p = float(scipy_stats.binomtest(18, n=20, p=0.5, alternative="two-sided").pvalue)
        assert res_asym["b_a_correct_b_wrong"] == 18 and res_asym["c_a_wrong_b_correct"] == 2
        assert math.isclose(res_asym["p_value"], expected_p, abs_tol=1e-12)
        assert res_asym["p_value"] < 0.001

    def test_stratified_bootstrap_ci_is_reproducible_and_bounded(self) -> None:
        rng = np.random.default_rng(11)
        y = np.array([0] * 30 + [1] * 30)
        scores = np.concatenate([rng.uniform(0.1, 0.55, 30), rng.uniform(0.45, 0.95, 30)])

        ci_1 = stratified_bootstrap_ci(y, scores, threshold=0.50, n_boot=300, seed=42)
        ci_2 = stratified_bootstrap_ci(y, scores, threshold=0.50, n_boot=300, seed=42)
        assert ci_1 == ci_2

        for metric_name in ("accuracy", "sensitivity", "specificity", "precision", "f1_score", "roc_auc", "pr_auc"):
            entry = ci_1[metric_name]
            assert 0.0 <= entry["ci_low"] <= entry["point"] <= entry["ci_high"] <= 1.0

    def test_bayes_ppv_decreases_monotonically_with_lower_prevalence(self) -> None:
        sens, spec = 0.80, 0.85
        table = compute_prevalence_table(sens, spec)
        assert set(table.keys()) == {"2:1", "6:1", "10:1"}

        for label, ratio in (("2:1", 2.0), ("6:1", 6.0), ("10:1", 10.0)):
            pi = 1.0 / (1.0 + ratio)
            exp_ppv = (sens * pi) / (sens * pi + (1.0 - spec) * (1.0 - pi))
            exp_npv = (spec * (1.0 - pi)) / (spec * (1.0 - pi) + (1.0 - sens) * pi)
            assert math.isclose(table[label]["prevalence"], pi, abs_tol=1e-12)
            assert math.isclose(table[label]["ppv"], exp_ppv, abs_tol=1e-12)
            assert math.isclose(table[label]["npv"], exp_npv, abs_tol=1e-12)

        assert table["2:1"]["ppv"] > table["6:1"]["ppv"] > table["10:1"]["ppv"]
        assert table["2:1"]["npv"] < table["6:1"]["npv"] < table["10:1"]["npv"]

    @pytest.mark.parametrize(
        ("y_bad", "s1_bad", "s2_bad"),
        [
            ([True, False, True, False], [0.1, 0.2, 0.8, 0.9], [0.2, 0.3, 0.7, 0.8]),
            ([0, 0, 1, 1], np.array([True, False, True, False]), [0.2, 0.3, 0.7, 0.8]),
            ([0, 0, 1, 1], [0.1, float("nan"), 0.8, 0.9], [0.2, 0.3, 0.7, 0.8]),
            ([0, 0, 1, 1], [0.1, 0.2, float("inf"), 0.9], [0.2, 0.3, 0.7, 0.8]),
            ([], [], []),
            ([0, 0, 0, 0], [0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5]),
            ([1, 1, 1, 1], [0.6, 0.7, 0.8, 0.9], [0.5, 0.6, 0.7, 0.8]),
        ],
    )
    def test_statistical_functions_reject_bool_nan_inf_empty_and_single_class(
        self,
        y_bad,
        s1_bad,
        s2_bad,
    ) -> None:
        with pytest.raises(ValueError):
            delong_roc_test(y_bad, s1_bad, s2_bad)
        with pytest.raises(ValueError):
            mcnemar_exact_test(y_bad, s1_bad, s2_bad, threshold=0.50)
        with pytest.raises(ValueError):
            compute_pr_auc(y_bad, s1_bad)
        with pytest.raises(ValueError):
            stratified_bootstrap_ci(y_bad, s1_bad, threshold=0.50, n_boot=50, seed=42)

    def test_scalar_params_reject_bool_nan_inf_across_stats_functions(self) -> None:
        y = [0, 0, 1, 1]
        s = [0.1, 0.2, 0.8, 0.9]
        with pytest.raises(ValueError):
            mcnemar_exact_test(y, s, s, threshold=True)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            mcnemar_exact_test(y, [75.0, 20.0, 80.0, 10.0], s, threshold=0.50)
        with pytest.raises(ValueError):
            stratified_bootstrap_ci(y, s, threshold=0.50, n_boot=True, seed=42)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            stratified_bootstrap_ci(y, s, threshold=0.50, n_boot=50, seed=True)  # type: ignore[arg-type]
        for bad in (True, np.bool_(True), float("nan"), float("inf"), 0.0, -1.0):
            with pytest.raises(ValueError):
                prevalence_adjusted_ppv_npv(0.8, 0.8, bad)  # type: ignore[arg-type]
        for bad_rate in (True, np.bool_(True), float("nan"), float("inf"), -0.1, 1.1):
            with pytest.raises(ValueError):
                prevalence_adjusted_ppv_npv(bad_rate, 0.8, 2.0)  # type: ignore[arg-type]

    def test_risk_percent_normalization_prevents_zero_sensitivity_at_050_threshold(self, tmp_path: Path) -> None:
        """Regression test verifying risk_* [0, 100] normalization works at 0.50 threshold."""
        rng = np.random.default_rng(42)
        n_per_class = 30
        rows = []
        for cond in CONDITIONS:
            for idx in range(2 * n_per_class):
                lbl = 1 if idx >= n_per_class else 0
                is_pos = lbl == 1
                p_cnn = float(rng.uniform(0.65, 0.95) if is_pos else rng.uniform(0.08, 0.38))
                if cond != "clean" and not is_pos and idx < 8:
                    p_cnn = float(rng.uniform(0.75, 0.92))
                p_masked = float(rng.uniform(0.62, 0.90) if is_pos else rng.uniform(0.10, 0.35))
                p_crisp = float(rng.choice([0.75, 1.0]) if is_pos else rng.choice([0.0, 0.25]))
                risk_sqrt = float(rng.uniform(58.0, 92.0) if is_pos else rng.uniform(12.0, 38.0))
                risk_hermite = float(rng.uniform(60.0, 94.0) if is_pos else rng.uniform(10.0, 35.0))
                risk_nobg = float(rng.uniform(52.0, 88.0) if is_pos else rng.uniform(15.0, 42.0))

                rows.append(
                    {
                        "image_id": f"ISIC_{idx:04d}",
                        "label": lbl,
                        "condition": cond,
                        "p_cnn": f"{p_cnn:.6f}",
                        "p_masked": f"{p_masked:.6f}",
                        "p_crisp": f"{p_crisp:.6f}",
                        "d_ver_sqrt": f"{p_cnn * 0.8:.6f}",
                        "d_ver_hermite": f"{p_cnn * 0.85:.6f}",
                        "alpha": f"{0.80 if is_pos else 0.15:.6f}",
                        "A": f"{0.70 if is_pos else 0.20:.6f}",
                        "B_geo": f"{0.72 if is_pos else 0.22:.6f}",
                        "B_grad": f"{0.74 if is_pos else 0.24:.6f}",
                        "B_fuzzy": f"{0.73 if is_pos else 0.23:.6f}",
                        "C": f"{0.68 if is_pos else 0.18:.6f}",
                        "risk_sqrt": f"{risk_sqrt:.6f}",
                        "risk_hermite": f"{risk_hermite:.6f}",
                        "risk_no_bgrad": f"{risk_nobg:.6f}",
                        "fallback_used": 0,
                        "mask_iou_vs_clean": f"{1.0 if cond == 'clean' else 0.94:.6f}",
                    }
                )

        csv_path = write_cases_csv(rows, tmp_path / "artifact_study_cases.csv")
        by_cond = load_cases_table(csv_path)
        for tier in ("risk_sqrt", "risk_hermite", "risk_no_bgrad"):
            vals = by_cond["clean"][tier]
            assert 0.0 <= float(np.min(vals)) and float(np.max(vals)) <= 1.0

        summary = analyze_study_tables(by_cond, n_boot=100, seed=42, threshold=0.50)
        for tier in TIER_COLUMNS:
            ops = summary["operating_points_clean"][tier]
            std = ops["standard_0_50"]
            assert std["sensitivity"] >= 0.85
            assert std["specificity"] >= 0.85
            assert ops["clinical_triage_0_35"]["threshold"] == pytest.approx(0.35)
            assert ops["spec_at_sens_90"]["sensitivity"] >= 0.90
            assert "youden_j" in ops["youden_j_optimal"]

        assert len(summary["paired_significance"]["clean"]) == len(PAIRED_COMPARISONS)
        assert set(summary["artifact_resilience"].keys()) == {"ruler", "ink", "gel"}

        json_p, md_p = run_stats_analysis(csv_path, tmp_path / "out", n_boot=100, seed=42)
        assert json_p.is_file() and md_p.is_file()
        parsed_json = json.loads(json_p.read_text(encoding="utf-8"))
        assert parsed_json["metadata"]["threshold_normalized"] == 0.50
        assert "DeLong" in md_p.read_text(encoding="utf-8")

    def test_zero_heavy_ml_imports_in_q1_modules(self) -> None:
        code = (
            "import sys\n"
            "import services.colab_trainer.artifact_perturbation_study\n"
            "import services.colab_trainer.stats_analysis\n"
            "assert 'tensorflow' not in sys.modules\n"
            "assert 'torch' not in sys.modules\n"
            "assert 'pandas' not in sys.modules\n"
            "assert 'sklearn' not in sys.modules\n"
        )
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert proc.returncode == 0, f"Unexpected top-level heavy import: {proc.stderr or proc.stdout}"
