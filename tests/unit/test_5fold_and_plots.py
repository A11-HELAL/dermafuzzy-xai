"""Offline unit tests for 5-fold lesion-grouped CV and Q1 publication figures.

Covers (zero TensorFlow required):
- Lesion-ID zero-leakage guard, missing lesion_id fallback warning, and NaN backfill
- 5-fold Mean +/- SD (ddof=1) aggregation across full_unseen and balanced_1to1 cohorts
- Pooled OOF 4-tier evaluation (including Clinical Triage risk >= 35.0) & DeLong/McNemar stats
- Per-fold checkpoint OOF CSV round-trip and resume corruption / fold-mismatch guards
- Single-class fold rejection and bool / np.bool_ / NaN / Inf rejection in aggregation helpers
- Missing input FileNotFoundError, empty/invalid CSV & JSON rejection, and zero figure-handle leaks
- CLI `--selftest` execution exiting cleanly with zero TensorFlow / PyTorch / pyplot leakage
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from services.colab_trainer.plot_q1_figures import (
    FIGURE_FILENAMES,
    PlotInputsConfig,
    _generate_selftest_cases_csv,
    render_all_q1_figures,
)
from services.colab_trainer.run_5fold_cv import (
    CV_TIERS,
    N_OUTER_FOLDS,
    OOF_FIELDNAMES,
    _build_selftest_oof_rows,
    aggregate_cv_results,
    assert_lesion_disjoint,
    compute_fold_mean_sd,
    compute_pooled_significance,
    evaluate_rows_4tiers,
    load_and_verify_fold_checkpoint,
    load_fold_oof_csv,
    resolve_lesion_groups,
    save_cv_outputs,
    validate_binary_split_labels,
    write_fold_oof_csv,
)
from services.colab_trainer.stats_analysis import run_stats_analysis


def _make_clean_oof_row(
    fold: int = 0,
    idx: int = 0,
    label: int = 0,
    in_bal: int = 1,
    risk_score: float = 22.0,
) -> dict[str, Any]:
    """Build a valid synthetic OOF prediction record for offline CV tests."""
    return {
        "fold": fold,
        "image_id": f"IMG_{fold}_{idx:03d}",
        "lesion_id": f"LES_{fold}_{idx:03d}",
        "label": label,
        "in_balanced_test": in_bal,
        "p_cnn": 0.85 if label == 1 else 0.15,
        "p_crisp": 0.75 if label == 1 else 0.0,
        "d_ver_sqrt": 0.80 if label == 1 else 0.10,
        "alpha": 0.80 if label == 1 else 0.15,
        "A": 0.70 if label == 1 else 0.20,
        "B_geo": 0.70 if label == 1 else 0.20,
        "B_grad": 0.70 if label == 1 else 0.20,
        "B_fuzzy": 0.70 if label == 1 else 0.20,
        "C": 0.70 if label == 1 else 0.20,
        "risk_score": risk_score,
        "fallback_used": 0,
    }


def _make_two_class_rows(fold: int = 0, n_each: int = 4) -> list[dict[str, Any]]:
    """Build minimal valid two-class OOF row batch for mutation/edge-case tests."""
    rows = [_make_clean_oof_row(fold=fold, idx=i, label=1, risk_score=75.0) for i in range(n_each)]
    rows.extend(
        _make_clean_oof_row(fold=fold, idx=n_each + i, label=0, risk_score=20.0)
        for i in range(n_each)
    )
    return rows


def _make_complete_plot_inputs(tmp_path: Path) -> PlotInputsConfig:
    """Create valid synthetic CSV/JSON inputs paired with real fis_calibration.json."""
    repo_root = Path(__file__).resolve().parents[2]
    real_calib = repo_root / "artifacts" / "fis_calibration.json"
    assert real_calib.is_file(), "Real artifacts/fis_calibration.json must exist"

    cases_csv = _generate_selftest_cases_csv(tmp_path / "artifact_study_cases.csv")
    stats_json, _ = run_stats_analysis(cases_csv, tmp_path, n_boot=30, seed=42)
    return PlotInputsConfig(
        cases_csv=cases_csv,
        stats_json=stats_json,
        calibration_json=real_calib,
        out_dir=tmp_path / "figs",
    )


class TestFiveFoldCVOffline:
    """Offline tests for run_5fold_cv.py (zero TensorFlow required)."""

    def test_overlapping_lesion_ids_raise_runtime_error_and_disjoint_passes(self) -> None:
        assert_lesion_disjoint(["HAM_0001", "HAM_0002"], ["HAM_0003", "HAM_0004"])

        with pytest.raises(RuntimeError, match="DATA LEAKAGE DETECTED"):
            assert_lesion_disjoint(["HAM_0001", "HAM_0002"], ["HAM_0002", "HAM_0005"])

        with pytest.raises(ValueError, match="non-empty"):
            assert_lesion_disjoint([], ["HAM_0001"])
        with pytest.raises(ValueError, match="non-empty"):
            assert_lesion_disjoint(["HAM_0001"], [])

    def test_missing_lesion_id_column_or_nan_warns_and_backfills_from_image_id(self) -> None:
        explicit = resolve_lesion_groups(["IMG_1", "IMG_2"], ["LES_A", "LES_B"])
        assert explicit == ["LES_A", "LES_B"]

        with pytest.warns(UserWarning, match="lacks 'lesion_id'"):
            fallback_all = resolve_lesion_groups(["IMG_1", "IMG_2"], None)
        assert fallback_all == ["IMG_1", "IMG_2"]

        with pytest.warns(UserWarning, match="Backfilled"):
            backfilled = resolve_lesion_groups(
                ["IMG_1", "IMG_2", "IMG_3", "IMG_4"],
                ["LES_1", None, "", float("nan")],
            )
        assert backfilled == ["LES_1", "IMG_2", "IMG_3", "IMG_4"]

        with pytest.raises(ValueError, match="empty"):
            resolve_lesion_groups([], None)
        with pytest.raises(ValueError, match="does not match"):
            resolve_lesion_groups(["IMG_1"], ["LES_1", "LES_2"])

    def test_five_fold_mean_sd_triage_and_pooled_significance_math(self) -> None:
        fold_rows: list[dict[str, Any]] = []
        for k in range(N_OUTER_FOLDS):
            for idx in range(30):
                lbl = 1 if idx < 10 else 0
                in_bal = 1 if (lbl == 1 or idx < 20) else 0
                # Borderline melanoma (risk=40) fails standard >= 50 but passes triage >= 35
                risk_sc = 40.0 if idx == 0 else (78.0 if lbl == 1 else 22.0)
                fold_rows.append(
                    _make_clean_oof_row(fold=k, idx=idx, label=lbl, in_bal=in_bal, risk_score=risk_sc)
                )

        summary = aggregate_cv_results(fold_rows)
        assert summary["n_folds"] == N_OUTER_FOLDS
        full = summary["cohorts"]["full_unseen"]
        bal = summary["cohorts"]["balanced_1to1"]
        assert full["n_total"] == 150 and full["n_mel"] == 50 and full["n_nv"] == 100
        assert bal["n_total"] == 100 and bal["n_mel"] == 50 and bal["n_nv"] == 50

        assert math.isclose(full["mean_sd"]["tier3_standard"]["sensitivity"]["mean"], 0.90, abs_tol=1e-12)
        assert math.isclose(full["mean_sd"]["tier3_triage"]["sensitivity"]["mean"], 1.00, abs_tol=1e-12)
        assert math.isclose(full["mean_sd"]["tier3_standard"]["sensitivity"]["sd"], 0.0, abs_tol=1e-12)

        for cohort in (full, bal):
            sig = cohort["pooled_significance"]
            assert "tier3_standard_vs_tier2_crisp" in sig
            assert "tier3_standard_vs_tier1_cnn" in sig
            for pair in sig.values():
                assert math.isfinite(pair["delong"]["p_value"])
                assert math.isfinite(pair["mcnemar"]["p_value"])

        fold_accs = [0.70, 0.75, 0.80, 0.85, 0.90]
        synth_fold_dicts = [
            {
                tier: {
                    m: (fold_accs[k] if m == "accuracy" else 0.8)
                    for m in ("accuracy", "sensitivity", "specificity", "precision", "f1_score", "roc_auc", "pr_auc")
                }
                for tier in CV_TIERS
            }
            for k in range(5)
        ]
        agg_sd = compute_fold_mean_sd(synth_fold_dicts)
        assert math.isclose(agg_sd["tier3_standard"]["accuracy"]["mean"], 0.80, abs_tol=1e-12)
        assert math.isclose(agg_sd["tier3_standard"]["accuracy"]["sd"], float(np.std(fold_accs, ddof=1)), abs_tol=1e-12)

    def test_fold_checkpoint_roundtrip_writes_and_verifies_exports(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(77)
        rows = _build_selftest_oof_rows(rng)
        resumed: list[dict[str, Any]] = []
        for k in range(N_OUTER_FOLDS):
            fold_csv = write_fold_oof_csv(
                [r for r in rows if r["fold"] == k],
                tmp_path / f"cv_fold_{k}_oof.csv",
            )
            loaded = load_and_verify_fold_checkpoint(fold_csv, expected_fold=k)
            assert len(loaded) == len([r for r in rows if r["fold"] == k])
            assert loaded[0]["fold"] == k
            resumed.extend(loaded)

        summary = aggregate_cv_results(resumed)
        save_cv_outputs(summary, resumed, tmp_path)

        assert (tmp_path / "cv_summary.json").is_file()
        assert (tmp_path / "cv_report.md").is_file()
        assert (tmp_path / "cv_oof_predictions.csv").is_file()
        md_text = (tmp_path / "cv_report.md").read_text(encoding="utf-8")
        assert "Mean ± SD Across 5 Folds" in md_text
        assert "tier3_triage" in md_text

    @pytest.mark.parametrize(
        "csv_content",
        [
            "",
            ",".join(OOF_FIELDNAMES) + "\n",
            "fold,image_id,label\n0,IMG_0,1\n",
            ",".join(OOF_FIELDNAMES) + "\n0,IMG_0,LES_0,1\n",
            ",".join(OOF_FIELDNAMES) + "\n0,IMG_0,LES_0,1,1,0.8,0.7,0.6,0.5,0.5,0.5,0.5,0.5,0.5,75.0,0,EXTRA\n",
            ",".join(OOF_FIELDNAMES) + "\n0,IMG_0,LES_0,1,1,0.8,0.7,0.6,0.5,0.5,0.5,0.5,0.5,0.5,nan,0\n",
            ",".join(OOF_FIELDNAMES) + "\n0,IMG_0,LES_0,1,1,0.8,0.7,0.6,0.5,0.5,0.5,0.5,0.5,0.5,105.0,0\n",
            ",".join(OOF_FIELDNAMES) + "\n0,IMG_0,LES_0,1,1,0.8,0.7,0.6,0.5,0.5,0.5,0.5,0.5,0.5,75.0,maybe\n",
            ",".join(OOF_FIELDNAMES) + "\n0,IMG_0,LES_0,True,1,0.8,0.7,0.6,0.5,0.5,0.5,0.5,0.5,0.5,75.0,0\n",
            ",".join(OOF_FIELDNAMES) + "\n0,,LES_0,1,1,0.8,0.7,0.6,0.5,0.5,0.5,0.5,0.5,0.5,75.0,0\n",
        ],
    )
    def test_malformed_or_empty_fold_csv_raises_value_error(
        self, tmp_path: Path, csv_content: str
    ) -> None:
        bad_csv = tmp_path / "bad_fold.csv"
        bad_csv.write_text(csv_content, encoding="utf-8")
        with pytest.raises(ValueError):
            load_fold_oof_csv(bad_csv)

    def test_resume_checkpoint_rejects_fold_mismatch_and_single_class_fold(
        self, tmp_path: Path
    ) -> None:
        fold0_rows = _make_two_class_rows(fold=0, n_each=3)
        fold0_csv = write_fold_oof_csv(fold0_rows, tmp_path / "cv_fold_0_oof.csv")
        with pytest.raises(ValueError, match="mismatched fold index"):
            load_and_verify_fold_checkpoint(fold0_csv, expected_fold=2)

        single_class_rows = [_make_clean_oof_row(fold=1, idx=i, label=0) for i in range(4)]
        sc_csv = write_fold_oof_csv(single_class_rows, tmp_path / "cv_fold_1_oof.csv")
        with pytest.raises(ValueError, match="single-class"):
            load_and_verify_fold_checkpoint(sc_csv, expected_fold=1)

    @pytest.mark.parametrize("only_label", [0, 1])
    def test_single_class_fold_raises_value_error_in_evaluation(self, only_label: int) -> None:
        rows = [
            _make_clean_oof_row(fold=k, idx=i, label=only_label, in_bal=1, risk_score=70.0 if only_label else 20.0)
            for k in range(N_OUTER_FOLDS)
            for i in range(4)
        ]
        with pytest.raises(ValueError, match="both classes"):
            evaluate_rows_4tiers(rows)
        with pytest.raises(ValueError, match="both classes"):
            compute_pooled_significance(rows)
        with pytest.raises(ValueError, match="both classes"):
            aggregate_cv_results(rows)
        with pytest.raises(ValueError, match="single-class"):
            validate_binary_split_labels([only_label, only_label], "single_fold")

    @pytest.mark.parametrize(
        ("field", "bad_value"),
        [
            ("label", True),
            ("label", np.bool_(False)),
            ("label", 2),
            ("p_cnn", True),
            ("p_cnn", float("nan")),
            ("p_crisp", float("inf")),
            ("risk_score", -1.0),
            ("risk_score", float("nan")),
        ],
    )
    def test_row_evaluation_rejects_bool_nan_inf_and_oob_values(
        self, field: str, bad_value: Any
    ) -> None:
        rows = _make_two_class_rows(fold=0, n_each=3)
        rows[0][field] = bad_value
        with pytest.raises(ValueError):
            evaluate_rows_4tiers(rows)

    @pytest.mark.parametrize("bad_val", [True, np.bool_(False), float("nan"), float("inf"), float("-inf")])
    def test_fold_mean_sd_rejects_bool_nan_inf(self, bad_val: Any) -> None:
        valid_metrics = {
            tier: {m: 0.8 for m in ("accuracy", "sensitivity", "specificity", "precision", "f1_score", "roc_auc", "pr_auc")}
            for tier in CV_TIERS
        }
        bad_metrics = {
            tier: {m: (bad_val if (tier == "tier3_standard" and m == "roc_auc") else 0.8) for m in valid_metrics[tier]}
            for tier in CV_TIERS
        }
        with pytest.raises(ValueError):
            compute_fold_mean_sd([valid_metrics, bad_metrics])
        with pytest.raises(ValueError, match="at least 2 folds"):
            compute_fold_mean_sd([valid_metrics])

    @pytest.mark.parametrize(
        ("field", "bad_value"),
        [
            ("fold", True),
            ("fold", np.bool_(True)),
            ("in_balanced_test", True),
            ("in_balanced_test", np.bool_(False)),
        ],
    )
    def test_aggregate_cv_results_rejects_bool_fold_or_flag(
        self, field: str, bad_value: Any
    ) -> None:
        rows = _build_selftest_oof_rows(np.random.default_rng(11))
        rows[0][field] = bad_value
        with pytest.raises(ValueError):
            aggregate_cv_results(rows)

    def test_run_5fold_cv_cli_selftest_exits_zero_without_tf(self) -> None:
        proc = subprocess.run(
            [sys.executable, "services/colab_trainer/run_5fold_cv.py", "--selftest"],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, f"5-fold selftest failed: {proc.stderr or proc.stdout}"
        assert "SELFTEST PASSED" in proc.stdout


class TestPlotQ1FiguresOffline:
    """Offline tests for plot_q1_figures.py using real fis_calibration.json."""

    def test_valid_study_artifacts_render_all_four_300dpi_figures(self, tmp_path: Path) -> None:
        cfg = _make_complete_plot_inputs(tmp_path)
        outputs = render_all_q1_figures(cfg)

        assert set(p.name for p in outputs.values()) == set(FIGURE_FILENAMES)
        for fig_path in outputs.values():
            assert fig_path.is_file() and fig_path.stat().st_size > 15000
            img = cv2.imread(str(fig_path))
            assert img is not None and img.ndim == 3 and img.shape[2] == 3
            assert img.shape[0] >= 1000 and img.shape[1] >= 2000
            assert float(np.std(img)) > 15.0

    @pytest.mark.parametrize("missing_target", ["cases", "stats", "calibration", "cv_summary"])
    def test_missing_input_file_raises_file_not_found(
        self, tmp_path: Path, missing_target: str
    ) -> None:
        base = _make_complete_plot_inputs(tmp_path)
        kwargs: dict[str, Any] = {
            "cases_csv": base.cases_csv,
            "stats_json": base.stats_json,
            "calibration_json": base.calibration_json,
            "out_dir": tmp_path / "out",
            "cv_summary_json": None,
        }
        if missing_target == "cases":
            kwargs["cases_csv"] = tmp_path / "nonexistent_cases.csv"
        elif missing_target == "stats":
            kwargs["stats_json"] = tmp_path / "nonexistent_stats.json"
        elif missing_target == "calibration":
            kwargs["calibration_json"] = tmp_path / "nonexistent_calib.json"
        elif missing_target == "cv_summary":
            kwargs["cv_summary_json"] = tmp_path / "nonexistent_cv.json"

        with pytest.raises(FileNotFoundError):
            render_all_q1_figures(PlotInputsConfig(**kwargs))

    @pytest.mark.parametrize(
        "corruption_kind",
        [
            "empty_csv",
            "header_only_csv",
            "single_class_clean",
            "missing_clean_condition",
            "corrupt_stats_json",
            "bool_dpi",
            "zero_dpi",
        ],
    )
    def test_invalid_or_empty_plot_inputs_raise_value_error(
        self, tmp_path: Path, corruption_kind: str
    ) -> None:
        cfg = _make_complete_plot_inputs(tmp_path)
        header = cfg.cases_csv.read_text(encoding="utf-8").splitlines()[0]

        if corruption_kind == "empty_csv":
            cfg.cases_csv.write_text("", encoding="utf-8")
        elif corruption_kind == "header_only_csv":
            cfg.cases_csv.write_text(header + "\n", encoding="utf-8")
        elif corruption_kind == "single_class_clean":
            lines = cfg.cases_csv.read_text(encoding="utf-8").splitlines()
            filtered = [lines[0]] + [ln for ln in lines[1:] if ",0,clean," in ln or ",clean," not in ln]
            cfg.cases_csv.write_text("\n".join(filtered) + "\n", encoding="utf-8")
        elif corruption_kind == "missing_clean_condition":
            lines = cfg.cases_csv.read_text(encoding="utf-8").splitlines()
            filtered = [lines[0]] + [ln for ln in lines[1:] if ",clean," not in ln]
            cfg.cases_csv.write_text("\n".join(filtered) + "\n", encoding="utf-8")
        elif corruption_kind == "corrupt_stats_json":
            cfg.stats_json.write_text(json.dumps({"metadata": {}}), encoding="utf-8")
        elif corruption_kind == "bool_dpi":
            cfg = PlotInputsConfig(
                cases_csv=cfg.cases_csv,
                stats_json=cfg.stats_json,
                calibration_json=cfg.calibration_json,
                out_dir=cfg.out_dir,
                dpi=True,  # type: ignore[arg-type]
            )
        elif corruption_kind == "zero_dpi":
            cfg = PlotInputsConfig(
                cases_csv=cfg.cases_csv,
                stats_json=cfg.stats_json,
                calibration_json=cfg.calibration_json,
                out_dir=cfg.out_dir,
                dpi=0,
            )

        with pytest.raises(ValueError):
            render_all_q1_figures(cfg)

    def test_rendering_closes_figures_with_zero_pyplot_handle_leaks(self, tmp_path: Path) -> None:
        import matplotlib.pyplot as plt

        cfg = _make_complete_plot_inputs(tmp_path)
        initial_fignums = list(plt.get_fignums())
        for _ in range(2):
            render_all_q1_figures(cfg)
            assert list(plt.get_fignums()) == initial_fignums

    def test_plot_cli_selftest_and_clean_import_have_no_heavy_module_leaks(self) -> None:
        proc = subprocess.run(
            [sys.executable, "services/colab_trainer/plot_q1_figures.py", "--selftest"],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, f"plot selftest failed: {proc.stderr or proc.stdout}"
        assert "SELFTEST PASSED" in proc.stdout

        leak_check = (
            "import sys\n"
            "import services.colab_trainer.run_5fold_cv\n"
            "import services.colab_trainer.plot_q1_figures\n"
            "assert 'tensorflow' not in sys.modules\n"
            "assert 'torch' not in sys.modules\n"
            "assert 'matplotlib.pyplot' not in sys.modules\n"
        )
        proc_leak = subprocess.run([sys.executable, "-c", leak_check], capture_output=True, text=True)
        assert proc_leak.returncode == 0, f"Import leakage detected: {proc_leak.stderr or proc_leak.stdout}"
