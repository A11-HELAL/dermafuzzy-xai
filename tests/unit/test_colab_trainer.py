"""Unit and smoke tests for services/colab_trainer/train_ham10000.py.

Verifies:
- Direct execution of run_synthetic_dry_run with artifact verification
- Strict JSON schema, monotonicity, and complete feature coverage in fis_calibration.json
- Valid 3-channel image rendering in sample_4panel_xai.png
- Correct markdown table and CSV formats for ablation study
- Standalone CLI execution of --dry-run-synthetic mode
- Sub-15 second execution on standard CPU
- CLI argument handling and unknown flag rejection (SystemExit with code 2)
- Graceful fallback when HAM10000 dataset paths are not present
- Zero TensorFlow / PyTorch leakage into sys.modules during dry-run execution
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path
import cv2
import numpy as np
import pytest

from dermafuzzy_core.config import CANONICAL_FEATURE_NAMES, load_config
from services.colab_trainer.train_ham10000 import (
    main,
    run_full_colab_pipeline,
    run_synthetic_dry_run,
)


class TestColabTrainerDirect:
    """Direct Python unit tests for colab trainer functions (without subprocess overhead)."""

    def test_run_synthetic_dry_run_artifacts(self, tmp_path: Path):
        """Verify run_synthetic_dry_run generates all required artifacts with valid contents."""
        config = load_config()
        output_dir = tmp_path / "artifacts"

        run_synthetic_dry_run(output_dir, config)

        calib_json = output_dir / "fis_calibration.json"
        report_md = output_dir / "ablation_report.md"
        metrics_csv = output_dir / "ablation_metrics.csv"
        sample_png = output_dir / "sample_4panel_xai.png"

        assert calib_json.exists(), "fis_calibration.json missing"
        assert report_md.exists(), "ablation_report.md missing"
        assert metrics_csv.exists(), "ablation_metrics.csv missing"
        assert sample_png.exists(), "sample_4panel_xai.png missing"

        # 1. Validate calibration JSON schema and strict monotonicity
        with open(calib_json, "r", encoding="utf-8") as f:
            calib_data = json.load(f)

        assert "cutoffs" in calib_data
        assert "n_samples" in calib_data
        assert calib_data["n_samples"] == 24
        assert "config_hash" in calib_data

        cutoffs = calib_data["cutoffs"]
        for feat in ("A", "B_fuzzy", "C", "D_verified"):
            assert feat in cutoffs, f"Feature {feat} missing from calibration JSON"
            p25, p50, p75 = cutoffs[feat]
            assert 0.0 < p25 < p50 < p75 < 1.0, f"Monotonicity violated for {feat}: {p25}, {p50}, {p75}"
            assert (p50 - p25) >= 1e-4 - 1e-9
            assert (p75 - p50) >= 1e-4 - 1e-9

        # 2. Validate ablation markdown report
        md_text = report_md.read_text(encoding="utf-8")
        assert "| Architecture Tier | Accuracy | Sensitivity | Specificity |" in md_text
        assert "Tier 1: Pure MobileNetV2 Black-Box" in md_text
        assert "Tier 2: Crisp ABCD Fixed Thresholds" in md_text
        assert "Tier 3: Proposed DermaFuzzy-XAI" in md_text

        # 3. Validate ablation CSV
        csv_text = metrics_csv.read_text(encoding="utf-8")
        lines = [line.strip() for line in csv_text.strip().splitlines() if line.strip()]
        assert len(lines) == 4  # Header + 3 tiers
        assert lines[0] == "tier,tp,tn,fp,fn,sensitivity,specificity,precision,accuracy,f1_score,auc_roc,threshold"
        assert "1_pure_mobilenetv2_blackbox" in lines[1]
        assert "2_crisp_abcd_fixed_thresholds" in lines[2]
        assert "3_proposed_dermafuzzy_xai" in lines[3]

        # 4. Validate rendered 4-panel image via OpenCV
        img = cv2.imread(str(sample_png))
        assert img is not None, "Failed to load sample_4panel_xai.png"
        assert img.ndim == 3 and img.shape[2] == 3
        assert img.shape[0] > 100 and img.shape[1] > 400
        # Check non-trivial visual information
        assert np.std(img) > 15.0

    def test_run_full_colab_pipeline_missing_data_fallback(self, tmp_path: Path):
        """Verify run_full_colab_pipeline gracefully falls back to synthetic dry run when dataset files are absent."""
        output_dir = tmp_path / "artifacts"
        args = Namespace(
            data_dir="non_existent_data_dir_12345",
            metadata_csv="non_existent_metadata_12345.csv",
            output_dir=str(output_dir),
            epochs=1,
            batch_size=16,
        )

        run_full_colab_pipeline(args)

        # Fallback must have produced artifacts without crashing
        assert (output_dir / "fis_calibration.json").exists()
        assert (output_dir / "ablation_report.md").exists()
        assert (output_dir / "sample_4panel_xai.png").exists()

    def test_synthetic_dry_run_zero_tf_leak(self, tmp_path: Path):
        """Verify running synthetic dry run does not import tensorflow or torch into sys.modules."""
        config = load_config()
        output_dir = tmp_path / "artifacts"

        run_synthetic_dry_run(output_dir, config)

        assert "tensorflow" not in sys.modules, "TensorFlow leaked during dry-run!"
        assert "torch" not in sys.modules, "PyTorch leaked during dry-run!"


class TestColabTrainerCLI:
    """Subprocess and CLI tests for colab trainer script."""

    def test_colab_trainer_cli_dry_run_synthetic(self, tmp_path: Path):
        """Verify running the script via CLI with --dry-run-synthetic executes under 15s and exits 0."""
        repo_root = Path(__file__).resolve().parents[2]
        script_path = repo_root / "services" / "colab_trainer" / "train_ham10000.py"
        output_dir = tmp_path / "artifacts"

        cmd = [
            sys.executable,
            str(script_path),
            "--dry-run-synthetic",
            "--output-dir",
            str(output_dir),
        ]

        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(repo_root))
        elapsed = time.time() - start_time

        assert result.returncode == 0, f"Script failed with code {result.returncode}:\n{result.stderr}\n{result.stdout}"
        assert elapsed < 15.0, f"Dry run took {elapsed:.2f}s, expected < 15.0s"
        assert (output_dir / "fis_calibration.json").exists()
        assert (output_dir / "ablation_report.md").exists()

    def test_colab_trainer_cli_unknown_arg_exits_2(self):
        """Verify passing an unrecognized argument causes argparse to exit with code 2."""
        repo_root = Path(__file__).resolve().parents[2]
        script_path = repo_root / "services" / "colab_trainer" / "train_ham10000.py"

        cmd = [
            sys.executable,
            str(script_path),
            "--unrecognized-invalid-flag",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(repo_root))
        assert result.returncode == 2, f"Expected exit code 2, got {result.returncode}"
        assert "unrecognized arguments" in result.stderr
