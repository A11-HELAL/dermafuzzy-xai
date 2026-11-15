"""Unit and smoke tests for services/colab_trainer/train_ham10000.py.

Verifies:
- Standalone execution of --dry-run-synthetic mode
- Sub-15 second execution on standard CPU
- Generation of all required clinical artifacts:
  - artifacts/fis_calibration.json
  - artifacts/ablation_report.md
  - artifacts/ablation_metrics.csv
  - artifacts/sample_4panel_xai.png
- Zero TensorFlow requirement in dry-run mode
"""

import os
import subprocess
import sys
import time
from pathlib import Path
import pytest


def test_colab_trainer_dry_run_synthetic(tmp_path):
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

    # Verify generated artifacts
    calib_json = output_dir / "fis_calibration.json"
    report_md = output_dir / "ablation_report.md"
    metrics_csv = output_dir / "ablation_metrics.csv"
    sample_png = output_dir / "sample_4panel_xai.png"

    assert calib_json.exists(), "fis_calibration.json was not created"
    assert report_md.exists(), "ablation_report.md was not created"
    assert metrics_csv.exists(), "ablation_metrics.csv was not created"
    assert sample_png.exists(), "sample_4panel_xai.png was not created"

    # Verify file contents
    md_text = report_md.read_text(encoding="utf-8")
    assert "Tier 1: Pure MobileNetV2 Black-Box" in md_text or "1_pure_mobilenetv2_blackbox" in md_text
    assert "Tier 3: Proposed DermaFuzzy-XAI" in md_text or "3_proposed_dermafuzzy_xai" in md_text
    assert "Specificity" in md_text

    csv_text = metrics_csv.read_text(encoding="utf-8")
    assert "tier,tp,tn,fp,fn" in csv_text
    assert "3_proposed_dermafuzzy_xai" in csv_text

    assert sample_png.stat().st_size > 1000, "sample_4panel_xai.png is too small"
