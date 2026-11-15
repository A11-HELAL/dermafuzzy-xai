# Implementation Plan: Clinical Evaluation, 3-Tier Ablation Study, 4-Panel XAI Visualizer & Colab MobileNetV2 Pipeline

**Branch**: `004-ablation-visualizer-colab-ci` | **Date**: 2026-11-15 | **Spec**: [spec.md](spec.md)  
**Input**: Feature specification from `/specs/004-ablation-visualizer-colab-ci/spec.md`  

---

## Summary

Implement the final architectural room (Room 5: Evaluation & Explainability Visualization) and CI automation for DermaFuzzy-XAI. This comprises:
1. `packages/dermafuzzy_core/evaluation.py`: Pure NumPy clinical evaluation metrics (Sensitivity, Specificity, Precision, Accuracy, F1-Score, and exact tied-score trapezoidal AUC-ROC) and the 3-tier Ablation Study engine (`Pure MobileNetV2 Black-Box` vs. `Crisp ABCD Fixed Thresholds` vs. `Proposed DermaFuzzy-XAI`).
2. `packages/dermafuzzy_core/visualizer.py`: Standardized publication-grade (300 DPI) 4-panel clinical XAI audit figures rendered via headless Matplotlib (`Figure` + `FigureCanvasAgg`) with lazy loading to preserve sub-second package startup.
3. `services/colab_trainer/train_ham10000.py`: End-to-end Google Colab training and fine-tuning pipeline for balanced HAM10000 ($n=2,000$) with a `--dry-run-synthetic` flag for instant local verification.
4. `.github/workflows/ci.yml`: Automated GitHub Actions CI workflow running linting, Pure Core zero-ML import verification, and the complete unit + metamorphic test suite on every push.

---

## Technical Context

**Language/Version**: Python 3.14.4 (locally) / Python 3.12+ (in CI and Colab)  
**Primary Dependencies**: `numpy>=1.24.0`, `opencv-python-headless>=4.8.0`, `scipy>=1.11.0`, `scikit-fuzzy>=0.5.0`, `matplotlib>=3.8.0`, `pyyaml>=6.0`  
**Storage**: JSON/CSV/Markdown file artifacts (`artifacts/ablation_report.md`, `artifacts/ablation_metrics.csv`, `artifacts/sample_4panel_xai.png`, `artifacts/fis_calibration.json`). Zero `pickle`.  
**Testing**: `pytest>=9.0.0` with unit tests in `tests/unit/test_evaluation.py` and `tests/unit/test_visualizer.py`.  
**Target Platform**: Clinical edge workstation (Intel Core i5-7200U @ 2.50GHz, 8GB RAM) for core inference/visualization; Google Colab (NVIDIA T4/V100 GPU) for deep training.  
**Project Type**: Algorithmic library (`packages/dermafuzzy_core`), training service (`services/colab_trainer`), and CI automation (`.github/workflows`).  
**Performance Goals**: 
- AUC-ROC pure NumPy calculation in $< 5\text{ms}$ on $N=1,000$ cohort.
- 4-Panel report generation in $< 1.5\text{s}$ at 300 DPI.
- Colab trainer `--dry-run-synthetic` smoke-test in $< 10\text{s}$.  
**Constraints**: 
- Constitution Principle I: Zero top-level heavy ML framework imports (`tensorflow`, `keras`, `torch`) in `packages/dermafuzzy_core/`.
- Lazy import of `matplotlib` inside visualizer functions to maintain instantaneous `< 50ms` package import.
- Deep array immutability on visualizer outputs (`arr.flags.writeable = False`).  

---

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Requirement | Plan Conformance | Status |
|---|---|---|---|
| **I. Pure Core & Zero Skew** | Room 5 evaluation & visualizer logic resides in `packages/dermafuzzy_core/`; zero TensorFlow/Keras imports; fast edge execution. | `evaluation.py` and `visualizer.py` use only NumPy, OpenCV, and Matplotlib (`Agg`). TensorFlow is isolated exclusively to `services/colab_trainer/`. | **PASS** |
| **II. Strict Mathematical Bounds** | All metrics normalized to $[0.0, 1.0]$; epsilon stabilizers on all denominators; read-only array returns. | Denominators protected with $\epsilon = 1e-6$ or explicit zero-count handling; visualizer returns read-only uint8 array. | **PASS** |
| **III. Zero Leakage & No-Pickle** | Zero pickle serialization; metrics and ablation reports exported via JSON, Markdown, and CSV. | Pure JSON/CSV/Markdown formats; zero `.pkl` / `.pickle` files. | **PASS** |
| **IV. Config-Driven & Test-First** | Zero magic numbers; parameters loaded from `derma_config.yaml`; comprehensive automated test suite. | Thresholds and class targets loaded from config; unit tests cover metrics, tied AUC, visualizer memory safety, and ablation study. | **PASS** |

---

## Project Structure

### Documentation (this feature)

```text
specs/004-ablation-visualizer-colab-ci/
├── spec.md                  # Feature specification
├── plan.md                  # This implementation plan
├── research.md              # Phase 0: Pure NumPy AUC-ROC, Ablation & Headless Matplotlib
├── data-model.md            # Phase 1: ClinicalMetrics, AblationReport, FourPanelReportConfig
├── quickstart.md            # Phase 1: Runnable verification scenarios
├── checklists/
│   └── requirements.md      # Spec quality validation checklist
└── contracts/
    ├── evaluation-api.md     # Room 5 evaluation API contract
    └── visualizer-api.md     # Room 5 visualizer API contract
```

### Source Code (repository root)

```text
packages/dermafuzzy_core/
├── __init__.py              # Re-export Room 1 through Room 5 public symbols
├── config.py                # Room 1: Unified configuration engine
├── preprocessing.py         # Room 2: OpenCV hair removal, CLAHE, segmentation
├── biomarkers.py            # Room 3: ABCD clinical biomarker formulations
├── gradcam.py               # Room 3: Saliency-gated attention verification
├── fis_engine.py            # Room 4: Data-calibrated Mamdani FIS & explanation
├── evaluation.py            # Room 5: Clinical metrics & 3-tier ablation study
└── visualizer.py            # Room 5: 4-Panel publication-grade XAI report generator

services/
└── colab_trainer/
    └── train_ham10000.py    # Google Colab MobileNetV2 training & evaluation pipeline

.github/
└── workflows/
    └── ci.yml               # Automated GitHub Actions CI workflow

artifacts/
├── fis_calibration.json     # Calibrated training percentiles
├── ablation_report.md       # Generated 3-tier ablation markdown table
├── ablation_metrics.csv     # Generated ablation metrics CSV
└── sample_4panel_xai.png    # Sample rendered 4-panel clinical explanation

tests/
├── unit/
│   ├── test_config.py       # Room 1 tests
│   ├── test_preprocessing.py # Room 2 tests
│   ├── test_gradcam.py      # Room 3 Grad-CAM tests
│   ├── test_biomarkers.py   # Room 3 ABCD biomarker tests
│   ├── test_fis_engine.py   # Room 4 Mamdani FIS tests
│   ├── test_evaluation.py   # Room 5 Evaluation & ablation tests
│   └── test_visualizer.py   # Room 5 4-Panel visualizer tests
└── metamorphic/
    └── test_metamorphic.py  # Room 6 End-to-end Metamorphic Relations (MR1–MR5)
```

**Structure Decision**:
Completes the full 6-room modular architectural layout. Room 5 logic is implemented in `evaluation.py` and `visualizer.py` within `packages/dermafuzzy_core/`. Deep training resides in `services/colab_trainer/` decoupled from the edge inference core. CI automation is located in `.github/workflows/ci.yml`.

---

## Complexity Tracking

> **No Constitution violations exist. All gates pass cleanly.**

| Checkpoint | Status | Rationale |
|---|---|---|
| Pure Core Dependency Guard | Compliant | Matplotlib is imported lazily inside `visualizer.py` functions using `FigureCanvasAgg`. Zero TensorFlow imports in core. |
| AUC-ROC Independence | Compliant | Exact tied-score trapezoidal integration is implemented in pure NumPy without importing Scikit-learn. |
| Memory Management | Compliant | Headless Matplotlib figures are explicitly cleared and closed after buffer extraction. |
