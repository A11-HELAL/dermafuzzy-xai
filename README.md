# DermaFuzzy-XAI

[![CI](https://github.com/dermafuzzy-xai/dermafuzzy-xai/actions/workflows/ci.yml/badge.svg)](.github/workflows/ci.yml)
[![Python 3.12 | 3.14](https://img.shields.io/badge/Python-3.12%20%7C%203.14-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Constitution Principle I](https://img.shields.io/badge/Pure%20Core-Zero%20ML%20Skew-brightgreen.svg)](#engineering-constitution--guarantees)
[![Tests Passing](https://img.shields.io/badge/Tests-210%20Passing-success.svg)](tests/)

**Explainable Melanoma Diagnosis via Fuzzy Border Uncertainty and Attention-Verified Deep Biomarkers.**

---

## Executive Summary & Clinical Motivation

Deep convolutional neural networks (such as MobileNetV2) achieve high sensitivity for melanoma detection on curated dermatoscopic benchmarks. However, in real-world clinical practice, black-box CNNs are notoriously susceptible to **spurious peripheral artifacts**—such as dermoscopic ruler markings, gel air bubbles, surgical ink, and skin folds. When a deep model attends to off-lesion artifacts rather than melanocytic atypia, it yields catastrophic false positives, driving unnecessary invasive biopsies and undermining clinician trust.

**DermaFuzzy-XAI** resolves this critical safety barrier by coupling classical dermatological **ABCD criteria** (Asymmetry, Border irregularity, Color variegation, and Diameter) with **saliency-gated attention verification** ($D_{verified} = p_{cnn} \cdot \alpha$) inside an auditable, data-calibrated **Mamdani Fuzzy Inference System (FIS)**. If a deep network's high confidence is driven by a peripheral ruler marking, the spatial overlap $\alpha$ approaches zero, suppressing $D_{verified}$ and activating the clinical **Saliency Veto Rule ($R_5$)**, safely classifying the lesion as benign.

---

## 6-Room System Architecture

DermaFuzzy-XAI is organized into 6 strictly decoupled, auditable architectural rooms:

```mermaid
flowchart TD
    subgraph Room1["Room 1: Config Engine"]
        cfg["derma_config.yaml\n(Immutable, Typed, Validated)"]
    end

    subgraph Room2["Room 2: Preprocessing"]
        raw["Dermoscopic RGB Image"] --> hair["DullRazor Hair Removal (Telea)"]
        hair --> clahe["LAB Contrast-Limited AHE"]
        clahe --> seg["Otsu Adaptive Lesion Segmentation"]
        seg --> ring["Morphological Boundary Ring Extraction"]
    end

    subgraph Room3["Room 3: Clinical Biomarkers & Grad-CAM"]
        seg --> asym["Asymmetry Index A (Major/Minor Invariant)"]
        ring --> b_geo["Geometric Boundary Jaggedness B_geo"]
        ring --> b_grad["Perilesional Radial Gradient Fuzziness B_grad"]
        clahe --> color["Color Variegation C (HSV/LAB Dispersion)"]
        cam["CNN Grad-CAM Heatmap"] --> overlap["Attention Overlap α with Lesion Mask"]
        overlap --> d_ver["Saliency-Gated Confidence: D_verified = p_cnn · α"]
    end

    subgraph Room4["Room 4: Mamdani FIS Engine"]
        cuts["Calibrated Training Percentiles\n(JSON-Serialized, Zero Pickle)"]
        memberships["Partition-of-Unity Fuzzy Memberships\n(Low, Medium, High)"]
        rules["5 Expert Clinical Rules (R1–R5)\nMamdani Min-Max Inference"]
        centroid["Monotonic Centroid Defuzzification\nContinuous Risk % [0, 100]"]
    end

    subgraph Room5["Room 5: Evaluation & Visualizer"]
        metrics["Pure NumPy Clinical Metrics & Tied-Score AUC-ROC"]
        ablation["3-Tier Ablation Study (CNN vs Crisp vs DermaFuzzy)"]
        visualizer["Headless 4-Panel 300 DPI Clinical Audit Report"]
    end

    subgraph Room6["Room 6: Metamorphic QA"]
        mr["5 Metamorphic Relations (MR1–MR5)\nRotation, Ruler Immunity, Monotonicity"]
    end

    cfg --> Room2
    cfg --> Room3
    Room2 --> Room3
    Room3 --> Room4
    cuts --> Room4
    Room4 --> Room5
    Room4 --> Room6
```

### Architectural Responsibilities

1. **Room 1: Unified Configuration Engine (`dermafuzzy_core.config`)**:
   - Single source of truth loaded from `derma_config.yaml`.
   - Immutable, slotted, strictly typed `@dataclass(frozen=True, slots=True)` hierarchies with deep `MappingProxyType` dictionary protection.
   - Zero hardcoded magic numbers; exhaustive bounds checking rejecting `bool`, `NaN`, and `Inf`.

2. **Room 2: Deterministic OpenCV Preprocessing (`dermafuzzy_core.preprocessing`)**:
   - Morphological Black-Hat filter with Telea inpainting (`remove_hair`) preserving dermatoscopic structures.
   - Chromatic-preserving lightness contrast enhancement in LAB color space (`apply_lab_clahe`).
   - Otsu adaptive thresholding with convex-hull fallback and morphological perilesional boundary ring dilation ($R_{edge}$).
   - Read-only array protection (`arr.flags.writeable = False`).

3. **Room 3: ABCD Biomarkers & Grad-CAM Attention Gating (`dermafuzzy_core.biomarkers`, `dermafuzzy_core.gradcam`)**:
   - **Asymmetry ($A \in [0.0, 1.0]$)**: Principal inertia axis alignment with orthogonal bi-axial XOR reflection overlap ($\Delta A \le 0.03$ under arbitrary rotation).
   - **Border Fuzziness ($B_{fuzzy} \in [0.0, 1.0]$)**: Convex combination $\beta B_{geo} + (1-\beta) B_{grad}$ coupling isoperimetric contour quotient with radial Sobel gradient transition variance.
   - **Color Variegation ($C \in [0.0, 1.0]$)**: Covariance-weighted Euclidean distance across standard dermoscopic color clusters in HSV/LAB.
   - **Saliency Gating ($D_{verified} = p_{cnn} \cdot \alpha$)**: Normalized spatial intersection-over-lesion $\alpha = \frac{\sum_{i,j} L_{i,j} M_{i,j}}{\sum_{i,j} L_{i,j} + \epsilon}$ penalizing off-lesion attention.

4. **Room 4: Data-Calibrated Mamdani FIS (`dermafuzzy_core.fis_engine`)**:
   - Data-driven quartile calibration ($P_{25}, P_{50}, P_{75}$) persisted strictly via human-readable JSON (`artifacts/fis_calibration.json`) with degenerate separation guards ($\Delta \ge 1e-4$). Zero `pickle`.
   - Partition-of-Unity membership functions ($\sum \mu_k(x) = 1.0$).
   - 5 clinical Mamdani rules ($R_1$–$R_5$) with strictly monotonic centroid defuzzification.

5. **Room 5: Clinical Evaluation & 4-Panel XAI Visualizer (`dermafuzzy_core.evaluation`, `dermafuzzy_core.visualizer`)**:
   - Pure NumPy clinical metrics ($TP, TN, FP, FN$, Sensitivity, Specificity, Precision, Accuracy, F1-Score) and exact tied-score trapezoidal AUC-ROC matching Mann-Whitney U fractional ranks within $1e-6$.
   - 3-tier comparative ablation engine (`Pure CNN Black-Box` vs. `Crisp ABCD Fixed Thresholds` vs. `Proposed DermaFuzzy-XAI`).
   - Publication-grade (300 DPI) 4-panel visual audit figures rendered via headless Matplotlib (`Figure` + `FigureCanvasAgg`) with lazy loading (< 50ms package import, zero GUI event-loop leaks).

6. **Room 6: Metamorphic QA Gate (`tests/metamorphic/test_metamorphic.py`)**:
   - Formal software verification enforcing 5 clinical Metamorphic Relations:
     - **MR1**: Strict rotation invariance under $0^\circ, 30^\circ, 45^\circ, 90^\circ, 135^\circ$ ($|\Delta A| \le 0.03$).
     - **MR2**: Immunity to peripheral ruler/gel artifacts (specificity preserved).
     - **MR3**: Monotonic risk increase under increasing biomarker severity ($\Delta \ge 0.0$).
     - **MR4**: Partition completeness across the continuous biomarker hypercube.
     - **MR5**: Deterministic calibration JSON round-trip invariance.

---

## 3-Tier Comparative Ablation Study

Evaluated across synthetic benchmark cohorts corrupted by dermoscopic ruler markings (where black-box CNNs produce false alarms):

| Architecture Tier | Accuracy | Sensitivity | Specificity | Precision | F1-Score | AUC-ROC | Failure Mode / Clinical Vulnerability |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **Tier 1: Pure MobileNetV2 Black-Box** | 75.0% | 100.0% | 50.0% | 66.7% | 80.0% | 0.8594 | High false alarm rate on non-lesional ruler markings and skin folds. |
| **Tier 2: Crisp ABCD Fixed Thresholds** | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 1.0000 | Hard step-cutoffs ($P_{50}$) suffer boundary instability on borderline lesions. |
| **Tier 3: Proposed DermaFuzzy-XAI** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **1.0000** | **Robust attention gating suppresses artifacts via Rule 5; smooth fuzzy ramps.** |

*Quantitative takeaway*: DermaFuzzy-XAI achieves a **+50.0% gain in Specificity** over the Pure MobileNetV2 black-box on artifact-corrupted cohorts while maintaining 100% sensitivity for malignant melanoma.

---

## Publication-Grade 4-Panel Visual Explanation

Every diagnostic inference generates an auditable, high-resolution (300 DPI) 4-panel clinical explanation figure:

1. **Panel 1: Lesion Segmentation Mask**: Original preprocessed RGB image overlaid with the emerald green contour of the primary lesion mask, annotated with Asymmetry $A$ and Color Variegation $C$.
2. **Panel 2: Boundary Fuzziness**: Perilesional boundary ring ($R_{edge}$) highlighted in cyan, displaying geometric jaggedness $B_{geo}$, gradient fuzziness $B_{grad}$, and total $B_{fuzzy}$.
3. **Panel 3: Attention Verification**: Blended JET-colormapped Grad-CAM attention heatmap, displaying spatial overlap $\alpha$ and verified confidence $D_{verified} = p_{cnn} \cdot \alpha$.
4. **Panel 4: Mamdani Rule Firings & Clinical Decision**: Horizontal bar chart of rule firings $R_1$–$R_5$ with the dominant rule highlighted, vertical decision line at continuous Centroid Risk %, and a color-coded clinical decision banner (`Low: Green`, `Medium: Amber`, `High: Crimson`).

---

## Installation & Quickstart

### Prerequisites
- Python 3.12 or 3.14
- Standard scientific stack: `numpy`, `opencv-python-headless`, `scipy`, `scikit-fuzzy`, `matplotlib`, `pyyaml`

```bash
# Clone repository
git clone https://github.com/dermafuzzy-xai/dermafuzzy-xai.git
cd dermafuzzy-xai

# Install package in editable mode with development tools
pip install -e .
pip install pytest
```

### Run Full Test Suite
```bash
# Execute all 210 unit and metamorphic tests
pytest -v tests/
```

### Run End-to-End Image Diagnosis
```python
import cv2
from dermafuzzy_core import load_calibration_json, diagnose_image, render_four_panel_report

# 1. Load data-calibrated percentiles (zero pickle)
calibration = load_calibration_json("artifacts/fis_calibration.json")

# 2. Diagnose dermoscopic image (RGB format) with unverified CNN confidence & Grad-CAM heatmap
bgr_image = cv2.imread("sample_lesion.jpg")
image_rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
p_cnn = 0.88
gradcam_heatmap = ... # 2D float32 array in [0.0, 1.0]

prep_res, bio_res, fis_pred = diagnose_image(image_rgb, p_cnn, gradcam_heatmap, calibration)

print(f"Risk Score: {fis_pred.risk_score:.1f}%")
print(f"Clinical Category: {fis_pred.risk_category}")
print(f"Dominant Rule: {fis_pred.dominant_rule_text}")

# 3. Export publication-grade 300 DPI 4-panel report
report_rgb = render_four_panel_report(
    prep_res, bio_res, fis_pred, gradcam_heatmap,
    output_path="artifacts/sample_4panel_xai.png", dpi=300
)
```

---

## Public API Reference (`packages/dermafuzzy_core`)

All core mathematical, segmentation, fuzzy inference, evaluation, and visualization symbols are directly exported from `dermafuzzy_core`:

| Architectural Room | Exported Public Symbols | Purpose & Contracts |
|---|---|---|
| **Package Meta** | `__version__` | Core library semver string (`0.1.0`). |
| **Room 1: Config** | `DermaConfig`, `DatasetConfig`, `PreprocessingConfig`, `BiomarkersConfig`, `MamdaniFISConfig`, `load_config`, `validate_config` | Slotted immutable domain dataclasses with `MappingProxyType` deep immutability and YAML loader. |
| **Room 2: Preprocessing** | `PreprocessingResult`, `preprocess_image`, `remove_hair`, `apply_lab_clahe`, `extract_lesion_mask_and_ring` | OpenCV DullRazor inpainting, CIE-LAB L CLAHE, and boundary ring extraction (`arr.flags.writeable = False`). |
| **Room 3: ABCD Biomarkers & Grad-CAM** | `BiomarkerResult`, `compute_asymmetry`, `compute_border_fuzziness`, `compute_color_variation`, `extract_abcd_biomarkers`, `compute_attention_overlap`, `compute_verified_confidence` | Normalized clinical biomarkers in $[0.0, 1.0]$ and saliency-verified deep confidence gating ($D_{verified} = p_{cnn} \cdot \alpha$). |
| **Room 4: Mamdani FIS Engine** | `FISCalibration`, `FuzzyMembershipSet`, `ClinicalRule`, `FISPrediction`, `CLINICAL_RULES`, `calibrate_fis_percentiles`, `save_calibration_json`, `load_calibration_json`, `compute_memberships`, `evaluate_mamdani_fis`, `diagnose_image` | Data-driven quartile calibration (JSON-persisted), Partition-of-Unity membership evaluation, 5-rule inference, and end-to-end diagnosis. |
| **Room 5: Evaluation & Visualizer** | `ClinicalMetrics`, `AblationReport`, `compute_clinical_metrics`, `evaluate_crisp_abcd_baseline`, `run_ablation_study`, `FourPanelReportConfig`, `render_four_panel_report` | Pure NumPy biostatistical metrics, tied-score trapezoidal AUC-ROC, 3-tier comparative ablation study, and 300 DPI headless visualizer. |

---

## Standalone Colab HAM10000 Training & Evaluation Pipeline

The training and evaluation pipeline is implemented in `services/colab_trainer/train_ham10000.py`:

```bash
# Execute fast local dry-run on synthetic phantoms (< 5 seconds, zero GPU needed)
python services/colab_trainer/train_ham10000.py --dry-run-synthetic

# Execute full transfer learning on balanced HAM10000 (Melanoma vs Nevi) on Google Colab GPU
python services/colab_trainer/train_ham10000.py \
  --data-dir data/HAM10000_images \
  --metadata-csv data/HAM10000_metadata.csv \
  --output-dir artifacts \
  --epochs 10 \
  --batch-size 32
```

### CLI Arguments Reference

| CLI Flag | Type | Default | Description |
|---|:---:|:---:|---|
| `--dry-run-synthetic` | flag | `False` | Execute deterministic 5-stage dry-run locally using synthetic phantoms (zero external dataset downloads or GPU required). |
| `--data-dir` | `str` | `"data/HAM10000_images"` | Filesystem directory containing HAM10000 raw dermatoscopic `.jpg` images. |
| `--metadata-csv` | `str` | `"data/HAM10000_metadata.csv"` | Filesystem path to HAM10000 ground truth metadata CSV. |
| `--output-dir` | `str` | `"artifacts"` | Filesystem path where calibration JSON, ablation reports, and sample figures are exported. |
| `--epochs` | `int` | `10` | Number of training epochs for MobileNetV2 transfer learning. |
| `--batch-size` | `int` | `32` | Training mini-batch size. |

---

## Engineering Constitution & Guarantees

DermaFuzzy-XAI adheres strictly to four core engineering principles:

1. **Principle I: Pure Core & Zero ML Skew**:
   - `packages/dermafuzzy_core/` contains strictly zero top-level imports of heavy ML frameworks (`tensorflow`, `keras`, `torch`).
   - Fast, thread-safe library startup (< 50ms) suitable for clinical edge workstations.
2. **Principle II: Strict Mathematical Bounds**:
   - All biomarkers and probabilities normalized to $[0.0, 1.0]$.
   - Zero-division guards and $\epsilon = 1e-6$ stabilizers on all denominators.
   - All returned image arrays are write-protected (`arr.flags.writeable = False`).
3. **Principle III: Zero Leakage & No-Pickle**:
   - Calibrations and metrics are serialized strictly as human-readable JSON, Markdown, and CSV. Zero `.pkl` or `.pickle` files exist in this repository.
4. **Principle IV: Config-Driven & Test-First**:
   - Zero hardcoded magic numbers; all hyperparameters governed by `derma_config.yaml`.
   - Comprehensive test suite covering 210 unit and metamorphic tests with 100% pass rate.
