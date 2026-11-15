# Research & Architectural Design: Clinical Evaluation, 3-Tier Ablation Study, 4-Panel XAI Visualizer & Colab MobileNetV2 Pipeline

**Feature**: [004-ablation-visualizer-colab-ci](spec.md)  
**Date**: 2026-11-15  
**Status**: Completed  

---

## 1. Problem Formulation & Scope

This specification completes the final room (Room 5: Evaluation & Explainability Visualization) of the DermaFuzzy-XAI architecture and establishes the cloud training service and continuous integration pipelines:
1. `packages/dermafuzzy_core/evaluation.py`: Quantitative clinical validation metrics and the 3-tier ablation comparator.
2. `packages/dermafuzzy_core/visualizer.py`: Standardized publication-grade 4-panel clinical XAI audit figures.
3. `services/colab_trainer/train_ham10000.py`: End-to-end cloud training pipeline for HAM10000 balanced cohort ($n=2,000$).
4. `.github/workflows/ci.yml`: Automated CI test and constitution verification gate.

---

## 2. Research Decision 1: Pure NumPy AUC-ROC with Tied-Score Resolution

### Mathematical Formulation
Given binary true labels $y_{true} \in \{0, 1\}^N$ ($N_p = \sum y_{true}$, $N_n = N - N_p$) and continuous risk scores $y_{prob} \in [0.0, 1.0]^N$:

1. **Ordering & Distinct Thresholds**:
   Sort scores descending using a stable sort. Identify distinct threshold values $v_k$ to avoid step-wise ordering bias on tied scores.
2. **Cumulative True/False Positives**:
   $$TP_k = \sum_{i: y_{prob, i} \ge v_k, y_{true, i} = 1} 1, \quad FP_k = \sum_{i: y_{prob, i} \ge v_k, y_{true, i} = 0} 1$$
3. **Rates & Origin**:
   Prepend origin $(FPR_0 = 0.0, TPR_0 = 0.0)$.
   $$TPR_k = \frac{TP_k}{N_p}, \quad FPR_k = \frac{FP_k}{N_n}$$
4. **Trapezoidal Integration**:
   $$\text{AUC} = \int_0^1 TPR(FPR) \, d(FPR) \approx \sum_{k=1}^K \frac{TPR_k + TPR_{k-1}}{2} \cdot (FPR_k - FPR_{k-1})$$

### Invariance to Ties
Handling distinct score thresholds ensures that tied prediction scores form diagonal trapezoids rather than artificial vertical or horizontal steps, matching exact Mann-Whitney U rank sums with fractional mid-ranks:
$$\text{AUC} = \frac{R_p - \frac{N_p(N_p + 1)}{2}}{N_p \cdot N_n}$$

---

## 3. Research Decision 2: 3-Tier Ablation Study Architecture

To quantitatively isolate and justify each algorithmic component, we evaluate three diagnostic tiers across identical test cohorts:

| Tier | System Architecture | Diagnostic Decision Mechanism | Clinical Vulnerability |
|---|---|---|---|
| **Tier 1** | Pure MobileNetV2 Black-Box | $p_{cnn} \ge 0.50 \implies \text{Malignant}$ | Spurious non-lesional artifacts (rulers, gel bubbles) trigger false positives ($FP$). |
| **Tier 2** | Crisp ABCD Fixed Thresholds | $I_j = \mathbb{I}(x_j \ge P_{50, j})$. If $\frac{1}{4}\sum I_j \ge 0.50 \implies \text{Malignant}$ | Hard cutoffs cause boundary instability and false negatives on subtle lesions. |
| **Tier 3** | Proposed DermaFuzzy-XAI | Grad-CAM Gating ($D_{verified}$) + Mamdani FIS ($R_1$–$R_5$) | Robust to off-lesion artifacts (Rule 5 suppression); high sensitivity via fuzzy overlap. |

### Metric Reporting
The `AblationReport` compiles:
- `Sensitivity (Recall)`: $\frac{TP}{TP + FN}$ (target $\ge 90\%$)
- `Specificity`: $\frac{TN}{TN + FP}$ (DermaFuzzy-XAI targets $+15\%$ to $+25\%$ gain over Tier 1 on artifact cohorts)
- `Precision`: $\frac{TP}{TP + FP}$
- `Accuracy`: $\frac{TP + TN}{N}$
- `F1-Score`: $\frac{2 \cdot P \cdot R}{P + R}$
- `AUC-ROC`: Trapezoidal area under the curve

---

## 4. Research Decision 3: Headless 4-Panel Clinical XAI Visualizer

### Architectural Requirements
- **Headless Isolation**: Use `from matplotlib.figure import Figure` and `from matplotlib.backends.backend_agg import FigureCanvasAgg` directly, avoiding `pyplot` or GUI event loop initialization.
- **Lazy Import**: Matplotlib is imported strictly inside visualization functions, keeping `import dermafuzzy_core` sub-second and thread-safe.
- **Figure Dimensions**: $16 \times 4.5$ inches at $300\text{ DPI}$ yielding $4800 \times 1350$ pixels.
- **Panel Composition**:
  - **Panel 1 (Lesion Segmentation)**: Preprocessed RGB image overlaid with emerald green contour of lesion mask ($M$). Title: Asymmetry $A$ and Color Variation $C$.
  - **Panel 2 (Boundary Fuzziness)**: Perilesional boundary ring ($R_{edge}$) highlighted in cyan/gold. Title: $B_{geo}$, $B_{grad}$, and total $B_{fuzzy}$.
  - **Panel 3 (Attention Verification)**: Preprocessed RGB blended with JET-colormapped Grad-CAM attention heatmap ($0.65 \times \text{RGB} + 0.35 \times \text{JET}$). Title: Overlap $\alpha$ and Verified Confidence $D_{verified}$.
  - **Panel 4 (Mamdani Rule Firings & Decision)**: Horizontal bar chart of rule firings $R_1$–$R_5 \in [0.0, 1.0]$. Dominant rule highlighted. Clinical decision banner (`Low: Green`, `Medium: Amber`, `High: Crimson`) displaying Centroid Risk $\%$.
- **Memory Safety**: `canvas.buffer_rgba()` extracted to NumPy array; `fig.clf()` and explicit object release prevent memory growth in batch runs.

---

## 5. Research Decision 4: Colab HAM10000 Service & Synthetic Dry-Run

### 5-Stage Pipeline (`train_ham10000.py`)
1. **Balanced Stratified Sampling**: Selects $1,000$ Melanoma (`mel`) and $1,000$ Benign Nevi (`nv`) from HAM10000 metadata. Splits $70\%$ train ($1,400$), $15\%$ val ($300$), $15\%$ test ($300$) using seed $42$.
2. **Deterministic Preprocessing**: Runs Room 2 OpenCV pipeline on all images.
3. **Transfer Learning & Grad-CAM**: Fine-tunes `MobileNetV2` on `out_relu` feature maps; computes Grad-CAM heatmaps via `tf.GradientTape`.
4. **Training Calibration**: Calibrates training split percentiles to `artifacts/fis_calibration.json`.
5. **Evaluation & Artifacts**: Evaluates 3-tier ablation study; exports `artifacts/ablation_report.md`, `artifacts/ablation_metrics.csv`, and sample `artifacts/sample_4panel_xai.png`.

### `--dry-run-synthetic` Mode
Allows local execution and CI testing without downloading HAM10000. Generates synthetic phantoms (circle, L-shape, multi-pigment, ruler artifacts), mocks feature extraction, executes full calibration, runs the ablation study, and renders the 4-panel figure in $< 10$ seconds.

---

## 6. Research Decision 5: GitHub Actions CI Matrix

- Runs on `push` and `pull_request` against `main`.
- Runner: `ubuntu-latest` with Python 3.12+.
- Executes:
  1. Package editable install: `pip install -e .[dev]` or `pip install -e . && pip install pytest matplotlib`.
  2. Constitution Principle I check: `python -c "import dermafuzzy_core; import sys; assert 'tensorflow' not in sys.modules; assert 'torch' not in sys.modules"`.
  3. Full unit and metamorphic test suite: `pytest -v tests/`.
  4. Colab trainer dry-run smoke test: `python services/colab_trainer/train_ham10000.py --dry-run-synthetic`.
