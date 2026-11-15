# Feature Specification: Clinical Evaluation, 3-Tier Ablation Study, 4-Panel XAI Visualizer & Colab MobileNetV2 Pipeline

**Feature Branch**: `004-ablation-visualizer-colab-ci`  
**Created**: 2026-11-15  
**Status**: Draft  
**Input**: User description: "Build Spec 004: Clinical Evaluation, 3-Tier Ablation Study, 4-Panel XAI Visualizer & Colab MobileNetV2 Pipeline (`004-ablation-visualizer-colab-ci`)."

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Clinical Evaluation Metrics & 3-Tier Ablation Study Engine (Room 5) (Priority: P1) 🎯 MVP

As a clinical biostatistician and regulatory medical reviewer, I want to evaluate binary diagnostic classification performance using rigorous clinical metrics (Sensitivity/Recall, Specificity, Precision, Accuracy, F1-Score, and AUC-ROC) and run an automated 3-tier ablation study comparing `Pure CNN Black-Box` vs. `Crisp ABCD Fixed Thresholds` vs. `Proposed DermaFuzzy-XAI`, so that I have quantitative, auditable proof of how fuzzy membership ramps and saliency verification suppress artifact-induced false positives while maintaining high sensitivity for malignant melanoma.

**Why this priority**:
Without rigorous evaluation metrics and comparative ablation evidence, no clinical diagnosis system can be scientifically validated, defended in peer-reviewed medical publications, or approved by medical device regulatory bodies (FDA/CE-MDR).

**Independent Test**:
Can be tested independently by supplying synthetic ground truth labels $y_{true} \in \{0, 1\}^N$ and continuous prediction probabilities $y_{prob} \in [0.0, 1.0]^N$, computing confusion matrices and exact trapezoidal AUC-ROC without external third-party dependencies, and evaluating all 3 ablation tiers to generate an immutable, formatted comparison report.

**Acceptance Scenarios**:
1. **Given** binary ground truth labels $y_{true}$ and continuous probability scores $y_{prob}$ in $[0.0, 1.0]$, **When** `compute_clinical_metrics` is executed at decision threshold $\tau = 0.50$, **Then** confusion matrix counts ($TP, TN, FP, FN$), sensitivity, specificity, precision, accuracy, F1-score, and exact sorted trapezoidal AUC-ROC are computed with epsilon stabilizers preventing zero-division errors.
2. **Given** a cohort with extreme test distributions (e.g., all-positive or all-negative cases), **When** metrics are computed, **Then** undefined ratios (e.g. $0/0$) safely resolve to $0.0$ or $1.0$ according to clinical convention without raising unhandled exceptions or returning `NaN`.
3. **Given** test cohort biomarker batches and raw CNN predictions, **When** `run_ablation_study` is executed across:
   - Tier 1: `Pure MobileNetV2 Black-Box` (unverified CNN confidence $p_{cnn}$)
   - Tier 2: `Crisp ABCD Fixed Thresholds` (hard step functions at $P_{50}$ without fuzzy ramps or spatial verification)
   - Tier 3: `Proposed DermaFuzzy-XAI` (full pipeline with Room 2 preprocessing, Room 3 Grad-CAM gating, and Room 4 Mamdani FIS)
   **Then** an immutable `AblationReport` is generated, demonstrating superior Specificity and F1-score in Tier 3 on artifact-corrupted test cohorts.
4. **Given** an `AblationReport`, **When** exported via `.to_markdown()` or `.to_csv()`, **Then** formatted comparison tables with percentage formatting and metric deltas are produced.

---

### User Story 2 - Publication-Grade 4-Panel Clinical XAI Visualizer (Room 5) (Priority: P2)

As a dermatologist examining a patient lesion, I want a standardized, publication-grade (300 DPI) 4-panel visual explanation report displaying the original lesion with segmented contour, the perilesional boundary ring with border irregularity scores, the Grad-CAM saliency overlay with attention overlap metrics, and a horizontal rule firing bar chart with final crisp risk score and clinical decision banner, so that I can immediately verify whether the AI decision is clinically sound or driven by spurious peripheral artifacts before making a biopsy recommendation.

**Why this priority**:
Clinicians reject black-box scores. A clear, multi-modal visual breakdown maps directly to dermatological ABCD criteria and allows instant human audit of model attention and rule logic.

**Independent Test**:
Can be tested independently by feeding a set of pre-computed diagnostic results (`PreprocessingResult`, `BiomarkerResult`, `FISPrediction`, and 2D Grad-CAM heatmap) into `render_four_panel_report`, verifying that a 4-panel image of shape $(H, W, 3)$ with 300 DPI is rendered via headless `matplotlib` (`Agg` backend), that all 4 panels contain required visual elements, that the output array is read-only (`arr.flags.writeable == False`), and that saving to `.png` and `.pdf` succeeds.

**Acceptance Scenarios**:
1. **Given** valid diagnostic results and attention heatmaps, **When** `render_four_panel_report` is executed, **Then** a 4-panel visual figure is rendered without requiring an active graphical desktop display server (`Agg` headless backend).
2. **Given** the 4 panels:
   - Panel 1: Preprocessed RGB image overlaid with the green contour of the primary lesion segmentation mask $M$.
   - Panel 2: Perilesional boundary ring $R_{edge}$ highlighted in cyan with geometric jaggedness $B_{geo}$, gradient fuzziness $B_{grad}$, and total $B_{fuzzy}$ annotated.
   - Panel 3: Jet colormap Grad-CAM attention heatmap overlaid with alpha transparency on RGB, annotated with spatial overlap $\alpha$ and verified confidence $D_{verified}$.
   - Panel 4: Horizontal bar chart showing firing strengths ($[0.0, 1.0]$) of all 5 clinical rules ($R_1$–$R_5$) with the dominant rule highlighted, final Centroid Risk % indicated by a vertical decision line, and a color-coded clinical decision banner (`Low Risk: Green`, `Medium Risk: Amber`, `High Risk: Crimson`).
   **When** rendered, **Then** all visual elements and annotations are rendered legibly at high resolution.
3. **Given** an optional file path ending in `.png` or `.pdf`, **When** `render_four_panel_report` is called with `output_path`, **Then** the file is written to disk and a read-only `uint8` NumPy array is returned.

---

### User Story 3 - Google Colab Balanced HAM10000 Pipeline & CI Automation (Room 5 Service & CI) (Priority: P3)

As a machine learning engineer and open-source contributor, I want a complete, standalone Google Colab training script (`services/colab_trainer/train_ham10000.py`) that trains MobileNetV2 on a balanced HAM10000 subset (1,000 Melanoma `mel` + 1,000 Benign Nevi `nv`), extracts `out_relu` Grad-CAM heatmaps, calibrates the training split to `artifacts/fis_calibration.json`, and exports ablation tables—equipped with a `--dry-run-synthetic` flag for local execution—and a GitHub Actions CI pipeline running the full 134+ test suite on every push.

**Why this priority**:
Provides the bridge between cloud GPU training on real clinical data (HAM10000) and lightweight clinical edge inference, while ensuring through automated CI that every commit strictly adheres to the engineering constitution and test pass gates.

**Independent Test**:
Can be tested independently by running `python services/colab_trainer/train_ham10000.py --dry-run-synthetic` locally, confirming that all 5 training pipeline stages execute to completion within seconds without external downloads, and verifying that `.github/workflows/ci.yml` passes GitHub Actions linting and executes `pytest` with a $\ge 95\%$ pass rate.

**Acceptance Scenarios**:
1. **Given** execution of `train_ham10000.py --dry-run-synthetic`, **When** launched on an edge CPU machine, **Then** the script generates synthetic balanced data ($n=20$ per class), runs mock transfer learning, generates mock Grad-CAM heatmaps, calibrates percentiles to `artifacts/fis_calibration.json`, and outputs an ablation summary table in $< 15$ seconds.
2. **Given** execution in a Google Colab GPU environment, **When** run with full arguments, **Then** HAM10000 metadata is filtered for balanced sampling ($1,000$ `mel` + $1,000$ `nv`), stratified split ($70\%$ train, $15\%$ val, $15\%$ test with `random_state=42`), MobileNetV2 is fine-tuned, `tf.GradientTape` extracts Grad-CAM on layer `out_relu`, and training percentiles are exported to `artifacts/fis_calibration.json`.
3. **Given** any commit or pull request to `main`, **When** GitHub Actions triggers `.github/workflows/ci.yml`, **Then** the CI runner installs dependencies, verifies zero heavy imports in `packages/dermafuzzy_core/`, and executes the complete `pytest -v` unit and metamorphic test suite with $100\%$ pass rate.

---

### Edge Cases

- **Zero Positive Cases ($TP + FN = 0$)**: Sensitivity denominator is zero. Epsilon stabilizer ($\epsilon = 1e-6$) or zero-division guard prevents crash and yields $0.0$.
- **Zero Negative Cases ($TN + FP = 0$)**: Specificity denominator is zero. Epsilon stabilizer yields $0.0$.
- **All Predictions Identical ($y_{prob} = \text{const}$)**: ROC curve has no threshold discrimination. AUC-ROC trapezoidal integration must return $0.50$ (chance level) rather than failing.
- **Headless Display Environments**: Running visualization code on Linux servers or CI containers without an X11 display server. Matplotlib must strictly use the headless `Agg` backend (`matplotlib.use("Agg")`) before importing `pyplot`.
- **Very Small or Blurry Lesion Masks in Visualizer**: Visualizer contours and bounding rings must not crash or produce invalid polygon coordinates when mask area is minimal.
- **Memory Leaks in Repeated Visualizations**: Matplotlib figures must be explicitly closed via `plt.close(fig)` after converting canvas to NumPy array to prevent memory leaks in batch report generation.

---

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide `compute_clinical_metrics(y_true, y_prob, threshold=0.50) -> ClinicalMetrics` in `packages/dermafuzzy_core/evaluation.py`.
- **FR-002**: System MUST compute confusion matrix counts ($TP, TN, FP, FN$) and clinical metrics: Sensitivity (Recall), Specificity, Precision, Accuracy, F1-Score, and AUC-ROC.
- **FR-003**: System MUST calculate AUC-ROC using an exact sorted trapezoidal integration algorithm implemented in pure NumPy without requiring `scikit-learn`.
- **FR-004**: System MUST provide `evaluate_crisp_abcd_baseline(biomarkers_batch, calibration) -> np.ndarray` implementing hard step-function thresholding at $P_{50}$ without fuzzy membership ramps or Grad-CAM attention gating.
- **FR-005**: System MUST provide `run_ablation_study(y_true, p_cnn_raw, biomarkers_batch, calibration, config=None) -> AblationReport` evaluating all 3 diagnostic tiers.
- **FR-006**: System MUST return an immutable `@dataclass(frozen=True, slots=True) class AblationReport` providing `.to_dict()`, `.to_markdown()`, and `.to_csv()` export methods.
- **FR-007**: System MUST provide `render_four_panel_report(prep_result, biomarker_result, fis_prediction, heatmap, output_path=None, dpi=300) -> np.ndarray` in `packages/dermafuzzy_core/visualizer.py`.
- **FR-008**: System MUST configure Matplotlib with the non-interactive `Agg` backend before importing `pyplot` to ensure headless execution in server and CI environments.
- **FR-009**: System MUST render Panel 1: Preprocessed RGB image with green lesion contour overlay.
- **FR-010**: System MUST render Panel 2: Boundary ring $R_{edge}$ overlay in cyan with $B_{geo}, B_{grad}, B_{fuzzy}$ scores.
- **FR-011**: System MUST render Panel 3: Jet Grad-CAM attention heatmap overlay with $\alpha$ and $D_{verified}$ scores.
- **FR-012**: System MUST render Panel 4: Horizontal Mamdani rule firing bar chart ($R_1$–$R_5$), dominant rule highlight, Centroid risk percentage, and color-coded risk decision banner (`Low: Green`, `Medium: Amber`, `High: Crimson`).
- **FR-013**: System MUST close all Matplotlib figure handles explicitly after rasterization to prevent memory leaks, returning a read-only NumPy array (`arr.flags.writeable = False`).
- **FR-014**: System MUST provide `services/colab_trainer/train_ham10000.py` implementing the 5-stage balanced MobileNetV2 pipeline with `--dry-run-synthetic` CLI support.
- **FR-015**: System MUST provide `.github/workflows/ci.yml` running linting, Pure Core zero-ML import verification, and `pytest -v` across unit and metamorphic test suites.
- **FR-016**: System MUST maintain strict Constitution Principle I compliance: zero module-level imports of `tensorflow`, `keras`, or `torch` in `packages/dermafuzzy_core/evaluation.py` and `packages/dermafuzzy_core/visualizer.py`.

---

### Key Entities *(include if feature involves data)*

- **`ClinicalMetrics`**: Immutable dataclass containing confusion matrix counts ($TP, TN, FP, FN$), sensitivity, specificity, precision, accuracy, f1_score, auc_roc, and threshold.
- **`AblationTierResult`**: Dataclass holding evaluation metrics and predictions for a single ablation tier.
- **`AblationReport`**: Immutable container holding comparative results across all 3 tiers (`Pure CNN`, `Crisp ABCD`, `DermaFuzzy-XAI`), delta improvements, and Markdown/CSV export methods.
- **`FourPanelReportConfig`**: Configuration settings for the clinical figure visualizer (figure size, DPI, colormaps, font sizes, banner colors).

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% test pass rate across all unit tests (`test_evaluation.py`, `test_visualizer.py`) and existing suites.
- **SC-002**: Pure NumPy AUC-ROC calculation matches Scipy/Scikit-learn trapezoidal integration within $|AUC_{numpy} - AUC_{ref}| \le 1e-4$.
- **SC-003**: 4-Panel report generation executes in $< 1.5\text{s}$ at 300 DPI on edge CPU hardware and returns a read-only uint8 array of shape $(H, W, 3)$ with non-zero pixel entropy.
- **SC-004**: Memory safety: Batch rendering of 50 reports exhibits zero figure handle leaks (`plt.get_fignums()` is empty after each render).
- **SC-005**: Ablation study demonstrability: DermaFuzzy-XAI exhibits statistically superior Specificity on artifact-corrupted synthetic benchmark cohorts compared to Pure CNN black-box.
- **SC-006**: Dry-run smoke test: `python services/colab_trainer/train_ham10000.py --dry-run-synthetic` completes end-to-end in $< 15\text{s}$ with exit code 0.

---

## Assumptions

- Matplotlib is available or installed in the virtual environment as an optional rendering dependency for Room 5 visualizer.
- Google Colab environment provides TensorFlow/Keras 2.x for `services/colab_trainer/train_ham10000.py`; local core library tests run strictly without importing TensorFlow.
- Binary melanoma classification uses label `1` for Malignant (`mel`) and `0` for Benign Nevus (`nv`), matching `config.dataset.target_classes`.
