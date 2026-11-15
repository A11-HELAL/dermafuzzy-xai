# Tasks: Clinical Evaluation, 3-Tier Ablation Study, 4-Panel XAI Visualizer & Colab MobileNetV2 Pipeline

**Input**: Design documents from `specs/004-ablation-visualizer-colab-ci/`
**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/`
**Tests**: Included per Constitution Principle IV (Test-First & Config-Driven Engineering) and Spec 004 verification criteria.
**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

---

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Verify dependencies and rendering environment for Room 5 evaluation and visualizer modules.

- [X] T001 Verify `matplotlib>=3.8.0` in `pyproject.toml` and confirm `FigureCanvasAgg` headless execution in Python 3.14

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core immutable data models that MUST be complete before any user story can be implemented.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [X] T002 [P] Implement `ClinicalMetrics` immutable dataclass with `@dataclass(frozen=True, slots=True)` (`tp, tn, fp, fn: int`, `sensitivity, specificity, precision, accuracy, f1_score, auc_roc, threshold: float`, `to_dict()`) in `packages/dermafuzzy_core/evaluation.py`
- [X] T003 [P] Implement `AblationReport` immutable dataclass with `MappingProxyType` immutability, `.to_dict()`, `.to_markdown_table()`, and `.to_csv()` in `packages/dermafuzzy_core/evaluation.py`
- [X] T004 [P] Implement `FourPanelReportConfig` dataclass (`figsize: tuple[float, float] = (16.0, 4.5)`, `dpi: int = 300`, `contour_color: tuple[int, int, int] = (0, 255, 0)`, `ring_color: tuple[int, int, int] = (0, 255, 255)`, `heatmap_alpha: float = 0.35`) in `packages/dermafuzzy_core/visualizer.py`

**Checkpoint**: Foundation ready - user story implementation can now begin.

---

## Phase 3: User Story 1 - Clinical Evaluation Metrics & 3-Tier Ablation Study Engine (Room 5) (Priority: P1) 🎯 MVP

**Goal**: Deliver pure NumPy clinical evaluation metrics (Sensitivity, Specificity, Precision, Accuracy, F1-Score, and exact tied-score trapezoidal AUC-ROC) and the 3-tier comparative ablation study engine (`Pure MobileNetV2 Black-Box` vs. `Crisp ABCD Fixed Thresholds` vs. `Proposed DermaFuzzy-XAI`).

**Independent Test**: Execute `pytest tests/unit/test_evaluation.py`. Verify exact confusion matrix counts, boundary edge cases (all-positive, all-negative cohorts), tied-score trapezoidal AUC-ROC matching Mann-Whitney U fractional ranks, and ablation report generation proving DermaFuzzy-XAI superiority on ruler-artifact cohorts.

### Tests for User Story 1 ⚠️
> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [X] T005 [P] [US1] Write unit tests in `tests/unit/test_evaluation.py` for `compute_clinical_metrics` covering confusion matrix counts ($TP, TN, FP, FN$), sensitivity/specificity, edge-case single-class cohorts with $\epsilon = 1e-6$ stabilizers, and tied-score pure NumPy trapezoidal AUC-ROC matching Mann-Whitney U mid-ranks ($|AUC_{numpy} - AUC_{ref}| \le 1e-4$)
- [X] T006 [P] [US1] Write unit tests in `tests/unit/test_evaluation.py` for `evaluate_crisp_abcd_baseline` and `run_ablation_study` verifying 3-tier metrics, `.to_markdown_table()`, `.to_csv()`, and proving DermaFuzzy-XAI's superior Specificity over Pure CNN on artifact-corrupted test cohorts

### Implementation for User Story 1

- [X] T007 [US1] Implement `compute_clinical_metrics(y_true, y_prob, threshold=0.50) -> ClinicalMetrics` with input validation (`bool`/`NaN`/`Inf` rejection), epsilon stabilizers, and exact sorted trapezoidal AUC-ROC handling ties in `packages/dermafuzzy_core/evaluation.py`
- [X] T008 [US1] Implement `evaluate_crisp_abcd_baseline(biomarkers_batch, calibration) -> np.ndarray` implementing hard step-function thresholding at $P_{50}$ in `packages/dermafuzzy_core/evaluation.py`
- [X] T009 [US1] Implement `run_ablation_study(y_true, p_cnn_raw, biomarkers_batch, calibration, config=None) -> AblationReport` evaluating Tier 1 (Pure CNN), Tier 2 (Crisp ABCD), and Tier 3 (DermaFuzzy-XAI) in `packages/dermafuzzy_core/evaluation.py`

**Checkpoint**: At this point, User Story 1 is fully functional and testable independently (`pytest tests/unit/test_evaluation.py` passes).

---

## Phase 4: User Story 2 - Publication-Grade 4-Panel Clinical XAI Visualizer (Room 5) (Priority: P2)

**Goal**: Deliver a publication-grade (300 DPI) 4-panel visual explanation report displaying preprocessed RGB with lesion contour, perilesional boundary ring with border scores, Grad-CAM attention overlay with overlap metrics, and horizontal Mamdani rule firing chart with clinical decision banner.

**Independent Test**: Execute `pytest tests/unit/test_visualizer.py`. Verify that `render_four_panel_report` renders a 300 DPI figure via headless `FigureCanvasAgg`, returns a read-only uint8 array (`arr.flags.writeable == False`), successfully exports PNG/PDF files, and exhibits zero figure handle leaks across batch renders.

### Tests for User Story 2 ⚠️
> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [X] T010 [P] [US2] Write unit tests in `tests/unit/test_visualizer.py` verifying `render_four_panel_report` outputs read-only uint8 RGB array (`arr.flags.writeable == False`), exports valid PNG/PDF files, produces non-zero pixel entropy, and maintains zero figure leaks across batch renders (`len(plt.get_fignums()) == 0`)

### Implementation for User Story 2

- [X] T011 [US2] Implement `render_four_panel_report(prep_result, biomarker_result, fis_prediction, heatmap, output_path=None, dpi=300) -> np.ndarray` in `packages/dermafuzzy_core/visualizer.py` using lazy `FigureCanvasAgg` rendering across Panels 1–4 (emerald contour, cyan boundary ring, JET Grad-CAM overlay, horizontal rule firing bar chart + banner)

**Checkpoint**: At this point, User Stories 1 AND 2 both work independently (`pytest tests/unit/test_evaluation.py tests/unit/test_visualizer.py` passes).

---

## Phase 5: User Story 3 - Google Colab Balanced HAM10000 Pipeline & CI Automation (Room 5 Service & CI) (Priority: P3)

**Goal**: Deliver the standalone Colab training pipeline (`services/colab_trainer/train_ham10000.py`) supporting MobileNetV2 fine-tuning on balanced HAM10000 with a `--dry-run-synthetic` mode for fast local verification, and automated GitHub Actions CI workflow.

**Independent Test**: Execute `pytest tests/unit/test_colab_trainer.py` and run `python services/colab_trainer/train_ham10000.py --dry-run-synthetic` to verify that all 5 training pipeline stages execute cleanly in $< 15$ seconds without external downloads.

### Tests for User Story 3 ⚠️
> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [X] T012 [P] [US3] Write smoke/unit tests in `tests/unit/test_colab_trainer.py` verifying `--dry-run-synthetic` generates expected artifacts (`artifacts/ablation_report.md`, `artifacts/ablation_metrics.csv`, `artifacts/sample_4panel_xai.png`) and exits with code 0 in $< 15$ seconds

### Implementation for User Story 3

- [X] T013 [US3] Implement `services/colab_trainer/train_ham10000.py` supporting both full Colab GPU execution (balanced HAM10000 subset, MobileNetV2 `out_relu` Grad-CAM, training calibration export) and `--dry-run-synthetic` fast local execution
- [X] T014 [US3] Create `.github/workflows/ci.yml` running linting, Constitution Principle I zero-ML-import scan, complete `pytest -v tests/` suite, and `train_ham10000.py --dry-run-synthetic`

**Checkpoint**: All user stories should now be independently functional.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Public API re-exports, documentation updates, and full end-to-end repository validation.

- [X] T015 [P] Re-export Room 5 symbols (`ClinicalMetrics`, `AblationReport`, `compute_clinical_metrics`, `evaluate_crisp_abcd_baseline`, `run_ablation_study`, `FourPanelReportConfig`, `render_four_panel_report`) in `packages/dermafuzzy_core/__init__.py`
- [X] T016 [P] Update root `README.md` with complete 6-room architecture documentation, mathematical formulas, quickstart examples, and publication-ready ablation tables
- [X] T017 Run complete test suite `pytest -v tests/` across all unit and metamorphic test files ensuring 100% pass rate
- [X] T018 Run Constitution Principle I verification script checking zero TensorFlow/PyTorch imports across all modules in `packages/dermafuzzy_core/`
- [X] T019 Execute quickstart validation scenarios from `specs/004-ablation-visualizer-colab-ci/quickstart.md`
- [X] T020 Run `python scripts/sync_live_state.py` and verify clean git status

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User Story 1 (P1): Can start after Foundational (Phase 2) - No dependencies on other stories
  - User Story 2 (P2): Can start after Foundational (Phase 2) - May use types from Room 2/3/4 but independently testable
  - User Story 3 (P3): Can start after Foundational (Phase 2) - Integrates US1 and US2 for end-to-end dry-run
- **Polish (Phase 6)**: Depends on all user stories being complete

### Parallel Opportunities

- Within Phase 2: T002, T003, T004 can run in parallel.
- Within Phase 3: T005 and T006 tests can be authored in parallel.
- Within Phase 4: T010 can be authored in parallel with US1 tests.
- Within Phase 5: T012 can be authored in parallel with US2 tests.
- Within Phase 6: T015 and T016 can run in parallel.
