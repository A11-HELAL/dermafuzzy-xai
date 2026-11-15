# Tasks: Data-Calibrated Mamdani Fuzzy Decision Support System & Metamorphic QA Gate

**Input**: Design documents from `/specs/003-mamdani-fis-metamorphic-qa/`  
**Prerequisites**: [plan.md](plan.md), [spec.md](spec.md), [research.md](research.md), [data-model.md](data-model.md), [contracts/](contracts/)  
**Tests**: Complete unit and metamorphic test suites required by specification and Constitution Principle IV.  
**Organization**: Tasks are grouped by user story to enable independent implementation and testing.  

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (`[US1]`, `[US2]`, `[US3]`)
- Include exact file paths in all descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Update packaging dependencies and establish test directories.

- [X] T001 Verify `pyproject.toml` dependencies include `scipy>=1.11.0` and `scikit-fuzzy>=0.5.0` alongside `numpy`, `opencv-python-headless`, and `pyyaml`, and create `tests/metamorphic/` directory

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Domain dataclasses and clinical rule definitions required by all user stories.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [X] T002 [P] Implement `FuzzyMembershipSet` and `ClinicalRule` dataclasses in `packages/dermafuzzy_core/fis_engine.py` using `@dataclass(frozen=True, slots=True)` with Partition-of-Unity helper methods
- [X] T003 [P] Implement `FISCalibration` and `FISPrediction` dataclasses in `packages/dermafuzzy_core/fis_engine.py` using `@dataclass(frozen=True, slots=True)` with `MappingProxyType` deep immutability and JSON/dictionary conversion helpers (`to_dict`, `to_json`, `from_dict`, `from_json`)
- [X] T004 [P] Define `CLINICAL_RULES` tuple of 5 canonical clinical rules (`R1` through `R5`) with clinical descriptions and consequent mappings in `packages/dermafuzzy_core/fis_engine.py`

**Checkpoint**: Foundation ready - domain entities and static clinical rules available for TDD.

---

## Phase 3: User Story 1 - Data-Driven Training Calibration & Secure JSON Persistence (Room 4) (Priority: P1) 🎯 MVP

**Goal**: Calibrate empirical percentiles ($P_{25}, P_{50}, P_{75}$) from $(N, 4)$ training distributions with degenerate separation guard ($\delta = 1e-4$) and persist strictly to schema-validated JSON (`artifacts/fis_calibration.json`, zero `pickle`).

**Independent Test**: Run `pytest tests/unit/test_fis_engine.py -k calibration` to verify empirical percentile calculation, degenerate column spacing, schema validation, mutation rejection via `MappingProxyType`, and zero `pickle` usage.

### Tests for User Story 1 ⚠️
> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [X] T005 [P] [US1] Write unit tests in `tests/unit/test_fis_engine.py` verifying `calibrate_fis_percentiles`, degenerate constant-column separation guard ($\delta = 1e-4$), `save_calibration_json`, `load_calibration_json`, `MappingProxyType` mutation rejection (`TypeError`), and negative validation (`bool`, `NaN`, `Inf`, wrong shapes, out-of-bounds values)

### Implementation for User Story 1

- [X] T006 [US1] Implement `calibrate_fis_percentiles(train_matrix, config=None)` with input validation, `np.percentile`, and the degenerate separation guard enforcing $0.0 < P_{25} < P_{50} < P_{75} < 1.0$ with minimum separation $\delta = 1e-4$ in `packages/dermafuzzy_core/fis_engine.py`
- [X] T007 [US1] Implement `save_calibration_json(calibration, filepath)` and `load_calibration_json(filepath)` with schema validation, directory creation, and `MappingProxyType` wrapping in `packages/dermafuzzy_core/fis_engine.py`
- [X] T008 [US1] Generate and calibrate baseline training artifact `artifacts/fis_calibration.json` using synthetic beta distributions representing balanced benign/malignant training samples

**Checkpoint**: User Story 1 (Room 4 MVP) is fully functional and testable independently.

---

## Phase 4: User Story 2 - Pure Mamdani Fuzzy Inference Engine & Clinical Rules (Room 4) (Priority: P2)

**Goal**: Implement vectorized continuous fuzzification satisfying Partition of Unity ($\mu_L + \mu_M + \mu_H \equiv 1.0$), evaluate the 5 Mamdani clinical rules with Complete Rule Partition Guard ($\sum W_c > 0$), defuzzify via Centroid over $[0.0, 100.0]$, and provide the end-to-end `diagnose_image` orchestrator.

**Independent Test**: Run `pytest tests/unit/test_fis_engine.py -k inference` to verify Partition of Unity over 10,000 points, pure Low ($< 20\%$), pure Medium ($\approx 50\%$), worked reference benchmark $(0.68, 0.74, 0.81, 0.80) \to [80\%, 88\%]$ High Risk, and `diagnose_image` pipeline execution.

### Tests for User Story 2 ⚠️
> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [X] T009 [P] [US2] Write unit tests in `tests/unit/test_fis_engine.py` verifying `compute_memberships` (Partition of Unity $\mu_L + \mu_M + \mu_H = 1.0 \pm 1e-6$ across 10,000 points), `evaluate_mamdani_fis` (pure Low, pure Medium, study guide worked reference case in $[80\%, 88\%]$, artifact protection case in `Low`), and `diagnose_image`

### Implementation for User Story 2

- [X] T010 [US2] Implement `compute_memberships(score, cutoffs)` with exact continuous piecewise-linear functions (`trapmf` for Low, `trimf` for Medium, `trapmf` for High) guaranteeing analytical Partition of Unity in `packages/dermafuzzy_core/fis_engine.py`
- [X] T011 [US2] Implement Mamdani rule antecedent evaluations for Rules 1–5 (`R1` through `R5`) using Zadeh t-norm ($\min$) and s-norm ($\max$) in `packages/dermafuzzy_core/fis_engine.py`
- [X] T012 [US2] Implement Consequent Universe discretization ($z \in [0.0, 100.0]$ with 1001 points), Mamdani max-clip aggregation with Complete Rule Partition Guard, Centroid defuzzification via `skfuzzy.defuzz`, dominant rule resolution, and risk categorization in `packages/dermafuzzy_core/fis_engine.py`
- [X] T013 [US2] Implement `diagnose_image(img_rgb, p_cnn, heatmap, calibration, config=None)` orchestrating Room 2 `preprocess_image` $\to$ Room 3 `extract_abcd_biomarkers` $\to$ Room 4 `evaluate_mamdani_fis` in `packages/dermafuzzy_core/fis_engine.py`

**Checkpoint**: User Stories 1 and 2 are fully integrated and verified via unit tests.

---

## Phase 5: User Story 3 - End-to-End Metamorphic Diagnostic QA Gate (Room 6) (Priority: P3)

**Goal**: Implement and verify 5 end-to-end clinical Metamorphic Relations (MR1–MR5) across Preprocessing $\to$ Biomarkers $\to$ FIS Decision.

**Independent Test**: Run `pytest tests/metamorphic/test_metamorphic.py` and verify 100% pass rate across all 5 metamorphic relations.

### Tests for User Story 3 ⚠️
> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [X] T014 [P] [US3] Create `tests/metamorphic/__init__.py` and implement diagnostic evaluation harness with synthetic phantoms in `tests/metamorphic/test_metamorphic.py`
- [X] T015 [US3] Implement MR1 (Rotation Invariance): rotating input image and heatmap by 90°, 180°, and 270° changes final `risk_score` by $|\Delta \text{score}| \le 2.0\%$ and preserves `risk_category` in `tests/metamorphic/test_metamorphic.py`
- [X] T016 [US3] Implement MR2 (Peripheral Ruler Artifact Immunity): a benign circular lesion with high raw CNN probability ($p_{cnn}=0.92$) whose Grad-CAM heatmap focuses on a peripheral ruler ($\alpha \le 0.15$) classifies as `Low Risk` ($< 35.0\%$) via Rule 5 in `tests/metamorphic/test_metamorphic.py`
- [X] T017 [US3] Implement MR3 (Monotonic Biomarker Risk Response): increasing any single biomarker monotonically from $0.10$ to $0.90$ with others held constant never decreases `risk_score` ($\Delta \ge -1e-6$) in `tests/metamorphic/test_metamorphic.py`
- [X] T018 [US3] Implement MR4 (Partition Completeness Grid Sweep): evaluating $5 \times 5 \times 5 \times 5 = 625$ continuous grid points across $[0.0, 1.0]^4$ yields finite risk scores in $[0.0, 100.0]$ with zero `NaN` or zero-division errors in `tests/metamorphic/test_metamorphic.py`
- [X] T019 [US3] Implement MR5 (Deterministic Calibration Serialization): saving and reloading `fis_calibration.json` produces bit-identical `risk_score` and `dominant_rule_id` across test batches in `tests/metamorphic/test_metamorphic.py`

**Checkpoint**: All 5 Metamorphic Relations pass deterministically.

---

## Phase 6: Public API Integration

**Purpose**: Re-export Room 4 public API symbols cleanly in package root.

- [X] T020 Update `packages/dermafuzzy_core/__init__.py` to re-export `FISCalibration`, `FuzzyMembershipSet`, `ClinicalRule`, `FISPrediction`, `CLINICAL_RULES`, `calibrate_fis_percentiles`, `save_calibration_json`, `load_calibration_json`, `compute_memberships`, `evaluate_mamdani_fis`, and `diagnose_image`

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Execute full verification suite and confirm compliance with constitution and quickstart guide.

- [X] T021 [P] Run complete test suite `pytest -v tests/` confirming 100% pass rate across unit and metamorphic test rooms
- [X] T022 [P] Verify Constitution Principle I compliance across all modules in `packages/dermafuzzy_core/` asserting zero imports of `tensorflow`, `keras`, or `torch`
- [X] T023 Run quickstart verification scenarios from `specs/003-mamdani-fis-metamorphic-qa/quickstart.md` confirming sub-second execution

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately.
- **Foundational (Phase 2)**: Depends on Setup (T001) - BLOCKS all user stories.
- **User Story 1 (Phase 3)**: Depends on Foundational (T002-T004). Delivers Room 4 MVP.
- **User Story 2 (Phase 4)**: Depends on US1 (T006-T008). Delivers Mamdani FIS engine.
- **User Story 3 (Phase 5)**: Depends on US2 (T010-T013). Delivers Room 6 Metamorphic QA.
- **Public API (Phase 6)**: Depends on US1, US2, and US3 completion.
- **Polish (Phase 7)**: Depends on all prior phases being complete.

### User Story Dependencies

- **User Story 1 (P1)**: Foundational dataclasses ready → TDD tests (T005) → implementation (T006-T008).
- **User Story 2 (P2)**: Calibration ready → TDD tests (T009) → implementation (T010-T013).
- **User Story 3 (P3)**: Preprocessing, Biomarkers, and FIS ready → TDD harness (T014) → MR1–MR5 implementation (T015-T019).

### Parallel Opportunities

- **Phase 2 Foundational**: `T002` (memberships/rules), `T003` (calibration/prediction), and `T004` (`CLINICAL_RULES`) can run in parallel.
- **Phase 3 & Phase 4 Tests**: Test writing tasks (`T005`, `T009`, `T014`) can be authored in parallel.
- **Phase 7 Polish**: Unit/metamorphic run (`T021`) and Constitution scan (`T022`) can run in parallel.

---

## Implementation Strategy

### MVP First (User Story 1 Only)
1. Complete Phase 1: Setup (`pyproject.toml`, `tests/metamorphic/`)
2. Complete Phase 2: Foundational dataclasses (`FISCalibration`, `FuzzyMembershipSet`, `FISPrediction`)
3. Complete Phase 3: User Story 1 (Calibration & JSON serialization)
4. **STOP and VALIDATE**: Verify `pytest tests/unit/test_fis_engine.py -k calibration` passes.

### Incremental Delivery
1. Add User Story 2 (Fuzzification, Rules, Defuzzification) $\to$ Verify worked example $(0.68, 0.74, 0.81, 0.80) \to 80\%-88\%$.
2. Add User Story 3 (Room 6 Metamorphic Relations MR1–MR5) $\to$ Verify rotation invariance, artifact immunity, monotonicity.
3. Re-export public API in `dermafuzzy_core/__init__.py`.
4. Final polish & quickstart verification.
