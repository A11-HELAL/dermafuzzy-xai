# Tasks: Walking Skeleton & Unified Configuration Engine

**Input**: Design documents from `/specs/001-walking-skeleton-config/`
**Prerequisites**: [plan.md](plan.md), [spec.md](spec.md), [research.md](research.md), [data-model.md](data-model.md), [contracts/](contracts/)
**Tests**: Unit tests are required by the specification and Constitution Principle IV.
**Organization**: Tasks are grouped by user story to enable independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (`[US1]`, `[US2]`, `[US3]`)
- Include exact file paths in all descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Establish the 6-room modular directory layout, root configuration, and build/test configuration.

- [X] T001 Create the 6-room modular repository directory structure: `packages/dermafuzzy_core/`, `services/colab_trainer/`, `artifacts/`, and `tests/unit/`
- [X] T002 [P] Create packaging and test discovery configuration in `pyproject.toml` with `[tool.pytest.ini_options]` setting `pythonpath = ["packages"]`
- [X] T003 [P] Create root configuration file `derma_config.yaml` specifying all default hyperparameters: `dataset` (name: "HAM10000", image_size: [224, 224], target_classes: {"malignant": "mel", "benign": "nv"}, samples_per_class: 1000, random_state: 42, split_ratios: {"train": 0.70, "val": 0.15, "test": 0.15}), `preprocessing` (blackhat_kernel_size: 17, inpaint_radius: 3, clahe_clip_limit: 2.0, clahe_tile_grid: [8, 8], gaussian_blur_kernel: 5, boundary_ring_kernel: 5, mask_area_bounds: [0.02, 0.90], eps: 1.0e-6), `biomarkers` (b_fuzzy_weights: {"w_geo": 0.5, "w_grad": 0.5}, feature_names: ["A", "B_fuzzy", "C", "D_verified"], value_range: [0.0, 1.0]), and `mamdani_fis` (percentiles: [25, 50, 75], defuzzification_method: "centroid", risk_thresholds: {"low_max": 35.0, "medium_max": 65.0})
- [X] T004 Create `packages/dermafuzzy_core/__init__.py` exposing version metadata and marking `dermafuzzy_core` as a pure Python package

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core domain entities and immutable dataclasses that all user stories depend on.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [X] T005 Implement immutable domain dataclasses in `packages/dermafuzzy_core/config.py` using standard library `@dataclass(frozen=True, slots=True)`: `DatasetConfig` (`name: str`, `image_size: tuple[int, int]`, `target_classes: dict[str, str]`, `samples_per_class: int`, `random_state: int`, `split_ratios: dict[str, float]`), `PreprocessingConfig` (`blackhat_kernel_size: int`, `inpaint_radius: int`, `clahe_clip_limit: float`, `clahe_tile_grid: tuple[int, int]`, `gaussian_blur_kernel: int`, `boundary_ring_kernel: int`, `mask_area_bounds: tuple[float, float]`, `eps: float`), `BiomarkersConfig` (`b_fuzzy_weights: dict[str, float]`, `feature_names: list[str]`, `value_range: tuple[float, float]`), `MamdaniFISConfig` (`percentiles: list[float]`, `defuzzification_method: str`, `risk_thresholds: dict[str, float]`), and aggregate root `DermaConfig`

**Checkpoint**: Foundation ready - domain entities are defined and typed; user story implementation can begin.

---

## Phase 3: User Story 1 - Unified Configuration Loading & Domain Typing (Priority: P1) 🎯 MVP

**Goal**: Parse and load `derma_config.yaml` from disk into immutable domain dataclasses with exact field fidelity.

**Independent Test**: Invoke `load_config()` on `derma_config.yaml` to confirm that all nested sections are loaded as typed attributes, and verify that attempting to mutate any attribute raises `dataclasses.FrozenInstanceError`.

### Tests for User Story 1 ⚠️
> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [X] T006 [US1] Write unit tests in `tests/unit/test_config.py` verifying successful loading of valid YAML configs, exact field type and value matches, `FileNotFoundError` on missing path, and runtime immutability enforcement (`FrozenInstanceError`)

### Implementation for User Story 1

- [X] T007 [US1] Implement `load_config(path: str | Path | None = None) -> DermaConfig` in `packages/dermafuzzy_core/config.py` using `yaml.safe_load` and `pathlib.Path`, converting raw dictionary sections into `DermaConfig` and sub-dataclass instances

**Checkpoint**: At this point, User Story 1 is fully functional and testable independently (MVP ready).

---

## Phase 4: User Story 2 - Strict Mathematical Invariant & Guardrail Enforcement (Priority: P2)

**Goal**: Validate domain mathematical constraints, kernel geometries, and percentile distributions, raising explicit `ValueError` on violations.

**Independent Test**: Run unit tests using pytest's `tmp_path` fixture passing corrupt configurations (even kernel size, invalid split ratio sum, invalid weight sum, non-monotonic percentiles, inverted area bounds) and confirm each raises `ValueError` with descriptive messaging.

### Tests for User Story 2 ⚠️
> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [X] T008 [US2] Write negative validation unit tests in `tests/unit/test_config.py` using `pytest.raises(ValueError)` and `tmp_path` covering: even kernel sizes (e.g. 16, 4), non-positive kernels, split ratio sums $\ne 1.0$, biomarker weight sums $\ne 1.0$, non-increasing or out-of-bounds percentiles (e.g. [50, 25, 75] or [0, 50, 100]), invalid mask area bounds, and non-[0.0, 1.0] value ranges

### Implementation for User Story 2

- [X] T009 [US2] Implement `validate_config(raw_dict: dict[str, Any]) -> DermaConfig` in `packages/dermafuzzy_core/config.py` enforcing:
  - $\sum_{k \in split\_ratios} split\_ratios[k] == 1.0 \pm 1e-6$ via `math.isclose(..., abs_tol=1e-6)`
  - $w\_geo + w\_grad == 1.0 \pm 1e-6$ via `math.isclose(..., abs_tol=1e-6)`
  - `blackhat_kernel_size`, `gaussian_blur_kernel`, and `boundary_ring_kernel` MUST be strictly positive odd integers ($k > 0 \land k \pmod 2 == 1$)
  - `percentiles` MUST be strictly monotonically increasing within $(0, 100)$ ($0 < p_1 < p_2 < \dots < 100$)
  - `mask_area_bounds` MUST satisfy $0.0 < min\_bound < max\_bound < 1.0$
  - `value_range` MUST equal `[0.0, 1.0]`

**Checkpoint**: User Stories 1 and 2 are functional and fully covered by positive and negative unit tests.

---

## Phase 5: User Story 3 - Walking Skeleton Architecture & Boundary Verification (Priority: P3)

**Goal**: Verify pure core boundary isolation, ensuring zero top-level heavy framework imports and no training-serving skew.

**Independent Test**: Verify that importing `dermafuzzy_core.config` executes in $<10$ms and introduces zero heavy ML framework entries (`tensorflow`, `keras`, `torch`, `cv2`) into `sys.modules`.

### Tests for User Story 3 ⚠️
> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [X] T010 [US3] Add unit test `test_zero_heavy_imports` in `tests/unit/test_config.py` inspecting `sys.modules` after importing `dermafuzzy_core.config` to assert that `tensorflow`, `keras`, `torch`, and `cv2` are absent

### Implementation for User Story 3

- [X] T011 [US3] Audit `packages/dermafuzzy_core/config.py` and `packages/dermafuzzy_core/__init__.py` to guarantee zero top-level imports of heavy ML frameworks and pure execution complying with Constitution Principle I

**Checkpoint**: All three user stories are functional and isolated within the architectural boundary.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Execute full verification suite and confirm compliance with project constitution and quickstart guide.

- [X] T012 Run complete unit test suite with `pytest -v tests/unit/test_config.py` and verify all tests pass deterministically in under 1.0 second
- [X] T013 Run quickstart verification scenarios from `specs/001-walking-skeleton-config/quickstart.md` to confirm end-to-end walking skeleton readiness

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately.
- **Foundational (Phase 2)**: Depends on Setup (T001-T004) completion - BLOCKS all user stories.
- **User Story 1 (Phase 3)**: Depends on Foundational dataclasses (T005). Delivers the initial MVP.
- **User Story 2 (Phase 4)**: Depends on User Story 1 configuration loading (T007).
- **User Story 3 (Phase 5)**: Depends on User Story 1 & 2 completion.
- **Polish (Phase 6)**: Depends on all user story phases (T001-T011) being complete.

### User Story Dependencies

- **User Story 1 (P1)**: Foundational dataclasses ready → TDD tests (T006) → implementation (T007).
- **User Story 2 (P2)**: Extends T007 loading with validation rules → TDD tests (T008) → implementation (T009).
- **User Story 3 (P3)**: Validates architectural cleanliness of T005, T007, and T009 → TDD tests (T010) → audit (T011).

### Parallel Opportunities

- **Phase 1 Setup**:
  - `T002` (`pyproject.toml`) and `T003` (`derma_config.yaml`) can execute in parallel once `T001` directories are created.
- **Phase 6 Polish**:
  - `T012` (`pytest` execution) and `T013` (quickstart verification) can run in sequence or parallel verification passes.

---

## Implementation Strategy

### MVP First (User Story 1 Only)
1. Complete Phase 1: Setup (`T001`, `T002`, `T003`, `T004`).
2. Complete Phase 2: Foundational dataclasses (`T005`).
3. Complete Phase 3: User Story 1 tests (`T006`) and loader implementation (`T007`).
4. **VALIDATE MVP**: Confirm `load_config()` successfully parses `derma_config.yaml`.

### Incremental Delivery
1. Add User Story 2 (`T008`, `T009`) to enforce domain mathematical bounds and rejection gates.
2. Add User Story 3 (`T010`, `T011`) to enforce pure core import boundary constraints.
3. Run Phase 6 Polish (`T012`, `T013`) to verify full test suite and quickstart readiness.
