# Feature Specification: Walking Skeleton & Unified Configuration Engine

**Feature Branch**: `001-walking-skeleton-config`

**Created**: 2026-11-15

**Status**: Draft

**Input**: User description: "Build Spec 001: Walking Skeleton & Unified Configuration Engine (001-walking-skeleton-config). Objective: Establish the 6-room modular repository structure for DermaFuzzy-XAI, the root derma_config.yaml configuration file, the pure validation engine packages/dermafuzzy_core/config.py, and the complete pytest unit test suite tests/unit/test_config.py."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Unified Configuration Loading & Domain Typing (Priority: P1)

As a computer vision researcher or clinical software engineer, I want the system to parse and load all pipeline hyperparameters from a single root configuration file into strongly-typed, immutable configuration structures so that all downstream training, feature extraction, and fuzzy inference components operate from a single, unambiguous source of truth.

**Why this priority**: Every stage of the DermaFuzzy-XAI pipeline (preprocessing, ABCD biomarker extraction, Grad-CAM attention verification, and Mamdani fuzzy inference) depends directly on standardized hyperparameters. Without verified, typed configuration loading, downstream processing cannot function deterministically.

**Independent Test**: Load a valid root configuration file from disk or memory, verify that all dataset, preprocessing, biomarker, and fuzzy logic parameters are parsed into structured domain objects, and verify that fields match expected types and values.

**Acceptance Scenarios**:

1. **Given** a valid `derma_config.yaml` file containing complete dataset, preprocessing, biomarker, and Mamdani FIS parameters, **When** the configuration loading function is invoked, **Then** an immutable configuration object is returned with all sections accessible as strongly-typed attributes.
2. **Given** a configuration file with missing sections or unparseable syntax, **When** configuration loading is attempted, **Then** the system raises a descriptive error detailing the missing section or format failure before any pipeline component runs.
3. **Given** loaded configuration objects, **When** an attempt is made to mutate any attribute at runtime, **Then** the mutation is rejected to prevent configuration drift during batch processing.

---

### User Story 2 - Strict Mathematical Invariant & Guardrail Enforcement (Priority: P2)

As a diagnostic systems developer, I want the configuration engine to strictly validate all mathematical constraints, spatial kernel geometries, and domain percentiles at startup so that invalid configurations are immediately rejected before executing computationally expensive operations.

**Why this priority**: Computer vision and fuzzy logic algorithms require strict mathematical properties (such as centered odd-sized kernels for symmetric spatial operations, normalized probability/weight distributions summing to 1.0, and monotonically increasing percentile thresholds). Catching violations at initialization eliminates silent numerical degradation and training failures.

**Independent Test**: Supply configurations with invalid values (even kernels, non-unitary split ratios or feature weights, or non-increasing percentiles) and verify that explicit, descriptive validation errors are raised.

**Acceptance Scenarios**:

1. **Given** a configuration where dataset `split_ratios` (train, val, test) do not sum to 1.0 (within $\epsilon = 1e-6$), **When** validation is executed, **Then** the system raises a `ValueError` identifying that split ratios must sum to 1.0.
2. **Given** a configuration where biomarker weights `b_fuzzy_weights` (`w_geo`, `w_grad`) do not sum to 1.0 (within $\epsilon = 1e-6$), **When** validation is executed, **Then** the system raises a `ValueError` indicating invalid weight normalization.
3. **Given** a configuration where any spatial kernel (`blackhat_kernel_size`, `gaussian_blur_kernel`, `boundary_ring_kernel`) is not a positive odd integer (e.g., an even integer or non-positive value), **When** validation is executed, **Then** the system raises a `ValueError` indicating that kernel sizes must be positive odd integers.
4. **Given** a configuration where `mamdani_fis.percentiles` are not strictly increasing within the open interval $(0, 100)$, **When** validation is executed, **Then** the system raises a `ValueError` indicating percentiles must be strictly increasing between 0 and 100.

---

### User Story 3 - Walking Skeleton Architecture & Boundary Verification (Priority: P3)

As a software maintainer, I want the repository organized into decoupled architectural spaces (pure algorithmic core, trainer services, serialized artifacts, and unit test suites) with an automated test suite verifying configuration integrity so that core algorithmic logic remains strictly isolated from heavy deep learning dependencies.

**Why this priority**: In accordance with Constitution Principle I, the pure core package must never depend on heavy ML frameworks or external state. Establishing this physical separation and automated verification early guarantees zero training-serving skew across future iterations.

**Independent Test**: Execute the unit test suite in an automated runner to verify all configuration loading behaviors and rejection gates without importing deep learning frameworks.

**Acceptance Scenarios**:

1. **Given** the repository workspace, **When** examining package imports, **Then** importing the core configuration engine executes without importing heavy deep learning frameworks (`tensorflow`, `keras`, `torch`).
2. **Given** the automated test runner, **When** executing configuration unit tests, **Then** all test cases covering valid configurations, default values, and boundary rejection conditions pass deterministically.

---

### Edge Cases

- **Floating-point rounding anomalies**: Summing split ratios (e.g., `0.70 + 0.15 + 0.15`) or weights may produce slight floating-point imprecision (e.g., `0.9999999999999999`). The system must validate sums using an absolute tolerance $\epsilon = 1e-6$.
- **Missing configuration file**: If the configuration file is not found at the requested path, the system must raise `FileNotFoundError` with a clear path description.
- **Empty or partial YAML content**: Empty files or YAML files missing root keys must be rejected with informative validation messages rather than unhandled key lookup errors.
- **Non-numeric kernel values or negative values**: Non-integer, non-positive, or even values for spatial kernels must be caught and rejected with explicit messaging.
- **Percentiles outside range (0, 100)**: Percentile lists containing values $\le 0$ or $\ge 100$, or identical values (e.g., `[25, 25, 75]`), must be rejected.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST establish the repository layout isolating pure logic in `packages/dermafuzzy_core/`, training workflows in `services/colab_trainer/`, output models/calibration in `artifacts/`, and automated tests in `tests/unit/`.
- **FR-002**: System MUST provide a root configuration file `derma_config.yaml` specifying all default hyperparameters across four distinct sections: `dataset`, `preprocessing`, `biomarkers`, and `mamdani_fis`.
- **FR-003**: System MUST define strongly-typed, immutable dataclasses to represent configuration sections (`DatasetConfig`, `PreprocessingConfig`, `BiomarkersConfig`, `MamdaniFISConfig`, and aggregate `DermaConfig`).
- **FR-004**: The configuration engine MUST load configuration data from YAML files or dictionaries and instantiate the typed configuration hierarchy.
- **FR-005**: The configuration engine MUST validate that dataset `split_ratios` (`train`, `val`, `test`) sum to $1.0 \pm 1e-6$, raising `ValueError` if the condition is violated.
- **FR-006**: The configuration engine MUST validate that biomarker border weights `b_fuzzy_weights` (`w_geo`, `w_grad`) sum to $1.0 \pm 1e-6$, raising `ValueError` if the condition is violated.
- **FR-007**: The configuration engine MUST validate that spatial filtering kernels (`blackhat_kernel_size`, `gaussian_blur_kernel`, `boundary_ring_kernel`) are positive odd integers ($k > 0$ and $k \pmod 2 == 1$), raising `ValueError` if any kernel is even, non-positive, or non-integral.
- **FR-008**: The configuration engine MUST validate that Mamdani FIS `percentiles` are strictly monotonically increasing and lie strictly within the range $(0, 100)$, raising `ValueError` otherwise.
- **FR-009**: The configuration engine MUST validate that `mask_area_bounds` defines a valid lower and upper bound $[min\_bound, max\_bound]$ where $0.0 < min\_bound < max\_bound < 1.0$.
- **FR-010**: The configuration engine MUST validate that biomarker value range is strictly $[0.0, 1.0]$.
- **FR-011**: System MUST provide automated unit tests in `tests/unit/test_config.py` verifying successful configuration parsing, field fidelity, and negative rejection testing for all validation rules.
- **FR-012**: `packages/dermafuzzy_core/config.py` MUST remain a pure Python module with zero top-level imports of heavy deep learning frameworks, complying with Constitution Principle I.

### Key Entities

- **DermaConfig**: Root domain configuration entity holding composite sub-configurations:
  - `dataset`: `DatasetConfig`
  - `preprocessing`: `PreprocessingConfig`
  - `biomarkers`: `BiomarkersConfig`
  - `mamdani_fis`: `MamdaniFISConfig`
- **DatasetConfig**: Dataset acquisition and partitioning configuration:
  - `name`: string identifier (e.g., `"HAM10000"`)
  - `image_size`: list/tuple of two integers `[width, height]` (e.g., `[224, 224]`)
  - `target_classes`: mapping of diagnostic category to lesion code (e.g., `{"malignant": "mel", "benign": "nv"}`)
  - `samples_per_class`: integer count of samples per diagnostic class (e.g., `1000`)
  - `random_state`: integer random seed for reproducibility (e.g., `42`)
  - `split_ratios`: dictionary of split proportions `{"train": 0.70, "val": 0.15, "test": 0.15}`
- **PreprocessingConfig**: Image enhancement and morphological parameters:
  - `blackhat_kernel_size`: positive odd integer for hair removal morphological kernel (e.g., `17`)
  - `inpaint_radius`: positive integer for telea inpainting radius (e.g., `3`)
  - `clahe_clip_limit`: float threshold for CIE-LAB L-channel contrast limit (e.g., `2.0`)
  - `clahe_tile_grid`: list/tuple of two integers for CLAHE grid dimensions (e.g., `[8, 8]`)
  - `gaussian_blur_kernel`: positive odd integer for smoothing kernel (e.g., `5`)
  - `boundary_ring_kernel`: positive odd integer for morphological boundary ring dilation/erosion (e.g., `5`)
  - `mask_area_bounds`: list/tuple of two floats for valid lesion relative area `[0.02, 0.90]`
  - `eps`: float numerical stabilizer for denominators (e.g., `1e-6`)
- **BiomarkersConfig**: Biomarker extraction weights and scales:
  - `b_fuzzy_weights`: mapping of geometric vs gradient border irregularity weights `{"w_geo": 0.5, "w_grad": 0.5}`
  - `feature_names`: ordered list of biomarker identifiers `["A", "B_fuzzy", "C", "D_verified"]`
  - `value_range`: valid bounded range for all biomarkers `[0.0, 1.0]`
- **MamdaniFISConfig**: Fuzzy inference system tuning and output thresholds:
  - `percentiles`: strictly ascending list of calibration percentiles (e.g., `[25, 50, 75]`)
  - `defuzzification_method`: string identifier for defuzzification algorithm (e.g., `"centroid"`)
  - `risk_thresholds`: mapping of risk tier limits `{"low_max": 35.0, "medium_max": 65.0}`

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of valid configuration files load into typed, immutable objects with exact field fidelity in under 50 milliseconds on standard CPU hardware.
- **SC-002**: 100% of invalid configuration states (even kernel sizes, non-unitary split/weight sums, disordered percentiles, invalid area bounds) trigger immediate validation failure with informative diagnostic error messages.
- **SC-003**: Automated test suite in `tests/unit/test_config.py` achieves 100% test pass rate covering both positive configuration scenarios and all boundary condition rejection gates.
- **SC-004**: Loading `packages/dermafuzzy_core/config.py` introduces zero heavy machine learning framework dependencies (`tensorflow`, `keras`, `torch`), maintaining a package import latency under 10 milliseconds.

## Assumptions

- Python standard library dataclasses (`dataclasses.dataclass(frozen=True)`) or equivalent immutable structures are suitable for immutable domain representations.
- `PyYAML` is used for parsing YAML configuration files.
- The project targets Python 3.10+ execution environments.
- Tolerance for floating-point equality checks on sums is fixed to $\epsilon = 1e-6$.
