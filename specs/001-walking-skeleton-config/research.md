# Research & Architecture Decisions: Walking Skeleton & Unified Configuration Engine

**Feature**: [001-walking-skeleton-config](spec.md) | **Date**: 2026-11-15

## Overview
This document records technical research, evaluation of alternatives, and architectural decisions made for the unified configuration engine and walking skeleton layout of DermaFuzzy-XAI.

---

## Decision 1: Pure Domain Modeling via Python Standard Library Dataclasses

### Context
The configuration engine requires strongly typed, structured objects representing hyperparameters across dataset, preprocessing, biomarkers, and fuzzy logic. Downstream consumers must not mutate configuration at runtime.

### Decision
Use Python standard library `@dataclass(frozen=True, slots=True)` with type annotations and explicit validation methods.

### Rationale
- **Zero External Overhead**: Included in Python 3.14 standard library; requires zero third-party dependencies in `packages/dermafuzzy_core/config.py`.
- **Guaranteed Immutability**: `frozen=True` prevents accidental runtime attribute reassignment during multi-stage processing.
- **Performance & Footprint**: `slots=True` provides compact memory layout and fast attribute access (<10ms import time, <50ms parsing time).
- **Compliance**: Fully complies with Constitution Principle I (pure, lightweight core) and Principle IV (config-driven gate).

### Alternatives Considered
- **Pydantic (`BaseModel`)**: Excellent validation ecosystem, but adds a heavy third-party dependency with runtime overhead and potential version incompatibilities across environments.
- **`attrs`**: High feature density, but introduces an external dependency when Python's built-in `dataclasses` already satisfy all requirements.
- **Plain `dict` / `TypedDict`**: Mutable by default and lacks runtime constraint validation without ad-hoc checks.

---

## Decision 2: YAML Parsing via Safe Loading (`yaml.safe_load`)

### Context
Root configuration is defined in human-readable YAML (`derma_config.yaml`). The parser must be secure and resistant to arbitrary code execution.

### Decision
Use PyYAML's `yaml.safe_load` exclusively, paired with `pathlib.Path` for path resolution.

### Rationale
- **Security**: `yaml.safe_load` restricts parsing to standard YAML scalars, sequences, and mappings, completely eliminating arbitrary Python object instantiation and code execution vectors.
- **Constitution Compliance**: Directly aligns with Constitution Principle III (strict no-pickle and secure serialization policy).
- **Readability**: YAML provides hierarchical, commented human-readable structure for clinical and machine learning practitioners.

### Alternatives Considered
- **`yaml.load(..., Loader=yaml.Loader)`**: Highly vulnerable to arbitrary code execution via Python object tags; strictly rejected.
- **`tomli` / `tomllib` (TOML)**: Native in Python 3.11+, but YAML is significantly more readable for multi-level nested arrays, dictionaries, and floating-point scientific hyperparameters.
- **JSON**: Less readable for configuration files due to lack of native comment support.

---

## Decision 3: Package Import Resolution via `pyproject.toml`

### Context
Tests in `tests/unit/` need to import `dermafuzzy_core` cleanly without requiring manual installation (`pip install -e .`) or brittle `sys.path.insert` hacks in test files.

### Decision
Configure standard `pyproject.toml` with `[tool.pytest.ini_options]` setting `pythonpath = ["packages"]`.

### Rationale
- **Standards-Compliant**: Modern `pytest` natively honors `tool.pytest.ini_options.pythonpath`, making `packages/dermafuzzy_core` immediately discoverable.
- **Zero Side-Effects**: Developers and CI can run `pytest` directly from the repository root without environment mutation.
- **Foundation for Packaging**: `pyproject.toml` serves as the single declarative metadata file for Python 3.14 packaging standards.

### Alternatives Considered
- **`sys.path.append(str(Path(__file__).parents[2] / 'packages'))`**: Pollutes test files with boilerplate and breaks when test files are moved.
- **`setup.py` / `setup.cfg`**: Deprecated packaging approaches replaced by PEP 517/518 and PEP 621.

---

## Decision 4: Mathematical Validation and Domain Constraints

### Context
Hyperparameters must satisfy strict physical, geometric, and numerical constraints before any image processing or fuzzy inference is triggered.

### Decision
Implement explicit validation gates inside `validate_config()` enforcing:
1. **Floating-point Sum Normalization**: $\sum split\_ratios = 1.0 \pm 1e-6$ and $\sum b\_fuzzy\_weights = 1e-6$ validated via `math.isclose(total, 1.0, abs_tol=1e-6)`.
2. **Spatial Kernel Geometries**: `blackhat_kernel_size`, `gaussian_blur_kernel`, and `boundary_ring_kernel` must be positive odd integers ($k > 0 \land k \pmod 2 == 1$).
3. **Mamdani FIS Monotonic Percentiles**: `percentiles` must be strictly increasing and bounded within $(0, 100)$ ($0 < p_1 < p_2 < \dots < 100$).
4. **Mask Area Bounds**: $0.0 < min\_area < max\_area < 1.0$.

### Rationale
- Centered OpenCV morphological and blurring operations require odd kernel sizes to establish a well-defined integer center pixel $(k//2, k//2)$. Even kernels cause spatial asymmetry and OpenCV assertion errors.
- Float comparison with tolerance prevents precision artifacts (e.g., `0.70 + 0.15 + 0.15 = 0.9999999999999999`).
- Strict monotonicity in Mamdani percentiles ensures valid fuzzy set partition boundaries without degenerate trapezoidal/triangular shapes.

### Alternatives Considered
- **Implicit validation during OpenCV / NumPy execution**: Errors occur late during pipeline execution, resulting in cryptic OpenCV C++ crashes instead of clear domain diagnostics.
- **Strict float equality (`== 1.0`)**: Fragile due to binary floating-point representation quirks.
