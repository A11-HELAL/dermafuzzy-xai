# Implementation Plan: Walking Skeleton & Unified Configuration Engine

**Branch**: `001-walking-skeleton-config` | **Date**: 2026-11-15 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/001-walking-skeleton-config/spec.md`

## Summary

Establish the foundational 6-room modular repository architecture for DermaFuzzy-XAI, create the root `derma_config.yaml` configuration file, implement the pure Python domain modeling and validation engine in `packages/dermafuzzy_core/config.py` using Python 3.14 standard library `dataclasses(frozen=True)` and `pathlib.Path` with PyYAML's `yaml.safe_load`, configure `pyproject.toml` with `pythonpath = ["packages"]` for clean `pytest` test discovery, and implement the deterministic unit test suite in `tests/unit/test_config.py`.

## Technical Context

**Language/Version**: Python 3.14 (active runtime: Python 3.14.4)

**Primary Dependencies**: Python standard library (`dataclasses`, `pathlib`, `math`, `typing`), `PyYAML` (`yaml.safe_load`)

**Storage**: Human-readable configuration files (`derma_config.yaml`), serialized schema/JSON contracts

**Testing**: `pytest` using `tmp_path` fixture for configuration mutation and negative testing

**Target Platform**: Clinical workstations and edge devices (Intel Core i5-7200U dual-core CPU with 8GB RAM baseline), cross-platform (Windows, Linux, macOS)

**Project Type**: Modular algorithmic core package with peripheral training service and verification harness

**Performance Goals**: Configuration loading latency < 50ms, module import latency < 10ms, memory footprint < 5MB

**Constraints**:
- Zero top-level heavy ML framework imports (`tensorflow`, `keras`, `torch`, `cv2`) in `packages/dermafuzzy_core/config.py` (Constitution Principle I).
- Zero `pickle` or `.pkl` files permitted repository-wide (Constitution Principle III).
- All kernels, weights, and thresholds driven by root `derma_config.yaml` with zero magic numbers (Constitution Principle IV).

**Scale/Scope**: 6-room modular directory structure, 5 domain dataclasses, 4 configuration sections, 100% test coverage for validation rules.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle / Gate | Assessment | Status |
|------------------|------------|--------|
| **I. Pure Core & Zero Training-Serving Skew** | `dermafuzzy_core/config.py` is 100% pure Python; uses only stdlib + PyYAML. Zero top-level heavy framework imports or network/database calls. | **PASS** |
| **II. Strict Mathematical & Domain Bounds** | Enforces odd positive kernels, float sum tolerances $\sum = 1.0 \pm 1e-6$ for split ratios and biomarker weights, monotonically increasing percentiles in $(0, 100)$, and normalized bounds. | **PASS** |
| **III. Zero Data Leakage & No-Pickle Security Policy** | PyYAML `yaml.safe_load` exclusively used; no Python `pickle` or unvalidated object deserialization. | **PASS** |
| **IV. Config-Driven & Test-First Gate** | Eliminates magic numbers by loading all hyperparameters from root `derma_config.yaml`; gated by automated `pytest` test suite with negative testing. | **PASS** |
| **Hardware Baseline (i5-7200U, 8GB RAM)** | Dataclasses with `slots=True` maintain minimal CPU/memory footprint (<5MB memory, <50ms execution). | **PASS** |

## Project Structure

### Documentation (this feature)

```text
specs/001-walking-skeleton-config/
├── plan.md              # This file (/speckit-plan command output)
├── research.md          # Phase 0 output (/speckit-plan command)
├── data-model.md        # Phase 1 output (/speckit-plan command)
├── quickstart.md        # Phase 1 output (/speckit-plan command)
├── contracts/           # Phase 1 output (/speckit-plan command)
│   ├── config-schema.json
│   └── config-api.md
├── checklists/
│   └── requirements.md
└── tasks.md             # Phase 2 output (/speckit-tasks command - NOT created by /speckit-plan)
```

### Source Code (repository root)

```text
dermafuzzy-xai/
├── derma_config.yaml                 # Root configuration file (single source of truth)
├── pyproject.toml                    # Modern packaging configuration & pytest pythonpath
├── packages/
│   └── dermafuzzy_core/             # Room 1: Pure deterministic algorithmic core
│       ├── __init__.py
│       └── config.py                 # Immutable dataclasses & validation engine
├── services/
│   └── colab_trainer/               # Room 2: Model training & Grad-CAM pipeline
├── artifacts/                        # Room 3: Human-readable JSON calibrations & weights
└── tests/
    └── unit/                         # Room 4: Deterministic unit tests & metamorphic QA
        └── test_config.py            # Configuration loading & rejection tests
```

**Structure Decision**: A decoupled 6-room modular architecture separating pure domain logic (`packages/dermafuzzy_core/`) from training pipelines (`services/colab_trainer/`), test suites (`tests/unit/`), and serialized artifacts (`artifacts/`), fully adhering to the DermaFuzzy-XAI Constitution.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

*No constitutional violations. Zero complexity exceptions requested.*
