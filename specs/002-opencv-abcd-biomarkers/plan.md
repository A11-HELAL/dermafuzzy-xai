# Implementation Plan: Production OpenCV Preprocessing, Perilesional Boundary Ring & Saliency-Verified ABCD Biomarkers

**Branch**: `002-opencv-abcd-biomarkers` | **Date**: 2026-11-15 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/002-opencv-abcd-biomarkers/spec.md`

## Summary

Implement Room 2 (`packages/dermafuzzy_core/preprocessing.py`) and Room 3 (`packages/dermafuzzy_core/biomarkers.py` and `packages/dermafuzzy_core/gradcam.py`) as 100% pure, deterministic, config-driven scientific modules with read-only NumPy array guarantees (`arr.flags.writeable = False`). Preprocessing removes hair occlusions via Morphological Black-Hat and Telea inpainting, normalizes illumination via CIE-LAB L-channel CLAHE while preserving chromatic channels, segments the lesion using Gaussian smoothing, inverse Otsu thresholding, border vignetting suppression, and LCC extraction, and derives the perilesional boundary ring $R_{edge}$. Biomarker extraction computes rotation-invariant Asymmetry $A$, hybrid Fuzzy Border Irregularity $B_{fuzzy}$, multi-channel Color Variation $C$, and attention-verified deep confidence $D_{verified} = p_{cnn} \cdot \alpha$. Public APIs are cleanly re-exported in `__init__.py`, `pyproject.toml` dependencies are updated, and comprehensive synthetic phantom test suites are established.

## Technical Context

**Language/Version**: Python 3.14 (3.14.4)

**Primary Dependencies**: `numpy` (2.4.4), `opencv-python-headless` (`cv2` 4.13.0), `pyyaml` (6.0.3), Python standard library (`math`, `dataclasses`, `types`, `pathlib`)

**Storage**: In-memory write-protected NumPy arrays (`writeable = False`), immutable frozen dataclasses (`PreprocessingResult`, `BiomarkerResult`)

**Testing**: `pytest` using synthetic geometric phantoms (circle, star, L-shape), blurred edge phantoms, multi-pigment color quadrants, and 2D Grad-CAM attention maps

**Target Platform**: Clinical workstations and edge devices (Intel Core i5-7200U dual-core CPU with 8GB RAM baseline), cross-platform (Windows, Linux, macOS)

**Project Type**: Pure algorithmic core scientific library

**Performance Goals**: End-to-end preprocessing + ABCD extraction < 100ms per 224x224 image, memory footprint < 10MB per sample

**Constraints**:
- Zero top-level heavy ML framework imports (`tensorflow`, `keras`, `torch`) in `packages/dermafuzzy_core/` (Constitution Principle I).
- CLAHE illumination normalization strictly restricted to `L` channel in CIE-LAB; `a` and `b` channels must remain bit-identical (Constitution Principle II).
- All biomarker values strictly normalized and clipped to $[0.0, 1.0]$ with $\epsilon = 1e-6$ denominator stabilizer (Constitution Principle II).
- Zero `pickle` or `.pkl` files permitted repository-wide (Constitution Principle III).
- All hyperparameters driven by root `derma_config.yaml` with zero magic numbers (Constitution Principle IV).

**Scale/Scope**: 3 core library modules (`preprocessing.py`, `biomarkers.py`, `gradcam.py`), 2 immutable domain result dataclasses, 3 unit test suites.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle / Gate | Assessment | Status |
|------------------|------------|--------|
| **I. Pure Core & Zero Training-Serving Skew** | All preprocessing and biomarker routines implemented as pure functions in `packages/dermafuzzy_core/`. Zero imports of `tensorflow`, `keras`, or `torch`. | **PASS** |
| **II. Strict Mathematical & Domain Bounds** | CLAHE applied strictly to `L` channel in CIE-LAB. $A, B_{fuzzy}, C, D_{verified}$ clipped to $[0.0, 1.0]$ with $\epsilon = 1e-6$ denominator stabilizer. All output arrays write-protected. | **PASS** |
| **III. Zero Data Leakage & No-Pickle Security Policy** | In-memory NumPy arrays and immutable dataclasses exclusively; zero Python `pickle` or unvalidated object deserializers. | **PASS** |
| **IV. Config-Driven & Test-First Gate** | Kernels, clip limits, weights, and stabilizers loaded from `derma_config.yaml`. Verified via synthetic phantoms testing rotation invariance ($|\Delta A| \le 0.03$) and artifact immunity. | **PASS** |
| **Hardware Baseline (i5-7200U, 8GB RAM)** | Vectorized NumPy and OpenCV operations execute in < 100ms with negligible RAM footprint (< 10MB per sample). | **PASS** |

## Project Structure

### Documentation (this feature)

```text
specs/002-opencv-abcd-biomarkers/
├── plan.md              # This file (/speckit-plan command output)
├── research.md          # Phase 0 output (/speckit-plan command)
├── data-model.md        # Phase 1 output (/speckit-plan command)
├── quickstart.md        # Phase 1 output (/speckit-plan command)
├── contracts/           # Phase 1 output (/speckit-plan command)
│   ├── preprocessing-api.md
│   ├── biomarkers-api.md
│   └── gradcam-api.md
├── checklists/
│   └── requirements.md
└── tasks.md             # Phase 2 output (/speckit-tasks command - NOT created by /speckit-plan)
```

### Source Code (repository root)

```text
dermafuzzy-xai/
├── derma_config.yaml
├── pyproject.toml                    # Updated dependencies: numpy, opencv-python-headless
├── packages/
│   └── dermafuzzy_core/
│       ├── __init__.py               # Re-exports PreprocessingResult, BiomarkerResult, etc.
│       ├── config.py                 # Unified Configuration Engine (Spec 001)
│       ├── preprocessing.py          # Room 2: OpenCV Preprocessing & Perilesional Ring
│       ├── biomarkers.py             # Room 3: ABCD Biomarker Extraction
│       └── gradcam.py                # Room 3: Saliency Attention Verification
└── tests/
    └── unit/
        ├── test_config.py            # Spec 001 unit tests
        ├── test_preprocessing.py     # Room 2 unit & phantom tests
        ├── test_biomarkers.py        # Room 3 ABCD unit & phantom tests
        └── test_gradcam.py           # Room 3 Grad-CAM verification tests
```

**Structure Decision**: Modular multi-room layout expanding `packages/dermafuzzy_core/` with pure image processing and biomarker modules, tested via dedicated synthetic phantom suites in `tests/unit/`.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

*No constitutional violations. Zero complexity exceptions requested.*
