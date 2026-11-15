# Implementation Plan: Data-Calibrated Mamdani Fuzzy Decision Support System & Metamorphic QA Gate

**Branch**: `003-mamdani-fis-metamorphic-qa` | **Date**: 2026-11-15 | **Spec**: [spec.md](spec.md)  
**Input**: Feature specification from `/specs/003-mamdani-fis-metamorphic-qa/spec.md`  

---

## Summary

Implement Room 4 (`packages/dermafuzzy_core/fis_engine.py`) and Room 6 (`tests/metamorphic/test_metamorphic.py` & `tests/unit/test_fis_engine.py`) to provide an auditable, data-calibrated Mamdani Fuzzy Inference System (FIS). The system calibrates $P_{25}, P_{50}, P_{75}$ cutoffs on training-split distributions with a minimum separation guard ($\delta = 1e-4$), serializes parameters strictly to human-readable JSON (`artifacts/fis_calibration.json`, zero `pickle`), evaluates piecewise-linear membership functions satisfying Partition of Unity ($\mu_L + \mu_M + \mu_H \equiv 1.0$), fires 5 clinical rules (Classic ABCD, Deep-Verified Rim, Early Structural, Borderline Dysplastic, Benign & Artifact Protection) with a Complete Rule Partition Guard, and defuzzifies via the Centroid method over $[0.0, 100.0]$. The entire diagnostic pipeline is rigorously validated against 5 clinical Metamorphic Relations (MR1–MR5).

---

## Technical Context

**Language/Version**: Python 3.14.4  
**Primary Dependencies**: `numpy>=1.24.0`, `scipy>=1.11.0`, `scikit-fuzzy>=0.5.0`, `opencv-python-headless>=4.8.0`, `pyyaml>=6.0`  
**Storage**: JSON file artifact (`artifacts/fis_calibration.json`) with JSON Schema validation and `MappingProxyType` immutability. No databases, zero `pickle`.  
**Testing**: `pytest>=9.0.0` with `pytest-cov`, unit tests in `tests/unit/test_fis_engine.py`, and metamorphic relation tests in `tests/metamorphic/test_metamorphic.py`.  
**Target Platform**: Clinical edge workstation (Intel Core i5-7200U @ 2.50GHz, 8GB RAM, Windows/Linux).  
**Project Type**: Pure algorithmic computational core library (`packages/dermafuzzy_core`).  
**Performance Goals**: Sub-second full pipeline execution: $< 5\text{ms}$ per FIS evaluation, $< 250\text{ms}$ end-to-end image diagnosis on i5-7200U.  
**Constraints**: 
- Constitution Principle I: Zero top-level heavy ML framework imports (`tensorflow`, `keras`, `torch`).
- Constitution Principle II: Strict $[0.0, 100.0]$ bounded risk outputs, $\epsilon = 1e-6$ stabilizers, Partition of Unity.
- Constitution Principle III: Zero pickle files; percentile calibration strictly on training split.
- Constitution Principle IV: Zero magic numbers; parameters loaded from `derma_config.yaml`.  
**Scale/Scope**: HAM10000 training distribution calibration ($N \approx 1400$ balanced samples), real-time single-sample diagnostic inference, batch hypercube metamorphic testing (625 combinations).

---

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Requirement | Plan Conformance | Status |
|---|---|---|---|
| **I. Pure Core & Zero Skew** | All FIS logic in `packages/dermafuzzy_core/fis_engine.py`; zero TensorFlow/Keras/PyTorch imports; runs on i5-7200U 8GB RAM. | Only standard library, `numpy`, `scipy`, and `skfuzzy` used. Zero DL imports. Sub-second execution guaranteed. | **PASS** |
| **II. Strict Mathematical Bounds** | Risk score clipped to $[0.0, 100.0]$; all membership ramps guarded with $\delta=1e-4$ separation; Partition of Unity $\sum \mu = 1.0$. | Analytical piecewise ramps guarantee $\sum \mu \equiv 1.0$; Centroid defuzzification over $z \in [0, 100]$ with complete partition guard ($\sum W_c > 0$). | **PASS** |
| **III. Zero Leakage & No-Pickle** | Percentiles calibrated strictly on training split; persisted to JSON (`artifacts/fis_calibration.json`); zero `.pkl` / `.pickle`. | Calibration functions accept training-only matrix; serializes exclusively via standard `json` with schema validation. Zero pickle. | **PASS** |
| **IV. Config-Driven & Test-First** | Zero magic numbers; risk thresholds from `derma_config.yaml`; comprehensive unit and metamorphic tests (MR1–MR5). | Thresholds (`low_max=35.0`, `medium_max=65.0`) read from `config.mamdani_fis`; 5 Metamorphic Relations in `test_metamorphic.py`. | **PASS** |

---

## Project Structure

### Documentation (this feature)

```text
specs/003-mamdani-fis-metamorphic-qa/
├── spec.md                  # Feature specification
├── plan.md                  # This implementation plan
├── research.md              # Phase 0: Mathematical formulations & MR design
├── data-model.md            # Phase 1: Entity definitions & invariants
├── quickstart.md            # Phase 1: Runnable validation scenarios
├── checklists/
│   └── requirements.md      # Spec quality validation checklist
└── contracts/
    ├── fis-engine-api.md     # Room 4 public API contract
    └── calibration-schema.json # JSON Schema for fis_calibration.json
```

### Source Code (repository root)

```text
packages/dermafuzzy_core/
├── __init__.py              # Re-export Room 1, Room 2, Room 3, and Room 4 symbols
├── config.py                # Room 1: Unified configuration engine
├── preprocessing.py         # Room 2: OpenCV hair removal, CLAHE, segmentation
├── biomarkers.py            # Room 3: ABCD clinical biomarker formulations
├── gradcam.py               # Room 3: Saliency-gated attention verification
└── fis_engine.py            # Room 4: Data-calibrated Mamdani FIS & explanation

artifacts/
└── fis_calibration.json     # Calibrated training percentiles (schema-validated JSON)

tests/
├── unit/
│   ├── phantoms.py          # Synthetic dermoscopic phantom generators
│   ├── test_config.py       # Room 1 unit tests
│   ├── test_preprocessing.py # Room 2 unit tests
│   ├── test_gradcam.py      # Room 3 Grad-CAM tests
│   ├── test_biomarkers.py   # Room 3 ABCD biomarker tests
│   └── test_fis_engine.py   # Room 4 Mamdani FIS unit tests
└── metamorphic/
    ├── __init__.py
    └── test_metamorphic.py  # Room 6 End-to-end Metamorphic Relations (MR1–MR5)
```

**Structure Decision**:
Follows the established 6-room modular architectural layout. Room 4 logic is implemented entirely within `packages/dermafuzzy_core/fis_engine.py`. Room 6 metamorphic evaluation resides in a dedicated `tests/metamorphic/` test room to separate metamorphic diagnostic stability checks from fast unit test suites.

---

## Complexity Tracking

> **No Constitution violations exist. All gates pass cleanly.**

| Checkpoint | Status | Rationale |
|---|---|---|
| Pure Core Dependency Guard | Compliant | No heavy deep learning frameworks are introduced; `scipy` and `scikit-fuzzy` are lightweight numerical libraries. |
| No-Pickle Serialization | Compliant | Standard library `json` format with explicit schema validation guarantees human inspectability and zero remote code execution. |
| Test Coverage & Invariants | Compliant | 5 Metamorphic Relations verify rotational invariance, artifact immunity, monotonicity, and grid completeness. |
