<!--
Sync Impact Report:
- Version change: Uninitialized template -> 1.0.0
- List of modified principles:
  - [PRINCIPLE_1_NAME] -> I. Pure Core & Zero Training-Serving Skew
  - [PRINCIPLE_2_NAME] -> II. Strict Mathematical & Domain Bounds
  - [PRINCIPLE_3_NAME] -> III. Zero Data Leakage & No-Pickle Security Policy
  - [PRINCIPLE_4_NAME] -> IV. Config-Driven & Test-First Gate
  - [PRINCIPLE_5_NAME] -> Removed (consolidated into 4 domain-specific non-negotiable principles)
- Added sections:
  - Hardware & Security Constraints
  - Metamorphic Quality Gates & Verification Workflow
- Removed sections:
  - None
- Follow-up TODOs:
  - None (all placeholders resolved)
-->

# DermaFuzzy-XAI Constitution

## Core Principles

### I. Pure Core & Zero Training-Serving Skew
All OpenCV preprocessing, ABCD biomarker mathematical formulations, Grad-CAM spatial verification, and Mamdani Fuzzy Inference System (FIS) reasoning logic MUST reside exclusively in `packages/dermafuzzy_core/` as 100% pure, deterministic Python functions.
- `packages/dermafuzzy_core/` MUST NEVER import heavy deep learning frameworks (TensorFlow, Keras, PyTorch) at the module top-level, and MUST NEVER make network requests or database calls.
- Deep learning inferencing (feature extraction / Grad-CAM heatmap generation) MUST be decoupled via clean interfaces, passing raw tensors or numpy arrays into pure verification functions.
- Rationale: Guarantees zero training-serving skew, reproducible edge inference, sub-second execution on constrained clinical workstations (specifically Intel i5-7200U with 8GB RAM), and zero environment dependency hell during core algorithmic testing.

### II. Strict Mathematical & Domain Bounds
Every diagnostic biomarker output—Asymmetry ($A$), Fuzzy Border Irregularity ($B_{fuzzy}$), Multi-Channel Color Variation ($C$), and Attention-Verified Deep Confidence ($D_{verified}$)—MUST be strictly normalized and bounded to $[0.0, 1.0]$ using `np.clip(val, 0.0, 1.0)`.
- All biomarker ratio and metric denominators MUST add an epsilon stabilizer ($\epsilon = 1e-6$) to definitively prevent `ZeroDivisionError`, `NaN`, or `Inf` states across edge-case masks (e.g., empty or single-pixel lesion regions).
- Contrast Limited Adaptive Histogram Equalization (CLAHE) for lighting and color normalization MUST be applied strictly to the luminance ($L$) channel in CIE-LAB color space (`clipLimit=2.0`, `tileGridSize=(8,8)`). CLAHE MUST NEVER be applied directly to RGB channels, preserving native chromatic ratios.
- Rationale: Unbounded values or mathematical anomalies corrupt fuzzy membership calculations and trigger silent diagnostic failure. Direct RGB histogram equalization distorts natural pigmentation boundaries critical for clinical color variation scoring.

### III. Zero Data Leakage & No-Pickle Security Policy
Statistical distribution parameters and percentiles ($P_{25}, P_{50}, P_{75}$) governing Mamdani membership functions MUST be calibrated strictly and exclusively on the training split. Calibration values MUST be serialized to human-readable, schema-validated JSON (`artifacts/fis_calibration.json`).
- Python `pickle` serialization (including `.pkl`, `.pickle`, `joblib` dump of unvalidated objects) is strictly forbidden across the entire repository.
- Model checkpoints, weights, and configuration artifacts MUST utilize secure, inspectable formats (`.json`, `.yaml`, SafeTensors, or standard ONNX/HDF5 where isolated).
- Rationale: Calculating membership bounds over whole datasets introduces subtle data leakage, producing overly optimistic validation results. Banning `pickle` eliminates arbitrary remote code execution attack vectors in clinical environments and ensures human-auditable calibration artifacts.

### IV. Config-Driven & Test-First Gate
Zero magic numbers are permitted in application or core code. All filter kernels, structural elements, fuzzy membership trapezoid/triangle vertices, loss weights, and clinical classification thresholds MUST be loaded from the root configuration file `derma_config.yaml`.
- Test-First Gate: Every package and module MUST achieve comprehensive automated test coverage via `pytest`. Pull requests MUST NOT be merged without passing unit tests and metamorphic quality assurance gates.
- Metamorphic QA Relations: The test suite MUST explicitly evaluate metamorphic diagnostic invariance:
  1. *Rotation Invariance*: Rotating an input dermoscopic image by 90°, 180°, or 270° MUST yield ABCD biomarker scores invariant within a tight numerical tolerance ($\Delta \le 0.02$).
  2. *Artifact Immunity*: The presence of synthetic or natural peripheral artifacts (e.g., dermoscopic gel bubbles, hair lines, or measurement rulers) MUST NOT distort lesion segmentation or shift biomarker values by more than $\Delta \le 0.01$.
- Rationale: Hardcoded heuristics destroy scientific reproducibility and prevent clinician-guided parameter tuning. Metamorphic testing ensures clinical robustness against pervasive real-world acquisition variances.

## Hardware & Security Constraints

### Compute and Memory Footprint
- Target Hardware Baseline: The entire inference and explainability pipeline MUST execute reliably on a standard host equipped with an Intel Core i5-7200U dual-core CPU and 8GB RAM.
- Memory Budget: Single-sample end-to-end processing MUST maintain a peak memory footprint below 1.5 GB RAM. Memory leaks, unbounded cache structures, and excessive batch copies in `packages/dermafuzzy_core/` are considered critical defects.
- Offline Capability: The core package MUST be capable of functioning completely air-gapped without external network, telemetry, or API calls.

### Secure Storage & Serialization Standards
- Human-Readable Artifacts: All calibration matrices, rule definitions, and metric summaries MUST be encoded in human-readable JSON or YAML format.
- Schema Integrity: Any serialized artifact (such as `artifacts/fis_calibration.json`) MUST be accompanied by explicit validation schemas before loading into memory.

## Metamorphic Quality Gates & Verification Workflow

### Pre-Merge Verification Requirements
1. **Linting and Type Safety**: Static typing enforced via `mypy` or `pyright`; zero unchecked dynamic types in `packages/dermafuzzy_core/`.
2. **Deterministic Unit Tests**: `pytest` execution verifying exact bounded outputs $[0.0, 1.0]$ and epsilon division safety across all edge cases (black images, white images, single-pixel segmentations).
3. **Metamorphic Validation**: Automated regression runs asserting rotation invariance and ruler artifact immunity before any release candidate tag.
4. **Configuration Consistency**: Automated validation that all parameters consumed in code exist with documented schemas in `derma_config.yaml`.

## Governance

- **Supremacy**: This Constitution represents the supreme architectural and engineering policy of the DermaFuzzy-XAI repository. Any pull request, design artifact, or implementation task conflicting with these principles is rejected by default.
- **Amendment Procedure**: Amendments require formal documentation via an RFC issue, demonstrable validation that the amendment does not violate memory limits or introduce data leakage, explicit approval by the lead maintainers, and an accompanying migration plan.
- **Versioning Policy**: The Constitution strictly adheres to Semantic Versioning (`MAJOR.MINOR.PATCH`):
  - **MAJOR**: Breaking changes to non-negotiable principles, fundamental pipeline re-architecture, or relaxation of core constraints.
  - **MINOR**: Addition of new core principles, new metamorphic relations, or substantial procedural extensions.
  - **PATCH**: Typographical fixes, phrasing clarifications, or non-semantic documentation enhancements.
- **Compliance Audit**: Every specification (`spec.md`), architectural plan (`plan.md`), and task definition (`tasks.md`) generated under Spec Kit MUST explicitly state compliance with this Constitution.

**Version**: 1.0.0 | **Ratified**: 2026-11-14 | **Last Amended**: 2026-11-14
