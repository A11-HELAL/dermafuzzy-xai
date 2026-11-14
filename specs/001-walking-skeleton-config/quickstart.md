# Quickstart Validation Guide: Walking Skeleton & Unified Configuration Engine

**Feature**: [001-walking-skeleton-config](spec.md) | **Date**: 2026-11-15

## Overview
This guide provides runnable instructions to verify that the Walking Skeleton repository layout, root `derma_config.yaml`, pure configuration loader `packages/dermafuzzy_core/config.py`, and test suite `tests/unit/test_config.py` function as expected.

---

## 1. Prerequisites
- Python 3.10+ (Current runtime: Python 3.14.4)
- Virtual environment activated with `pytest` and `PyYAML` installed:
  ```powershell
  python -c "import yaml, pytest; print('Environment ready')"
  ```

---

## 2. Directory Layout Verification
Verify that all 6 modular rooms and configuration files exist:
```powershell
# Check root configuration and packaging
Test-Path derma_config.yaml
Test-Path pyproject.toml

# Check core package room
Test-Path packages/dermafuzzy_core/__init__.py
Test-Path packages/dermafuzzy_core/config.py

# Check trainer room, artifacts room, and test room
Test-Path services/colab_trainer
Test-Path artifacts
Test-Path tests/unit/test_config.py
```

---

## 3. Configuration Loading Verification
Test loading and immutability directly from Python:
```powershell
python -c "from dermafuzzy_core.config import load_config; cfg = load_config(); print('Loaded config dataset:', cfg.dataset.name); print('Image size:', cfg.dataset.image_size)"
```

Expected Output:
```text
Loaded config dataset: HAM10000
Image size: (224, 224)
```

Verify immutability (shallow & deep):
```powershell
python -c "from dermafuzzy_core.config import load_config; cfg = load_config(); cfg.dataset.name = 'OTHER'"
# Output: dataclasses.FrozenInstanceError: cannot assign to field 'name'

python -c "from dermafuzzy_core.config import load_config; cfg = load_config(); cfg.dataset.target_classes['malignant'] = 'OTHER'"
# Output: TypeError: 'mappingproxy' object does not support item assignment
```

---

## 4. Automated Test Suite Execution
Run the complete unit test suite via `pytest`:
```powershell
pytest -v tests/unit/test_config.py
```

Expected Output:
```text
tests/unit/test_config.py::test_load_config_valid PASSED
tests/unit/test_config.py::test_config_immutability PASSED
tests/unit/test_config.py::test_rejects_even_kernel_sizes PASSED
tests/unit/test_config.py::test_rejects_invalid_split_ratios PASSED
tests/unit/test_config.py::test_rejects_invalid_b_fuzzy_weights PASSED
tests/unit/test_config.py::test_rejects_disordered_percentiles PASSED
tests/unit/test_config.py::test_zero_heavy_imports PASSED
============================== N passed in X.XXs ==============================
```

---

## 5. Pure Core Import Guard Verification
Ensure importing `dermafuzzy_core.config` does not import any heavy ML frameworks:
```powershell
python -c "import sys; import dermafuzzy_core.config; assert 'tensorflow' not in sys.modules; assert 'torch' not in sys.modules; assert 'cv2' not in sys.modules; print('Pure core import verified!')"
```

Expected Output:
```text
Pure core import verified!
```
