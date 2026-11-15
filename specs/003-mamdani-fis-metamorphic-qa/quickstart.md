# Quickstart Validation Guide: Mamdani Fuzzy Decision Support & Metamorphic QA

**Feature**: [003-mamdani-fis-metamorphic-qa](spec.md) | **Date**: 2026-11-15  

---

## 1. Prerequisites Check
Verify that `numpy`, `scipy`, `skfuzzy`, and `yaml` are available:
```powershell
python -c "import numpy, scipy, skfuzzy, yaml; print('All Room 4 dependencies ready!')"
```

---

## 2. Percentile Calibration & JSON Round-Trip Verification
Calibrate percentiles on synthetic training distributions and verify JSON serialization:
```powershell
python -c "
import numpy as np
from dermafuzzy_core.fis_engine import calibrate_fis_percentiles, save_calibration_json, load_calibration_json

# Generate synthetic training split
np.random.seed(42)
train_data = np.random.beta(2, 5, size=(100, 4))
calib = calibrate_fis_percentiles(train_data)
path = save_calibration_json(calib, 'artifacts/fis_calibration.json')
reloaded = load_calibration_json(path)
print('Calibrated cutoffs for A:', reloaded.get_cutoffs('A'))
print('JSON round-trip identical:', calib.cutoffs == reloaded.cutoffs)
"
```

---

## 3. Partition of Unity & Worked Reference Example
Verify Partition of Unity ($\mu_L + \mu_M + \mu_H = 1.0$) and the study guide benchmark case:
```powershell
python -c "
import numpy as np
from dermafuzzy_core.fis_engine import FISCalibration, compute_memberships, evaluate_mamdani_fis

# Standard quartiles calibration
cutoffs = {feat: (0.25, 0.50, 0.75) for feat in ('A', 'B_fuzzy', 'C', 'D_verified')}
calib = FISCalibration.from_dict({'cutoffs': cutoffs, 'n_samples': 100, 'generated_at': '2026-11-15T00:00:00Z', 'config_hash': '0'*64})

# Test Partition of Unity on 100 points
test_x = np.linspace(0.0, 1.0, 100)
for x in test_x:
    m = compute_memberships(x, (0.25, 0.50, 0.75))
    assert abs(1.0 - (m.low + m.medium + m.high)) < 1e-6

# Worked Reference Example (0.68, 0.74, 0.81, 0.80) -> High Risk [80%-88%]
pred = evaluate_mamdani_fis((0.68, 0.74, 0.81, 0.80), calib)
print('Worked Example Risk Score:', pred.risk_score, '%')
print('Risk Category:', pred.risk_category)
print('Dominant Rule:', pred.dominant_rule_id, '-', pred.dominant_rule_text)
assert pred.risk_category == 'High' and 80.0 <= pred.risk_score <= 88.0
"
```

---

## 4. Peripheral Ruler Artifact Protection Verification
Verify that a benign nevus corrupted by an off-lesion ruler artifact classifies as `Low Risk`:
```powershell
python -c "
from dermafuzzy_core.fis_engine import FISCalibration, evaluate_mamdani_fis
cutoffs = {feat: (0.25, 0.50, 0.75) for feat in ('A', 'B_fuzzy', 'C', 'D_verified')}
calib = FISCalibration.from_dict({'cutoffs': cutoffs, 'n_samples': 100, 'generated_at': '2026-11-15T00:00:00Z', 'config_hash': '0'*64})

# Low ABCD features, p_cnn=0.92, alpha=0.12 -> D_verified = 0.1104
pred = evaluate_mamdani_fis((0.08, 0.10, 0.05, 0.1104), calib)
print('Artifact Protection Risk Score:', pred.risk_score, '%')
print('Risk Category:', pred.risk_category)
print('Dominant Rule:', pred.dominant_rule_id, '-', pred.dominant_rule_text)
assert pred.risk_category == 'Low' and pred.risk_score < 35.0
assert pred.dominant_rule_id == 'R5'
"
```

---

## 5. Automated Test Suite Execution
Run both unit tests and end-to-end metamorphic relations:
```powershell
pytest -v tests/unit/test_fis_engine.py tests/metamorphic/test_metamorphic.py
```
