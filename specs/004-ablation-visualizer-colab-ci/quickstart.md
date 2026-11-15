# Quickstart Validation Guide: Clinical Evaluation, 3-Tier Ablation & 4-Panel XAI Visualizer

**Feature**: [004-ablation-visualizer-colab-ci](spec.md) | **Date**: 2026-11-15  

---

## 1. Prerequisites Check
Verify that `matplotlib`, `numpy`, `scipy`, `skfuzzy`, and `cv2` are available:
```powershell
python -c "import matplotlib, numpy, scipy, skfuzzy, cv2; print('All Room 5 dependencies ready! Matplotlib version:', matplotlib.__version__)"
```

---

## 2. Clinical Metrics & Pure NumPy AUC-ROC Verification
Compute confusion matrix, sensitivity, specificity, and exact AUC-ROC:
```powershell
python -c "
import numpy as np
from dermafuzzy_core.evaluation import compute_clinical_metrics

y_true = np.array([1, 1, 1, 0, 0, 0])
y_prob = np.array([0.95, 0.85, 0.70, 0.30, 0.20, 0.10])
metrics = compute_clinical_metrics(y_true, y_prob, threshold=0.50)
print('Accuracy:', metrics.accuracy)
print('Sensitivity:', metrics.sensitivity)
print('Specificity:', metrics.specificity)
print('AUC-ROC:', metrics.auc_roc)
assert metrics.accuracy == 1.0 and metrics.auc_roc == 1.0
"
```

---

## 3. 3-Tier Ablation Study Execution
Run comparative ablation study across Pure CNN vs. Crisp ABCD vs. DermaFuzzy-XAI:
```powershell
python -c "
import numpy as np
from dermafuzzy_core.fis_engine import load_calibration_json
from dermafuzzy_core.evaluation import run_ablation_study

calib = load_calibration_json('artifacts/fis_calibration.json')

# 4 synthetic patients: 2 malignant, 1 benign, 1 benign with off-lesion ruler artifact
y_true = np.array([1, 1, 0, 0])
p_cnn_raw = np.array([0.88, 0.92, 0.15, 0.90])  # Note: patient 4 has spurious high CNN
biomarkers = np.array([
    [0.70, 0.75, 0.80, 0.85],  # Malignant
    [0.65, 0.70, 0.75, 0.88],  # Malignant
    [0.05, 0.05, 0.05, 0.08],  # Clean Benign
    [0.08, 0.10, 0.05, 0.11],  # Benign with ruler (D_verified suppressed to 0.11)
])

report = run_ablation_study(y_true, p_cnn_raw, biomarkers, calib)
print(report.to_markdown_table())
assert report.tiers['3_proposed_dermafuzzy_xai'].specificity > report.tiers['1_pure_mobilenetv2_blackbox'].specificity
"
```

---

## 4. 4-Panel Clinical XAI Visualizer Verification
Render a 300 DPI 4-panel audit figure and verify read-only array output:
```powershell
python -c "
import numpy as np
from dermafuzzy_core.fis_engine import load_calibration_json, diagnose_image
from dermafuzzy_core.visualizer import render_four_panel_report
from tests.unit.phantoms import make_circle_phantom, make_gradcam_heatmap

calib = load_calibration_json('artifacts/fis_calibration.json')
rgb, mask = make_circle_phantom(size=(224, 224), radius=50)
heatmap = make_gradcam_heatmap(size=(224, 224), on_lesion=True)
prep_res, bio_res, fis_pred = diagnose_image(rgb, 0.85, heatmap, calib)

fig_rgb = render_four_panel_report(prep_res, bio_res, fis_pred, heatmap, output_path='artifacts/sample_4panel_xai.png', dpi=150)
print('4-Panel Rendered Shape:', fig_rgb.shape)
print('Read-Only Array Guarantee:', fig_rgb.flags.writeable == False)
assert fig_rgb.ndim == 3 and fig_rgb.shape[2] == 3 and fig_rgb.flags.writeable == False
"
```

---

## 5. Colab Trainer Synthetic Dry-Run Verification
Execute the local smoke-test of the Colab HAM10000 training script:
```powershell
python services/colab_trainer/train_ham10000.py --dry-run-synthetic
```
