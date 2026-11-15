# Quickstart Validation Guide: Production OpenCV Preprocessing & Saliency-Verified ABCD Biomarkers

**Feature**: [002-opencv-abcd-biomarkers](spec.md) | **Date**: 2026-11-15

## Overview
This guide provides runnable scenarios to verify Room 2 (`preprocessing.py`) and Room 3 (`biomarkers.py` and `gradcam.py`) in `packages/dermafuzzy_core/`.

---

## 1. Prerequisites
Verify that Python 3.14+, `opencv-python-headless`, `numpy`, and `pytest` are available:
```powershell
python -c "import cv2, numpy, yaml, pytest; print('Dependencies ready: cv2', cv2.__version__, 'numpy', numpy.__version__)"
```

---

## 2. Preprocessing & Deep Array Immutability Verification
Run a verification script testing synthetic hair removal, CLAHE, segmentation, and write-protection:
```powershell
python -c "import numpy as np; from dermafuzzy_core.preprocessing import preprocess_image; img = np.full((224, 224, 3), 180, dtype=np.uint8); cv2_circle = True; res = preprocess_image(img); print('Preprocessing executed! Fallback used:', res.fallback_used); print('Array writeable flags:', res.clean_rgb.flags.writeable, res.lesion_mask.flags.writeable)"
```

Expected Output:
```text
Preprocessing executed! Fallback used: True
Array writeable flags: False False
```

Verify write-protection exception:
```powershell
python -c "import numpy as np; from dermafuzzy_core.preprocessing import preprocess_image; img = np.full((224, 224, 3), 180, dtype=np.uint8); res = preprocess_image(img); res.clean_rgb[0, 0, 0] = 0"
```

Expected Output:
```text
ValueError: assignment destination is read-only
```

---

## 3. ABCD Feature Extraction Verification
Extract ABCD features from a synthetic circular lesion phantom:
```powershell
python -c "import numpy as np, cv2; from dermafuzzy_core.biomarkers import extract_abcd_biomarkers; mask = np.zeros((224, 224), dtype=np.uint8); cv2.circle(mask, (112, 112), 50, 1, -1); ring = cv2.dilate(mask, np.ones((5, 5), np.uint8)) - cv2.erode(mask, np.ones((5, 5), np.uint8)); rgb = np.full((224, 224, 3), 200, dtype=np.uint8); rgb[mask > 0] = [80, 50, 40]; res = extract_abcd_biomarkers(rgb, mask, ring, p_cnn=0.85, heatmap=mask.astype(float)); print('ABCD vector:', res.as_vector()); print('Alpha:', res.alpha); print('D_verified:', res.D_verified)"
```

Expected Output:
```text
ABCD vector: (0.0..., 0.0..., 0.0..., 0.85)
Alpha: 1.0
D_verified: 0.85
```

---

## 4. Grad-CAM Automatic 7x7 Upsampling Verification
Verify that a low-resolution $7 \times 7$ activation map correctly interpolates and calculates on-lesion overlap:
```powershell
python -c "import numpy as np; from dermafuzzy_core.gradcam import compute_attention_overlap, compute_verified_confidence; mask = np.zeros((224, 224), dtype=np.uint8); mask[56:168, 56:168] = 1; hm7 = np.zeros((7, 7), dtype=np.float32); hm7[2:5, 2:5] = 1.0; alpha = compute_attention_overlap(hm7, mask); d_ver = compute_verified_confidence(0.90, alpha); print('Alpha from 7x7:', alpha); print('Verified confidence:', d_ver)"
```

Expected Output:
```text
Alpha from 7x7: 1.0
Verified confidence: 0.9
```

---

## 5. Automated Test Suite Execution
Execute the entire test suite across all rooms:
```powershell
pytest -v tests/unit/
```
