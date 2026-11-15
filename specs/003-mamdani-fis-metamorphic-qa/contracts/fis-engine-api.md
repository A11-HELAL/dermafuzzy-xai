# Contract: Room 4 Mamdani FIS Engine API

**Feature**: [003-mamdani-fis-metamorphic-qa](../spec.md)  
**Module**: `packages/dermafuzzy_core/fis_engine.py`  
**Date**: 2026-11-15  

---

## 1. Exported Symbols

The module `packages/dermafuzzy_core/fis_engine.py` exports:
- `FISCalibration`
- `FuzzyMembershipSet`
- `ClinicalRule`
- `FISPrediction`
- `CLINICAL_RULES`
- `calibrate_fis_percentiles`
- `save_calibration_json`
- `load_calibration_json`
- `compute_memberships`
- `evaluate_mamdani_fis`
- `diagnose_image`

---

## 2. Function Specifications

### 2.1 `calibrate_fis_percentiles`
```python
def calibrate_fis_percentiles(
    train_matrix: np.ndarray | Sequence[Sequence[float]],
    config: DermaConfig | None = None,
) -> FISCalibration:
    """Calibrate empirical biomarker percentiles from a training dataset split.
    
    Args:
        train_matrix: 2D array-like of shape (N, 4) with N >= 4.
                      Columns correspond to [A, B_fuzzy, C, D_verified].
                      All elements must be finite floats in [0.0, 1.0].
        config: Optional DermaConfig instance. Defaults to load_config().
        
    Returns:
        FISCalibration instance with monotonic, non-degenerate cutoffs.
        
    Raises:
        ValueError: If train_matrix is not 2D, N < 4, contains bool/NaN/Inf,
                    or values fall outside [0.0, 1.0].
    """
```

### 2.2 `save_calibration_json` & `load_calibration_json`
```python
def save_calibration_json(
    calibration: FISCalibration,
    filepath: str | Path = "artifacts/fis_calibration.json",
) -> Path:
    """Serialize calibration parameters to human-readable JSON.
    
    Args:
        calibration: FISCalibration instance to persist.
        filepath: Target filesystem path. Parent directories are created automatically.
        
    Returns:
        Path of written JSON artifact.
        
    Raises:
        ValueError: If calibration data is invalid.
    """

def load_calibration_json(
    filepath: str | Path = "artifacts/fis_calibration.json",
) -> FISCalibration:
    """Load calibration parameters from human-readable JSON with schema validation.
    
    Args:
        filepath: Filesystem path to read from.
        
    Returns:
        Immutable FISCalibration instance with MappingProxyType cutoffs.
        
    Raises:
        FileNotFoundError: If filepath does not exist.
        ValueError: If JSON content fails schema validation or cutoffs are degenerate.
    """
```

### 2.3 `compute_memberships`
```python
def compute_memberships(
    score: float,
    cutoffs: tuple[float, float, float],
) -> FuzzyMembershipSet:
    """Compute linguistic memberships satisfying Partition of Unity.
    
    Args:
        score: Single biomarker score in [0.0, 1.0].
        cutoffs: Calibrated (P25, P50, P75) monotonic tuple.
        
    Returns:
        FuzzyMembershipSet with low, medium, high satisfying sum == 1.0.
        
    Raises:
        ValueError: If score is bool, non-finite, or outside [0.0, 1.0].
    """
```

### 2.4 `evaluate_mamdani_fis`
```python
def evaluate_mamdani_fis(
    biomarkers: BiomarkerResult | tuple[float, float, float, float],
    calibration: FISCalibration,
    config: DermaConfig | None = None,
) -> FISPrediction:
    """Evaluate Mamdani fuzzy inference system with centroid defuzzification.
    
    Args:
        biomarkers: Either BiomarkerResult or (A, B_fuzzy, C, D_verified) tuple in [0.0, 1.0]^4.
        calibration: Calibrated percentiles container.
        config: Optional DermaConfig instance. Defaults to load_config().
        
    Returns:
        Immutable FISPrediction with risk_score, risk_category, dominant_rule, and firings.
        
    Raises:
        ValueError: If any input biomarker is bool, non-finite, or out of [0.0, 1.0].
    """
```

### 2.5 `diagnose_image`
```python
def diagnose_image(
    img_rgb: np.ndarray,
    p_cnn: float,
    heatmap: np.ndarray,
    calibration: FISCalibration,
    config: DermaConfig | None = None,
) -> tuple[PreprocessingResult, BiomarkerResult, FISPrediction]:
    """Execute end-to-end diagnostic pipeline: Room 2 -> Room 3 -> Room 4.
    
    Args:
        img_rgb: Raw dermoscopic image (uint8, 3 channels).
        p_cnn: Raw deep CNN probability in [0.0, 1.0].
        heatmap: 2D Grad-CAM attention heatmap (2D array).
        calibration: Calibrated FIS percentiles.
        config: Optional DermaConfig instance. Defaults to load_config().
        
    Returns:
        Tuple of (PreprocessingResult, BiomarkerResult, FISPrediction).
    """
```
