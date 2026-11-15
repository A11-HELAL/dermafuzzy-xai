# Contract: Evaluation & Ablation Engine API

**Feature**: [004-ablation-visualizer-colab-ci](../spec.md)  
**Module**: `packages/dermafuzzy_core/evaluation.py`  
**Date**: 2026-11-15  

---

## 1. Exported Symbols

The module `packages/dermafuzzy_core/evaluation.py` exports:
- `ClinicalMetrics`
- `AblationReport`
- `compute_clinical_metrics`
- `evaluate_crisp_abcd_baseline`
- `run_ablation_study`

---

## 2. Function Specifications

### 2.1 `compute_clinical_metrics`
```python
def compute_clinical_metrics(
    y_true: np.ndarray | Sequence[int],
    y_prob: np.ndarray | Sequence[float],
    threshold: float = 0.50,
) -> ClinicalMetrics:
    """Compute biostatistical clinical classification metrics and pure NumPy AUC-ROC.
    
    Args:
        y_true: 1D array-like of binary ground truth labels in {0, 1}.
        y_prob: 1D array-like of continuous prediction probabilities in [0.0, 1.0]
                (automatically normalized if provided in [0.0, 100.0]).
        threshold: Decision cutoff for binary classification (default 0.50).
        
    Returns:
        Immutable ClinicalMetrics dataclass with TP, TN, FP, FN, Sensitivity,
        Specificity, Precision, Accuracy, F1-Score, and AUC-ROC.
        
    Raises:
        ValueError: If inputs have mismatched lengths, invalid shapes, non-finite values,
                    or invalid labels outside {0, 1}.
    """
```

### 2.2 `evaluate_crisp_abcd_baseline`
```python
def evaluate_crisp_abcd_baseline(
    biomarkers_batch: np.ndarray | Sequence[Sequence[float]] | Sequence[BiomarkerResult],
    calibration: FISCalibration,
) -> np.ndarray:
    """Evaluate Baseline 2 (Crisp ABCD + Fixed Thresholds) using step indicators at P50.
    
    Args:
        biomarkers_batch: Batch of ABCD biomarker vectors (N, 4).
        calibration: Calibrated percentiles container supplying P50 cutoffs.
        
    Returns:
        1D float NumPy array of shape (N,) containing fraction of triggered crisp indicators.
        
    Raises:
        ValueError: If input dimensions or biomarker scores are invalid.
    """
```

### 2.3 `run_ablation_study`
```python
def run_ablation_study(
    y_true: np.ndarray | Sequence[int],
    p_cnn_raw: np.ndarray | Sequence[float],
    biomarkers_batch: np.ndarray | Sequence[Sequence[float]] | Sequence[BiomarkerResult],
    calibration: FISCalibration,
    config: DermaConfig | None = None,
) -> AblationReport:
    """Execute 3-tier comparative ablation study.
    
    Tiers evaluated:
    1. Pure MobileNetV2 Black-Box (raw p_cnn_raw).
    2. Crisp ABCD + Fixed Thresholds (step indicators at P50).
    3. Proposed DermaFuzzy-XAI (Mamdani FIS centroid risk).
    
    Args:
        y_true: Ground truth binary labels in {0, 1}.
        p_cnn_raw: Unverified deep CNN confidence probabilities in [0.0, 1.0].
        biomarkers_batch: Batch of ABCD biomarker vectors (N, 4).
        calibration: Calibrated FIS percentiles container.
        config: Optional DermaConfig instance. Defaults to load_config().
        
    Returns:
        Immutable AblationReport dataclass with comparative metrics and export formatters.
    """
```
