# API Contract: `dermafuzzy_core.gradcam`

**Feature**: [002-opencv-abcd-biomarkers](../spec.md) | **Date**: 2026-11-15

## Module: `packages/dermafuzzy_core/gradcam.py`

This module provides spatial attention verification routines to penalize ungrounded deep learning predictions.

### 1. Function `compute_attention_overlap`

```python
def compute_attention_overlap(
    heatmap: np.ndarray | None,
    mask: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Calculate spatial attention overlap alpha in [0.0, 1.0].

    Validates heatmap array, applies ReLU, upsamples via bilinear interpolation
    if spatial dimensions differ from mask, and computes on-lesion activation proportion.

    Parameters
    ----------
    heatmap : np.ndarray | None
        2D floating-point Grad-CAM activation map (e.g. 7x7 or 224x224). If None, returns 0.0.
    mask : np.ndarray
        Binary lesion mask in {0, 1}, 2D uint8.
    eps : float, default=1e-6
        Denominator stabilizer.

    Returns
    -------
    float
        Attention overlap ratio alpha in [0.0, 1.0]. On-lesion >= 0.85; off-lesion <= 0.15.

    Raises
    ------
    ValueError
        If heatmap is not a 2D array or contains non-finite values.
    """
```

### 2. Function `compute_verified_confidence`

```python
def compute_verified_confidence(
    p_cnn: float,
    alpha: float,
) -> float:
    """Compute attention-verified deep confidence D_verified = np.clip(p_cnn * alpha, 0.0, 1.0).

    Parameters
    ----------
    p_cnn : float
        Raw deep learning classifier prediction in [0.0, 1.0].
    alpha : float
        Spatial attention overlap ratio in [0.0, 1.0].

    Returns
    -------
    float
        Verified confidence score D_verified in [0.0, 1.0].

    Raises
    ------
    ValueError
        If p_cnn or alpha are boolean, non-finite, or outside [0.0, 1.0].
    """
```
