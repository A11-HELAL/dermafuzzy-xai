# API Contract: `dermafuzzy_core.biomarkers`

**Feature**: [002-opencv-abcd-biomarkers](../spec.md) | **Date**: 2026-11-15

## Module: `packages/dermafuzzy_core/biomarkers.py`

This module provides normalized, explainable ABCD feature extraction routines.

### 1. Function `compute_asymmetry`

```python
def compute_asymmetry(
    mask: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Compute rotation-invariant asymmetry score A in [0.0, 1.0].

    Calculates spatial central moments, rotates mask by principal axis angle -theta,
    centers at ((W-1)/2, (H-1)/2), folds across vertical and horizontal center axes,
    and returns normalized mismatch area ratio.

    Parameters
    ----------
    mask : np.ndarray
        Binary lesion mask in {0, 1}, 2D uint8.
    eps : float, default=1e-6
        Denominator stabilizer.

    Returns
    -------
    float
        Asymmetry score A in [0.0, 1.0]. Ideal circle <= 0.05; L-shape >= 0.35.
    """
```

### 2. Function `compute_border_fuzziness`

```python
def compute_border_fuzziness(
    clean_rgb: np.ndarray,
    mask: np.ndarray,
    boundary_ring: np.ndarray,
    config: DermaConfig | None = None,
) -> tuple[float, float, float]:
    """Compute hybrid border irregularity B_fuzzy, geometric score B_geo, and gradient score B_grad.

    Parameters
    ----------
    clean_rgb : np.ndarray
        Cleaned RGB image, shape (H, W, 3), dtype uint8.
    mask : np.ndarray
        Binary lesion mask in {0, 1}.
    boundary_ring : np.ndarray
        Binary boundary band in {0, 1}.
    config : DermaConfig | None, optional
        Loaded configuration with b_fuzzy_weights.

    Returns
    -------
    tuple[float, float, float]
        (B_fuzzy, B_geo, B_grad), all strictly clipped to [0.0, 1.0].
    """
```

### 3. Function `compute_color_variation`

```python
def compute_color_variation(
    clean_rgb: np.ndarray,
    mask: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Compute multi-channel color variation score C in [0.0, 1.0].

    Evaluates dispersion across CIE-LAB (L, a, b) and HSV (S, V) channels for pixels
    strictly inside the lesion mask.

    Parameters
    ----------
    clean_rgb : np.ndarray
        Cleaned RGB image, shape (H, W, 3), dtype uint8.
    mask : np.ndarray
        Binary lesion mask in {0, 1}.
    eps : float, default=1e-6
        Denominator stabilizer.

    Returns
    -------
    float
        Color variation score C in [0.0, 1.0]. Uniform patch <= 0.03; multi-pigment >= 0.45.
    """
```

### 4. Function `extract_abcd_biomarkers`

```python
def extract_abcd_biomarkers(
    clean_rgb: np.ndarray,
    mask: np.ndarray,
    boundary_ring: np.ndarray,
    config: DermaConfig | None = None,
    p_cnn: float = 0.0,
    heatmap: np.ndarray | None = None,
) -> BiomarkerResult:
    """Orchestrate end-to-end ABCD feature extraction and attention verification.

    Returns
    -------
    BiomarkerResult
        Immutable dataclass holding A, B_fuzzy, C, D_verified, B_geo, B_grad, alpha.
    """
```
