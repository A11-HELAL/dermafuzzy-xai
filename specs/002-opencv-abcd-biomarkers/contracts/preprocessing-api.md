# API Contract: `dermafuzzy_core.preprocessing`

**Feature**: [002-opencv-abcd-biomarkers](../spec.md) | **Date**: 2026-11-15

## Module: `packages/dermafuzzy_core/preprocessing.py`

This module provides 100% pure, deterministic, write-protected OpenCV image preprocessing and lesion segmentation.

### 1. Function `preprocess_image`

```python
def preprocess_image(
    img_rgb: np.ndarray,
    config: DermaConfig | None = None,
) -> PreprocessingResult:
    """Preprocess a raw dermoscopic image end-to-end.

    Executes:
    1. Input validation (3D uint8 RGB) and deterministic resize to config dimensions.
    2. Morphological Black-Hat and Telea inpainting for hair artifact removal.
    3. Illumination equalization via CIE-LAB CLAHE on the L channel.
    4. Gaussian smoothing, Otsu thresholding, border vignetting suppression, and LCC segmentation.
    5. Fallback check for degenerate empty/saturated frames.
    6. Perilesional boundary ring derivation.
    7. Read-only array protection on all output buffers.

    Parameters
    ----------
    img_rgb : np.ndarray
        Input RGB image, shape (H, W, 3) and dtype uint8.
    config : DermaConfig | None, optional
        Loaded configuration. If None, default configuration is loaded.

    Returns
    -------
    PreprocessingResult
        Immutable dataclass holding clean_rgb, hair_mask, lesion_mask, boundary_ring,
        area_ratio, and fallback_used.

    Raises
    ------
    ValueError
        If img_rgb is not a 3D uint8 array with 3 channels, is empty, or contains non-finite values.
    """
```

### 2. Function `remove_hair`

```python
def remove_hair(
    img_rgb: np.ndarray,
    kernel_size: int = 17,
    inpaint_radius: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Isolate dark hair occlusions via Black-Hat morphology and restore texture via Telea inpainting.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (inpainted_rgb, hair_mask) where hair_mask is uint8 in {0, 255}.
    """
```

### 3. Function `apply_lab_clahe`

```python
def apply_lab_clahe(
    img_rgb: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply CLAHE strictly to the L channel in CIE-LAB space, keeping a and b bit-identical.

    Returns
    -------
    np.ndarray
        Clean RGB image with normalized illumination, shape (H, W, 3), dtype uint8.
    """
```

### 4. Function `extract_lesion_mask_and_ring`

```python
def extract_lesion_mask_and_ring(
    clean_rgb: np.ndarray,
    config: DermaConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """Segment primary lesion mask M and compute perilesional boundary ring R_edge.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, float, bool]
        (lesion_mask, boundary_ring, area_ratio, fallback_used).
        Both masks are uint8 in {0, 1}.
    """
```
