# Contract: Visualizer Engine API

**Feature**: [004-ablation-visualizer-colab-ci](../spec.md)  
**Module**: `packages/dermafuzzy_core/visualizer.py`  
**Date**: 2026-11-15  

---

## 1. Exported Symbols

The module `packages/dermafuzzy_core/visualizer.py` exports:
- `FourPanelReportConfig`
- `render_four_panel_report`

---

## 2. Function Specifications

### 2.1 `render_four_panel_report`
```python
def render_four_panel_report(
    prep_result: PreprocessingResult,
    biomarker_result: BiomarkerResult,
    fis_prediction: FISPrediction,
    heatmap: np.ndarray,
    output_path: str | Path | None = None,
    dpi: int = 300,
    config: FourPanelReportConfig | None = None,
) -> np.ndarray:
    """Render standardized 4-panel clinical XAI audit figure.
    
    Panels:
    1. Primary Lesion Segmentation: Preprocessed RGB with emerald green contour overlay.
    2. Boundary Fuzziness: Cyan/gold perilesional ring overlay with B_geo, B_grad, B_fuzzy.
    3. Attention Verification: Jet Grad-CAM attention heatmap overlay with alpha and D_verified.
    4. Clinical Rule Firings: Horizontal bar chart of R1-R5, dominant rule, and decision banner.
    
    Args:
        prep_result: PreprocessingResult containing clean_rgb, lesion_mask, boundary_ring.
        biomarker_result: BiomarkerResult containing A, B_fuzzy, C, D_verified, alpha.
        fis_prediction: FISPrediction containing risk_score, risk_category, rule_firings.
        heatmap: 2D Grad-CAM attention heatmap array.
        output_path: Optional filesystem path (.png or .pdf) to persist figure.
        dpi: Resolution dots-per-inch (default 300).
        
    Returns:
        Rendered RGB image as a read-only 3-channel uint8 NumPy array (arr.flags.writeable = False).
        
    Raises:
        ValueError: If input results or heatmap dimensions are invalid.
    """
```
