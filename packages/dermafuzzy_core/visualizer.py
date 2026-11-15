"""Room 5: Publication-Grade 4-Panel Clinical XAI Visualizer for DermaFuzzy-XAI.

Provides:
- FourPanelReportConfig: Configuration dataclass for figure geometry, DPI, and colors.
- render_four_panel_report: Publication-grade (300 DPI) 4-panel visual explanation report
  rendered via headless Matplotlib (Figure + FigureCanvasAgg) with lazy loading to ensure
  fast (< 50ms) package imports and zero GUI event-loop leaks.

Strictly adheres to:
- Constitution Principle I: Zero tensorflow/torch/keras imports.
- Constitution Principle II: Strict mathematical bounds and locked array outputs (arr.flags.writeable = False).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import cv2
import numpy as np

from dermafuzzy_core.biomarkers import BiomarkerResult
from dermafuzzy_core.fis_engine import FISPrediction
from dermafuzzy_core.preprocessing import PreprocessingResult

# Visualizer named constants
DEFAULT_FIGSIZE: Final[tuple[float, float]] = (16.0, 4.5)
DEFAULT_DPI: Final[int] = 300
DEFAULT_CONTOUR_COLOR: Final[tuple[int, int, int]] = (0, 255, 0)      # Emerald green (RGB)
DEFAULT_RING_COLOR: Final[tuple[int, int, int]] = (0, 255, 255)       # Cyan (RGB)
DEFAULT_HEATMAP_ALPHA: Final[float] = 0.35

BANNER_COLOR_LOW: Final[str] = "#28a745"
BANNER_COLOR_MEDIUM: Final[str] = "#ffc107"
BANNER_COLOR_HIGH: Final[str] = "#dc3545"
BAR_COLOR_DOMINANT: Final[str] = "#dc3545"
BAR_COLOR_DEFAULT: Final[str] = "#4a90e2"
CENTROID_LINE_COLOR: Final[str] = "#212529"


@dataclass(frozen=True, slots=True)
class FourPanelReportConfig:
    """Configuration options for the 4-panel clinical XAI audit report."""

    figsize: tuple[float, float] = DEFAULT_FIGSIZE
    dpi: int = DEFAULT_DPI
    contour_color: tuple[int, int, int] = DEFAULT_CONTOUR_COLOR
    ring_color: tuple[int, int, int] = DEFAULT_RING_COLOR
    heatmap_alpha: float = DEFAULT_HEATMAP_ALPHA

    def __post_init__(self) -> None:
        """Validate config parameters."""
        if type(self.dpi) is bool or not isinstance(self.dpi, (int, np.integer)) or self.dpi <= 0:
            raise ValueError(f"dpi must be a positive integer, got {self.dpi}")

        if type(self.heatmap_alpha) is bool or not (0.0 <= float(self.heatmap_alpha) <= 1.0):
            raise ValueError(f"heatmap_alpha must be in [0.0, 1.0], got {self.heatmap_alpha}")

        if len(self.figsize) != 2 or self.figsize[0] <= 0 or self.figsize[1] <= 0:
            raise ValueError(f"figsize must be a tuple of 2 positive numbers, got {self.figsize}")

        for name, col in (("contour_color", self.contour_color), ("ring_color", self.ring_color)):
            if len(col) != 3 or not all(0 <= c <= 255 for c in col):
                raise ValueError(f"{name} must be a 3-element RGB tuple with values in [0, 255], got {col}")


def _overlay_contour(rgb: np.ndarray, mask: np.ndarray, color_rgb: tuple[int, int, int]) -> np.ndarray:
    """Overlay binary contour on RGB image in specified RGB color."""
    out = rgb.copy()
    if np.any(mask > 0):
        # findContours expects single-channel uint8
        bin_mask = (mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # OpenCV drawContours uses color in native format matching image (RGB here)
        cv2.drawContours(out, contours, -1, color_rgb, 2)
    return out


def _overlay_ring(rgb: np.ndarray, ring_mask: np.ndarray, color_rgb: tuple[int, int, int]) -> np.ndarray:
    """Blend boundary ring in specified RGB color over RGB image."""
    out = rgb.copy().astype(np.float32)
    ring_idx = ring_mask > 0
    if np.any(ring_idx):
        col_arr = np.array(color_rgb, dtype=np.float32)
        out[ring_idx] = 0.45 * out[ring_idx] + 0.55 * col_arr
    return np.clip(out, 0, 255).astype(np.uint8)


def _overlay_heatmap(rgb: np.ndarray, heatmap: np.ndarray, alpha: float) -> np.ndarray:
    """Blend JET-colormapped attention heatmap over RGB image."""
    h, w = rgb.shape[:2]
    # Resize heatmap to match image dimensions
    if heatmap.shape[:2] != (h, w):
        heat_resized = cv2.resize(heatmap.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        heat_resized = heatmap.astype(np.float32)

    heat_norm = np.clip(heat_resized, 0.0, 1.0)
    heat_u8 = (heat_norm * 255.0).astype(np.uint8)
    jet_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    jet_rgb = cv2.cvtColor(jet_bgr, cv2.COLOR_BGR2RGB)

    blended = (1.0 - alpha) * rgb.astype(np.float32) + alpha * jet_rgb.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


def _render_panel_1_lesion(
    ax: Any,
    clean_rgb: np.ndarray,
    lesion_mask: np.ndarray,
    biomarker_result: BiomarkerResult,
    config: FourPanelReportConfig,
) -> None:
    """Render Panel 1: Lesion contour overlay and A / C biomarker readouts."""
    panel_img = _overlay_contour(clean_rgb, lesion_mask, config.contour_color)
    ax.imshow(panel_img)
    ax.set_title(
        f"1. Lesion Contour\nAsym A={biomarker_result.A:.3f} | Color C={biomarker_result.C:.3f}",
        fontsize=10,
        fontweight="bold",
    )
    ax.axis("off")


def _render_panel_2_boundary(
    ax: Any,
    clean_rgb: np.ndarray,
    boundary_ring: np.ndarray,
    biomarker_result: BiomarkerResult,
    config: FourPanelReportConfig,
) -> None:
    """Render Panel 2: Cyan perilesional boundary ring overlay and B_geo / B_grad / B_fuzzy readouts."""
    panel_img = _overlay_ring(clean_rgb, boundary_ring, config.ring_color)
    ax.imshow(panel_img)
    ax.set_title(
        f"2. Boundary Ring (Cyan)\nB_geo={biomarker_result.B_geo:.3f} | B_grad={biomarker_result.B_grad:.3f}\nTotal B_fuzzy={biomarker_result.B_fuzzy:.3f}",
        fontsize=10,
        fontweight="bold",
    )
    ax.axis("off")


def _render_panel_3_attention(
    ax: Any,
    clean_rgb: np.ndarray,
    heatmap: np.ndarray,
    biomarker_result: BiomarkerResult,
    config: FourPanelReportConfig,
) -> None:
    """Render Panel 3: Attention heatmap overlay with overlap alpha and D_verified readouts."""
    panel_img = _overlay_heatmap(clean_rgb, heatmap, config.heatmap_alpha)
    ax.imshow(panel_img)
    ax.set_title(
        f"3. Attention Verification\nOverlap α={biomarker_result.alpha:.3f}\nD_verified={biomarker_result.D_verified:.3f}",
        fontsize=10,
        fontweight="bold",
    )
    ax.axis("off")


def _render_panel_4_rules(ax: Any, fis_prediction: FISPrediction) -> None:
    """Render Panel 4: Horizontal Mamdani rule firing strengths, centroid line, and decision banner."""
    rule_labels = ["R1 (Low)", "R2 (Early)", "R3 (Border)", "R4 (Invasive)", "R5 (Veto)"]
    rule_keys = ["R1", "R2", "R3", "R4", "R5"]
    firings = [float(fis_prediction.rule_firings.get(k, 0.0)) for k in rule_keys]

    dominant_key = (
        getattr(fis_prediction, "dominant_rule_id", None)
        or (getattr(fis_prediction.dominant_rule, "rule_id", None) if hasattr(fis_prediction, "dominant_rule") else None)
        or "R1"
    )
    bar_colors = [BAR_COLOR_DOMINANT if k == dominant_key else BAR_COLOR_DEFAULT for k in rule_keys]

    y_pos = np.arange(len(rule_labels))
    ax.barh(y_pos, firings, color=bar_colors, edgecolor="black", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(rule_labels, fontsize=9)
    ax.set_xlim(0.0, 1.05)
    ax.set_xlabel("Firing Strength", fontsize=9)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    risk_pct = float(fis_prediction.risk_score)
    risk_ratio = risk_pct / 100.0
    ax.axvline(risk_ratio, color=CENTROID_LINE_COLOR, linestyle=":", linewidth=1.8, label=f"Risk: {risk_pct:.1f}%")
    ax.legend(loc="lower right", fontsize=8)

    cat = fis_prediction.risk_category
    if "Low" in cat:
        banner_bg = BANNER_COLOR_LOW
        banner_fg = "white"
    elif "Medium" in cat:
        banner_bg = BANNER_COLOR_MEDIUM
        banner_fg = "black"
    else:
        banner_bg = BANNER_COLOR_HIGH
        banner_fg = "white"

    banner_text = f"DECISION: {cat.upper()} ({risk_pct:.1f}%)"
    ax.set_title(
        banner_text,
        fontsize=10,
        fontweight="bold",
        color=banner_fg,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=banner_bg, edgecolor="none"),
    )


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
    2. Boundary Fuzziness: Cyan perilesional ring overlay with B_geo, B_grad, B_fuzzy.
    3. Attention Verification: Jet Grad-CAM attention heatmap overlay with alpha and D_verified.
    4. Clinical Rule Firings: Horizontal bar chart of R1-R5, dominant rule, and decision banner.

    Args:
        prep_result: PreprocessingResult containing clean_rgb, lesion_mask, boundary_ring.
        biomarker_result: BiomarkerResult containing A, B_fuzzy, C, D_verified, alpha.
        fis_prediction: FISPrediction containing risk_score, risk_category, rule_firings.
        heatmap: 2D Grad-CAM attention heatmap array.
        output_path: Optional filesystem path (.png or .pdf) to persist figure.
        dpi: Resolution dots-per-inch (default 300).
        config: Optional FourPanelReportConfig instance.

    Returns:
        Rendered RGB image as a read-only 3-channel uint8 NumPy array (arr.flags.writeable = False).

    Raises:
        ValueError: If input results, heatmap dimensions, or file extensions are invalid.
    """
    if config is None:
        config = FourPanelReportConfig(dpi=dpi)
    elif dpi != 300 and config.dpi == 300:
        # Override dpi if passed explicitly
        config = FourPanelReportConfig(
            figsize=config.figsize,
            dpi=dpi,
            contour_color=config.contour_color,
            ring_color=config.ring_color,
            heatmap_alpha=config.heatmap_alpha,
        )

    if type(dpi) is bool or not isinstance(dpi, (int, np.integer)) or dpi <= 0:
        raise ValueError(f"dpi must be a positive integer, got {dpi}")

    if not isinstance(prep_result, PreprocessingResult):
        raise ValueError(f"prep_result must be PreprocessingResult, got {type(prep_result).__name__}")
    if not isinstance(biomarker_result, BiomarkerResult):
        raise ValueError(f"biomarker_result must be BiomarkerResult, got {type(biomarker_result).__name__}")
    if not isinstance(fis_prediction, FISPrediction):
        raise ValueError(f"fis_prediction must be FISPrediction, got {type(fis_prediction).__name__}")

    # Validate heatmap
    if type(heatmap) is bool or not isinstance(heatmap, np.ndarray):
        raise ValueError(f"heatmap must be a 2D numpy array, got {type(heatmap).__name__}")
    if heatmap.ndim != 2:
        raise ValueError(f"heatmap must be 2D array, got ndim={heatmap.ndim} shape={heatmap.shape}")
    if not np.all(np.isfinite(heatmap)):
        raise ValueError("heatmap contains NaN or Inf values")
    if heatmap.size == 0:
        raise ValueError("heatmap cannot be empty")

    # Validate output path
    if output_path is not None:
        path_obj = Path(output_path)
        ext = path_obj.suffix.lower()
        if ext not in (".png", ".pdf"):
            raise ValueError(f"output_path must end with .png or .pdf, got {ext}")

    # Lazy import of matplotlib to preserve fast sub-second startup
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    # Setup headless figure and canvas
    fig = Figure(figsize=config.figsize, dpi=config.dpi)
    canvas = FigureCanvasAgg(fig)
    axes = fig.subplots(1, 4)

    clean_rgb = prep_result.clean_rgb
    _render_panel_1_lesion(axes[0], clean_rgb, prep_result.lesion_mask, biomarker_result, config)
    _render_panel_2_boundary(axes[1], clean_rgb, prep_result.boundary_ring, biomarker_result, config)
    _render_panel_3_attention(axes[2], clean_rgb, heatmap, biomarker_result, config)
    _render_panel_4_rules(axes[3], fis_prediction)

    fig.tight_layout()

    # Save to file if path requested
    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(p), dpi=config.dpi, bbox_inches="tight")

    # Render buffer to NumPy RGB array
    canvas.draw()
    rgba_buffer = np.asarray(canvas.buffer_rgba())
    rgb_arr = rgba_buffer[:, :, :3].copy()

    # Clean up figure resources immediately to avoid leaks
    fig.clf()
    del fig
    del canvas

    # Enforce read-only array invariant
    rgb_arr.flags.writeable = False
    return rgb_arr
