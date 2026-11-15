"""Unit tests for Room 5 4-Panel Clinical XAI Visualizer.

Covers:
- Lazy import of matplotlib (preserves < 50ms package startup, zero global GUI leaks)
- Output array invariants: read-only uint8 array (arr.flags.writeable == False), shape (H, W, 3)
- Non-zero pixel entropy across all 4 panels
- File persistence: .png and .pdf export
- Memory safety: zero figure handle leaks across batch rendering
- Degenerate inputs: all-zero mask, all-zero boundary ring, small and large heatmaps
- Clinical decision banner categories: Low, Medium, High risk styling
- Dominant rule highlighting: R1, R2, R3, R4, R5
- Input validation: invalid data types, non-finite values, and invalid file extensions
"""

import os
import subprocess
import sys
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest

from dermafuzzy_core.biomarkers import BiomarkerResult
from dermafuzzy_core.fis_engine import CLINICAL_RULES, FISPrediction
from dermafuzzy_core.preprocessing import PreprocessingResult
from dermafuzzy_core.visualizer import FourPanelReportConfig, render_four_panel_report


def _make_dummy_visualizer_inputs(
    lesion_empty: bool = False,
    ring_empty: bool = False,
    risk_category: str = "High",
    dominant_rule_id: str = "R4",
):
    """Create realistic synthetic diagnostic artifacts for visualizer testing."""
    h, w = 128, 128
    clean_rgb = np.full((h, w, 3), 180, dtype=np.uint8)

    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - 64) ** 2 + (y - 64) ** 2)

    if lesion_empty:
        lesion_mask = np.zeros((h, w), dtype=np.uint8)
    else:
        lesion_mask = (dist_from_center <= 32).astype(np.uint8) * 255
        clean_rgb[lesion_mask > 0] = [60, 40, 30]

    if ring_empty:
        boundary_ring = np.zeros((h, w), dtype=np.uint8)
    else:
        boundary_ring = ((dist_from_center > 28) & (dist_from_center <= 36)).astype(np.uint8) * 255

    clean_rgb.flags.writeable = False
    lesion_mask.flags.writeable = False
    boundary_ring.flags.writeable = False

    hair_mask = np.zeros((h, w), dtype=np.uint8)
    hair_mask.flags.writeable = False

    prep_result = PreprocessingResult(
        clean_rgb=clean_rgb,
        hair_mask=hair_mask,
        lesion_mask=lesion_mask,
        boundary_ring=boundary_ring,
        area_ratio=0.15 if not lesion_empty else 0.0,
        fallback_used=False,
    )

    biomarker_result = BiomarkerResult(
        A=0.12,
        B_fuzzy=0.18,
        C=0.08,
        D_verified=0.65,
        alpha=0.82,
        B_geo=0.10,
        B_grad=0.15,
    )

    rule_firings = {
        "R1": 0.10,
        "R2": 0.25,
        "R3": 0.15,
        "R4": 0.85,
        "R5": 0.05,
    }

    fis_prediction = FISPrediction(
        risk_score=78.5,
        risk_category=risk_category,
        dominant_rule_id=dominant_rule_id,
        dominant_rule_text=f"{dominant_rule_id}: Alert",
        rule_firings=MappingProxyType(rule_firings),
        memberships=MappingProxyType({}),
    )

    heatmap = np.zeros((16, 16), dtype=np.float32)
    heatmap[4:12, 4:12] = 0.95

    return prep_result, biomarker_result, fis_prediction, heatmap


class TestFourPanelReportConfig:
    """Test FourPanelReportConfig dataclass."""

    def test_default_config(self):
        cfg = FourPanelReportConfig()
        assert cfg.figsize == (16.0, 4.5)
        assert cfg.dpi == 300
        assert cfg.contour_color == (0, 255, 0)
        assert cfg.ring_color == (0, 255, 255)
        assert cfg.heatmap_alpha == 0.35

    @pytest.mark.parametrize(
        ("kwargs", "match_str"),
        [
            ({"dpi": 0}, "dpi"),
            ({"dpi": -10}, "dpi"),
            ({"dpi": True}, "dpi"),
            ({"heatmap_alpha": -0.1}, "heatmap_alpha"),
            ({"heatmap_alpha": 1.5}, "heatmap_alpha"),
            ({"heatmap_alpha": True}, "heatmap_alpha"),
            ({"figsize": (16.0,)}, "figsize"),
            ({"figsize": (-1.0, 4.5)}, "figsize"),
            ({"contour_color": (0, 255)}, "contour_color"),
            ({"contour_color": (0, 255, 300)}, "contour_color"),
            ({"ring_color": (-5, 255, 0)}, "ring_color"),
        ],
    )
    def test_config_validation(self, kwargs: dict[str, Any], match_str: str):
        with pytest.raises(ValueError, match=match_str):
            FourPanelReportConfig(**kwargs)


class TestRenderFourPanelReport:
    """Test render_four_panel_report rendering, file output, and invariants."""

    def test_render_output_array_invariants(self):
        prep, bio, fis, heatmap = _make_dummy_visualizer_inputs()
        arr = render_four_panel_report(prep, bio, fis, heatmap, dpi=100)

        assert isinstance(arr, np.ndarray)
        assert arr.ndim == 3
        assert arr.shape[2] == 3
        assert arr.dtype == np.uint8
        assert not arr.flags.writeable
        assert np.std(arr) > 10.0

    def test_save_png_and_pdf(self, tmp_path: Path):
        prep, bio, fis, heatmap = _make_dummy_visualizer_inputs()

        png_path = tmp_path / "report.png"
        pdf_path = tmp_path / "report.pdf"

        arr_png = render_four_panel_report(prep, bio, fis, heatmap, output_path=png_path, dpi=100)
        assert png_path.exists()
        assert png_path.stat().st_size > 1000
        assert isinstance(arr_png, np.ndarray)

        arr_pdf = render_four_panel_report(prep, bio, fis, heatmap, output_path=pdf_path, dpi=100)
        assert pdf_path.exists()
        assert pdf_path.stat().st_size > 1000
        assert isinstance(arr_pdf, np.ndarray)

    def test_zero_figure_handle_leaks(self):
        import matplotlib.pyplot as plt

        prep, bio, fis, heatmap = _make_dummy_visualizer_inputs()

        initial_figns = len(plt.get_fignums())
        for _ in range(5):
            _ = render_four_panel_report(prep, bio, fis, heatmap, dpi=80)
            assert len(plt.get_fignums()) == initial_figns

    def test_render_degenerate_blank_mask_and_ring(self):
        """Verify report renders successfully when lesion mask or boundary ring is completely empty."""
        prep, bio, fis, heatmap = _make_dummy_visualizer_inputs(lesion_empty=True, ring_empty=True)
        arr = render_four_panel_report(prep, bio, fis, heatmap, dpi=80)
        assert isinstance(arr, np.ndarray)
        assert arr.shape[2] == 3
        assert not arr.flags.writeable

    @pytest.mark.parametrize("category", ["Low", "Medium", "High"])
    def test_render_decision_banner_categories(self, category: str):
        """Verify all three clinical risk categories render without error."""
        prep, bio, fis, heatmap = _make_dummy_visualizer_inputs(risk_category=category)
        arr = render_four_panel_report(prep, bio, fis, heatmap, dpi=80)
        assert arr.shape[2] == 3

    @pytest.mark.parametrize("dominant_rule", ["R1", "R2", "R3", "R4", "R5"])
    def test_render_dominant_rule_highlighting(self, dominant_rule: str):
        """Verify each clinical rule dominant selection renders without error."""
        prep, bio, fis, heatmap = _make_dummy_visualizer_inputs(dominant_rule_id=dominant_rule)
        arr = render_four_panel_report(prep, bio, fis, heatmap, dpi=80)
        assert arr.shape[2] == 3

    def test_render_heatmap_resizing_resolutions(self):
        """Verify low-resolution (7x7) and high-resolution (256x256) heatmaps are scaled correctly."""
        prep, bio, fis, _ = _make_dummy_visualizer_inputs()

        # 7x7 small heatmap
        small_hm = np.ones((7, 7), dtype=np.float32)
        arr_small = render_four_panel_report(prep, bio, fis, small_hm, dpi=80)
        assert arr_small.shape[2] == 3

        # 256x256 large heatmap
        large_hm = np.ones((256, 256), dtype=np.float32)
        arr_large = render_four_panel_report(prep, bio, fis, large_hm, dpi=80)
        assert arr_large.shape[2] == 3

    def test_input_validations(self):
        prep, bio, fis, heatmap = _make_dummy_visualizer_inputs()

        # Reject non-PreprocessingResult
        with pytest.raises(ValueError, match="prep_result"):
            render_four_panel_report("bad_prep", bio, fis, heatmap, dpi=80)  # type: ignore

        # Reject non-BiomarkerResult
        with pytest.raises(ValueError, match="biomarker_result"):
            render_four_panel_report(prep, "bad_bio", fis, heatmap, dpi=80)  # type: ignore

        # Reject non-FISPrediction
        with pytest.raises(ValueError, match="fis_prediction"):
            render_four_panel_report(prep, bio, "bad_fis", heatmap, dpi=80)  # type: ignore

        # Reject 1D or 3D heatmap
        with pytest.raises(ValueError, match="2D"):
            render_four_panel_report(prep, bio, fis, np.zeros((10,)), dpi=80)

        with pytest.raises(ValueError, match="2D"):
            render_four_panel_report(prep, bio, fis, np.zeros((10, 10, 3)), dpi=80)

        # Reject non-array heatmap
        with pytest.raises(ValueError, match="2D numpy array"):
            render_four_panel_report(prep, bio, fis, "not_an_array", dpi=80)  # type: ignore

        # Reject empty heatmap
        with pytest.raises(ValueError, match="cannot be empty"):
            render_four_panel_report(prep, bio, fis, np.zeros((0, 0)), dpi=80)

        # Reject NaN / Inf in heatmap
        bad_heat = heatmap.copy()
        bad_heat[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN or Inf"):
            render_four_panel_report(prep, bio, fis, bad_heat, dpi=80)

        # Reject invalid file extension
        with pytest.raises(ValueError, match="output_path must end with .png or .pdf"):
            render_four_panel_report(prep, bio, fis, heatmap, output_path="report.txt", dpi=80)

        # Reject invalid DPI
        with pytest.raises(ValueError, match="dpi"):
            render_four_panel_report(prep, bio, fis, heatmap, dpi=-10)

        with pytest.raises(ValueError, match="dpi"):
            render_four_panel_report(prep, bio, fis, heatmap, dpi=True)  # type: ignore

    def test_fast_import_and_no_toplevel_matplotlib(self):
        """Verify importing dermafuzzy_core does not eagerly import matplotlib."""
        code = """
import sys
import dermafuzzy_core
assert 'matplotlib' not in sys.modules, 'matplotlib should not be loaded at module top-level'
"""
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert res.returncode == 0, res.stderr
