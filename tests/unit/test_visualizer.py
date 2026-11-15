"""Unit tests for Room 5 4-Panel Clinical XAI Visualizer.

Covers:
- Lazy import of matplotlib (preserves < 50ms package startup, zero global GUI leaks)
- Output array invariants: read-only uint8 array (arr.flags.writeable == False), shape (H, W, 3)
- Non-zero pixel entropy across all 4 panels
- File persistence: .png and .pdf export
- Memory safety: zero figure handle leaks across batch rendering
- Input validation: invalid heatmaps, non-finite values, and invalid file extensions
"""

import os
import subprocess
import sys
import numpy as np
import pytest
from types import MappingProxyType

from dermafuzzy_core.preprocessing import PreprocessingResult
from dermafuzzy_core.biomarkers import BiomarkerResult
from dermafuzzy_core.fis_engine import FISPrediction, CLINICAL_RULES
from dermafuzzy_core.visualizer import FourPanelReportConfig, render_four_panel_report


def _make_dummy_visualizer_inputs():
    """Create realistic synthetic diagnostic artifacts for visualizer testing."""
    h, w = 128, 128
    clean_rgb = np.full((h, w, 3), 180, dtype=np.uint8)
    # Circle in center
    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - 64) ** 2 + (y - 64) ** 2)
    lesion_mask = (dist_from_center <= 32).astype(np.uint8) * 255
    boundary_ring = ((dist_from_center > 28) & (dist_from_center <= 36)).astype(np.uint8) * 255

    clean_rgb[lesion_mask > 0] = [60, 40, 30]

    # PreprocessingResult requires flags.writeable = False
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
        area_ratio=0.15,
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

    rule = CLINICAL_RULES[3]  # R4

    rule_firings = {
        "R1": 0.10,
        "R2": 0.25,
        "R3": 0.15,
        "R4": 0.85,
        "R5": 0.05,
    }

    fis_prediction = FISPrediction(
        risk_score=78.5,
        risk_category="High",
        dominant_rule_id="R4",
        dominant_rule_text="R4: Malignant Alert",
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

    def test_config_validation(self):
        with pytest.raises(ValueError, match="dpi"):
            FourPanelReportConfig(dpi=0)

        with pytest.raises(ValueError, match="heatmap_alpha"):
            FourPanelReportConfig(heatmap_alpha=1.5)


class TestRenderFourPanelReport:
    """Test render_four_panel_report rendering, file output, and invariants."""

    def test_render_output_array_invariants(self):
        prep, bio, fis, heatmap = _make_dummy_visualizer_inputs()
        # Use lower dpi for fast test execution
        arr = render_four_panel_report(prep, bio, fis, heatmap, dpi=100)

        assert isinstance(arr, np.ndarray)
        assert arr.ndim == 3
        assert arr.shape[2] == 3
        assert arr.dtype == np.uint8
        # Enforce read-only array
        assert not arr.flags.writeable

        # Non-zero pixel entropy
        assert np.std(arr) > 10.0

    def test_save_png_and_pdf(self, tmp_path):
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

    def test_input_validations(self):
        prep, bio, fis, heatmap = _make_dummy_visualizer_inputs()

        # Reject 1D or 3D heatmap
        with pytest.raises(ValueError, match="2D"):
            render_four_panel_report(prep, bio, fis, np.zeros((10,)), dpi=80)

        with pytest.raises(ValueError, match="2D"):
            render_four_panel_report(prep, bio, fis, np.zeros((10, 10, 3)), dpi=80)

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

    def test_fast_import_and_no_toplevel_matplotlib(self):
        """Verify importing dermafuzzy_core does not eagerly import matplotlib."""
        code = """
import sys
import dermafuzzy_core
# Verify matplotlib was not loaded on initial package import
assert 'matplotlib' not in sys.modules, 'matplotlib should not be loaded at module top-level'
"""
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert res.returncode == 0, res.stderr
