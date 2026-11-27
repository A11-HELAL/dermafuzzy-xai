"""Publication-Grade 300 DPI Q1 Visualization Generator for DermaFuzzy-XAI.

Uses pure headless Matplotlib (Figure + FigureCanvasAgg) without global pyplot
state leaks. Reads study outputs (`artifact_study_cases.csv`, `stats_summary.json`,
`fis_calibration.json`, and optional `cv_summary.json`) and exports 4 300-DPI PNGs:
    1. fig_roc_pr.png
    2. fig_threshold_triage.png
    3. fig_gate_and_memberships.png
    4. fig_artifact_resilience.png

Also provides `--selftest` generating synthetic CSV/JSON inputs into a temporary
directory to verify clean rendering of all 4 PNGs offline.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping

import numpy as np

# Ensure repo packages and sibling colab_trainer modules resolve from any CWD
_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_PACKAGES_DIR: Final[Path] = _REPO_ROOT / "packages"
if str(_PACKAGES_DIR) not in sys.path:
    sys.path.insert(0, str(_PACKAGES_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dermafuzzy_core.fis_engine import FISCalibration, compute_memberships, load_calibration_json
from services.colab_trainer.artifact_perturbation_study import ARTIFACT_CONDITIONS, write_cases_csv
from services.colab_trainer.stats_analysis import (
    CLINICAL_TRIAGE_THRESHOLD,
    CONDITIONS,
    DEFAULT_THRESHOLD,
    PERCENT_SCALE,
    TARGET_HIGH_SENSITIVITY,
    _validate_binary_and_scores,
    _validate_int_param,
    compute_pr_auc,
    compute_roc_auc_exact,
    compute_threshold_metrics,
    load_cases_table,
    run_stats_analysis,
)

# ------------------------------------------------------------------------------
# Module Constants & Publication Color Palette (Zero Magic Numbers Policy)
# ------------------------------------------------------------------------------

DEFAULT_DPI: Final[int] = 300
MIN_DPI: Final[int] = 1
FIGSIZE_TWO_PANEL: Final[tuple[float, float]] = (11.5, 4.8)
FIGSIZE_GATE_MEMBERSHIP: Final[tuple[float, float]] = (12.0, 6.4)
FIGSIZE_RESILIENCE: Final[tuple[float, float]] = (12.0, 5.0)

COLOR_NAVY: Final[str] = "#102a56"
COLOR_SLATE: Final[str] = "#64748b"
COLOR_AMBER: Final[str] = "#d97706"
COLOR_CRIMSON: Final[str] = "#b91c1c"
COLOR_TEAL: Final[str] = "#0f766e"
COLOR_DARK_TEXT: Final[str] = "#1e293b"
COLOR_GRID: Final[str] = "#cbd5e1"

FONT_TITLE: Final[int] = 11
FONT_LABEL: Final[int] = 10
FONT_TICK: Final[int] = 9
FONT_LEGEND: Final[int] = 8
FONT_ANNOT: Final[int] = 8
FONT_MEMBER_LEGEND: Final[int] = 7
BAR_LABEL_FONTSIZE: Final[int] = 7

LINEWIDTH_PRIMARY: Final[float] = 2.4
LINEWIDTH_SECONDARY: Final[float] = 1.8
LINEWIDTH_CURVE_SECONDARY: Final[float] = 1.6
LINEWIDTH_GATE_PRIMARY: Final[float] = 2.5
LINEWIDTH_GATE_HERMITE: Final[float] = 2.2
LINEWIDTH_CUTOFF_TRIAGE: Final[float] = 1.8
LINEWIDTH_CUTOFF_STD: Final[float] = 1.6
LINEWIDTH_MEMBERSHIP: Final[float] = 2.0
LINEWIDTH_REFERENCE: Final[float] = 1.2
BAR_EDGE_LINEWIDTH: Final[float] = 0.6

GRID_ALPHA: Final[float] = 0.45
LEGEND_ALPHA: Final[float] = 0.94
LEGEND_ALPHA_SOLID: Final[float] = 0.95
GATE_SAT_ALPHA: Final[float] = 0.50

UNIT_MIN: Final[float] = 0.0
UNIT_MAX: Final[float] = 1.0
AXIS_UPPER_PAD_RATIO: Final[float] = 1.02
GATE_YLIM_MAX: Final[float] = 1.05
AXIS_TALL_PAD_RATIO: Final[float] = 1.08

THRESHOLD_SWEEP_MIN: Final[float] = 0.05
THRESHOLD_SWEEP_MAX: Final[float] = 0.95
THRESHOLD_SWEEP_STEPS: Final[int] = 91
THRESHOLD_PCT_MIN: Final[float] = THRESHOLD_SWEEP_MIN * PERCENT_SCALE
THRESHOLD_PCT_MAX: Final[float] = THRESHOLD_SWEEP_MAX * PERCENT_SCALE
THRESHOLD_RATE_YLIM_MAX: Final[float] = 105.0
THRESHOLD_LEGEND_ANCHOR: Final[tuple[float, float]] = (0.5, -0.15)
THRESHOLD_LEGEND_NCOL: Final[int] = 3

TRIAGE_ARROW_Y: Final[float] = 18.0
TRIAGE_BOX_OFFSET_LEFT: Final[float] = 17.0
STD_BOX_OFFSET_RIGHT: Final[float] = 6.0
TRIAGE_BOX_Y: Final[float] = 12.0
TARGET_SENS_LABEL_X: Final[float] = 93.0
TARGET_SENS_LABEL_DY: Final[float] = 2.5
BBOX_PAD: Final[float] = 0.25
ARROW_LINEWIDTH: Final[float] = 1.2

GATE_GRID_STEPS: Final[int] = 201
RISK_GRID_STEPS: Final[int] = 501
SQRT_GATE_TAU_SAT: Final[float] = 0.25
GATE_TEXT_OFFSET_X: Final[float] = 0.02
GATE_TEXT_Y: Final[float] = 0.12
HERMITE_TAU_LOW: Final[float] = 0.10
HERMITE_TAU_HIGH: Final[float] = 0.38
HERMITE_TAU_SPAN: Final[float] = HERMITE_TAU_HIGH - HERMITE_TAU_LOW
HERMITE_COEFF_A: Final[float] = 3.0
HERMITE_COEFF_B: Final[float] = 2.0

MEMBER_LEGEND_NCOL: Final[int] = 3
GRID_SPEC_NROWS: Final[int] = 3
GRID_SPEC_NCOLS: Final[int] = 2
GRID_SPEC_LEFT_WIDTH: Final[float] = 1.15
GRID_SPEC_RIGHT_WIDTH: Final[float] = 1.0
GRID_SPEC_HSPACE: Final[float] = 0.48
GRID_SPEC_WSPACE: Final[float] = 0.28
FIG3_SUBPLOT_LEFT: Final[float] = 0.07
FIG3_SUBPLOT_RIGHT: Final[float] = 0.97
FIG3_SUBPLOT_TOP: Final[float] = 0.93
FIG3_SUBPLOT_BOTTOM: Final[float] = 0.10

RESILIENCE_BAR_WIDTH: Final[float] = 0.19
RESILIENCE_LABEL_PAD_FRAC: Final[float] = 0.015
RESILIENCE_SPEC_YLIM_MAX: Final[float] = 114.0
RESILIENCE_AUC_YLIM_MAX: Final[float] = 1.16
BAR_TEXT_ROTATION_DEG: Final[int] = 90

# Exact Room 4 Mamdani output membership trapezoid/triangle knots in [0, 100]
RISK_KNOTS_LOW: Final[tuple[float, float, float, float]] = (0.0, 0.0, 15.0, 35.0)
RISK_KNOTS_MED: Final[tuple[float, float, float]] = (25.0, 50.0, 75.0)
RISK_KNOTS_HIGH: Final[tuple[float, float, float, float]] = (65.0, 80.0, 100.0, 100.0)

TIER_SPECS_PLOT: Final[tuple[tuple[str, str, str, str], ...]] = (
    ("p_cnn", "Tier 1: MobileNetV2 (Raw CNN)", COLOR_SLATE, "--"),
    ("p_masked", "Tier 1b: Masked CNN", COLOR_CRIMSON, ":"),
    ("p_crisp", "Tier 2: Crisp ABCD (P50)", COLOR_AMBER, "-."),
    ("risk_sqrt", "Tier 3: DermaFuzzy-XAI", COLOR_NAVY, "-"),
)
REQUIRED_CLEAN_PLOT_KEYS: Final[tuple[str, ...]] = ("label", "p_cnn", "p_masked", "p_crisp", "risk_sqrt")

FIGURE_FILENAMES: Final[tuple[str, ...]] = (
    "fig_roc_pr.png",
    "fig_threshold_triage.png",
    "fig_gate_and_memberships.png",
    "fig_artifact_resilience.png",
)

DEFAULT_CALIBRATION_PATH: Final[str] = "artifacts/fis_calibration.json"
DEFAULT_CASES_PATH: Final[str] = "artifacts/q1_study/artifact_study_cases.csv"
DEFAULT_STATS_PATH: Final[str] = "artifacts/q1_study/stats_summary.json"
DEFAULT_OUT_DIR: Final[str] = "artifacts/q1_figures"

SELFTEST_SEED: Final[int] = 42
SELFTEST_SAMPLES_PER_CLASS: Final[int] = 35
SELFTEST_N_BOOT: Final[int] = 50
MIN_VALID_PNG_BYTES: Final[int] = 10000


@dataclass(frozen=True, slots=True)
class PlotInputsConfig:
    """Configuration DTO holding validated source artifact paths and output directory."""

    cases_csv: Path
    stats_json: Path
    calibration_json: Path
    out_dir: Path
    cv_summary_json: Path | None = None
    dpi: int = DEFAULT_DPI


# ------------------------------------------------------------------------------
# Validation, Curve & Memory-Safe Canvas Helpers
# ------------------------------------------------------------------------------

def _validate_dpi(dpi: Any) -> int:
    """Validate target raster DPI rejecting bool, np.bool_, floats, and non-positive values."""
    return _validate_int_param(dpi, "dpi", min_val=MIN_DPI)


def _validate_curve_inputs(y_bin: Any, scores: Any, name: str = "scores") -> tuple[np.ndarray, np.ndarray]:
    """Validate 1D finite binary labels (both classes present) and matching score vector."""
    return _validate_binary_and_scores(y_bin, scores, score_name=name, require_both_classes=True)


def _validate_clean_data_mapping(clean_data: Mapping[str, Any]) -> np.ndarray:
    """Validate required tier arrays inside clean_data before curve rendering."""
    if not isinstance(clean_data, Mapping):
        raise ValueError(f"clean_data must be a mapping, got {type(clean_data).__name__}")
    missing = [k for k in REQUIRED_CLEAN_PLOT_KEYS if k not in clean_data]
    if missing:
        raise ValueError(f"clean_data missing required column(s): {missing}")

    y_ref, _ = _validate_curve_inputs(clean_data["label"], clean_data["p_cnn"], name="p_cnn")
    for tier_key in ("p_masked", "p_crisp", "risk_sqrt"):
        _validate_curve_inputs(y_ref, clean_data[tier_key], name=tier_key)
    return y_ref


def _validate_stats_summary_schema(stats_summary: Any) -> Mapping[str, Any]:
    """Validate loaded stats_summary JSON object has metrics_with_ci with active conditions."""
    if not isinstance(stats_summary, Mapping):
        raise ValueError(f"stats_summary JSON must be an object, got {type(stats_summary).__name__}")
    metrics_with_ci = stats_summary.get("metrics_with_ci")
    if not isinstance(metrics_with_ci, Mapping) or not metrics_with_ci:
        raise ValueError("stats_summary JSON must contain a non-empty 'metrics_with_ci' mapping")
    if not any(c in metrics_with_ci for c in CONDITIONS):
        raise ValueError(
            f"stats_summary['metrics_with_ci'] contains none of expected conditions {CONDITIONS}"
        )
    return metrics_with_ci


def _compute_roc_curve_points(y_bin: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute step-accurate ROC (FPR, TPR) points handling ties identically to compute_roc_auc_exact."""
    y_valid, s_valid = _validate_curve_inputs(y_bin, scores, "roc_scores")
    n_pos = int(np.sum(y_valid == 1))
    n_neg = len(y_valid) - n_pos

    distinct_desc = np.sort(np.unique(s_valid))[::-1]
    fprs = [UNIT_MIN]
    tprs = [UNIT_MIN]
    for v in distinct_desc:
        pred = s_valid >= v
        tprs.append(float(np.sum(pred & (y_valid == 1)) / n_pos))
        fprs.append(float(np.sum(pred & (y_valid == 0)) / n_neg))
    if fprs[-1] < UNIT_MAX or tprs[-1] < UNIT_MAX:
        fprs.append(UNIT_MAX)
        tprs.append(UNIT_MAX)
    return np.asarray(fprs, dtype=np.float64), np.asarray(tprs, dtype=np.float64)


def _compute_pr_curve_points(y_bin: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute step-accurate PR (Recall, Precision) points matching compute_pr_auc."""
    y_valid, s_valid = _validate_curve_inputs(y_bin, scores, "pr_scores")
    n_pos = int(np.sum(y_valid == 1))
    order = np.argsort(-s_valid, kind="mergesort")
    y_sorted = y_valid[order]
    s_sorted = s_valid[order]

    distinct_idx = np.flatnonzero(np.r_[np.diff(s_sorted) != 0.0, True])
    tps = np.cumsum(y_sorted, dtype=np.float64)[distinct_idx]
    fps = (distinct_idx.astype(np.float64) + 1.0) - tps
    precisions = np.r_[UNIT_MAX, tps / (tps + fps)]
    recalls = np.r_[UNIT_MIN, tps / float(n_pos)]
    return recalls, precisions


def _style_clean_axis(ax: Any) -> None:
    """Apply uniform publication-grade spine removal, tick sizing, and dashed grid."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=FONT_TICK, colors=COLOR_DARK_TEXT)
    ax.grid(True, linestyle="--", alpha=GRID_ALPHA, color=COLOR_GRID)


def _save_and_close_figure(fig: Any, out_path: Path, dpi: int) -> Path:
    """Render headless FigureCanvasAgg cleanly, write PNG at target DPI, and free all handles."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    valid_dpi = _validate_dpi(dpi)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas: Any = None
    try:
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        fig.savefig(str(out_path), dpi=valid_dpi, bbox_inches="tight")
        return out_path
    finally:
        fig.clf()
        if "matplotlib.pyplot" in sys.modules:
            sys.modules["matplotlib.pyplot"].close(fig)
        if canvas is not None:
            del canvas
        del fig
        gc.collect()


# ------------------------------------------------------------------------------
# Figure 1: Side-by-Side ROC & Precision-Recall Curves (`clean`)
# ------------------------------------------------------------------------------

def render_fig_roc_pr(
    clean_data: Mapping[str, np.ndarray],
    out_path: Path,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """Render 2-panel side-by-side ROC and Precision-Recall curves across 4 clinical tiers."""
    from matplotlib.figure import Figure

    valid_dpi = _validate_dpi(dpi)
    y_bin = _validate_clean_data_mapping(clean_data)
    fig = Figure(figsize=FIGSIZE_TWO_PANEL, dpi=valid_dpi)
    ax_roc, ax_pr = fig.subplots(1, 2)

    for tier_key, label, color, linestyle in TIER_SPECS_PLOT:
        scores = np.asarray(clean_data[tier_key], dtype=np.float64)
        roc_auc = compute_roc_auc_exact(y_bin, scores)
        pr_auc = compute_pr_auc(y_bin, scores)
        lw = LINEWIDTH_PRIMARY if tier_key == "risk_sqrt" else LINEWIDTH_SECONDARY

        fpr, tpr = _compute_roc_curve_points(y_bin, scores)
        ax_roc.plot(fpr, tpr, color=color, linestyle=linestyle, linewidth=lw, label=f"{label} (AUC={roc_auc:.4f})")

        rec, prec = _compute_pr_curve_points(y_bin, scores)
        ax_pr.step(rec, prec, where="post", color=color, linestyle=linestyle, linewidth=lw, label=f"{label} (AP={pr_auc:.4f})")

    ax_roc.plot(
        [UNIT_MIN, UNIT_MAX],
        [UNIT_MIN, UNIT_MAX],
        color=COLOR_GRID,
        linestyle=":",
        linewidth=LINEWIDTH_REFERENCE,
        label="Random Chance (AUC=0.5000)",
    )
    ax_roc.set_title("A. Receiver Operating Characteristic (Clean)", fontsize=FONT_TITLE, fontweight="bold", color=COLOR_DARK_TEXT)
    ax_roc.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=FONT_LABEL)
    ax_roc.set_ylabel("True Positive Rate (Sensitivity)", fontsize=FONT_LABEL)
    ax_roc.set_xlim(UNIT_MIN, UNIT_MAX)
    ax_roc.set_ylim(UNIT_MIN, AXIS_UPPER_PAD_RATIO)
    _style_clean_axis(ax_roc)
    ax_roc.legend(loc="lower right", fontsize=FONT_LEGEND, frameon=True, framealpha=LEGEND_ALPHA)

    prevalence = float(np.mean(y_bin))
    ax_pr.axhline(prevalence, color=COLOR_GRID, linestyle=":", linewidth=LINEWIDTH_REFERENCE, label=f"Chance Baseline (AP={prevalence:.4f})")
    ax_pr.set_title("B. Precision-Recall Curve (Clean)", fontsize=FONT_TITLE, fontweight="bold", color=COLOR_DARK_TEXT)
    ax_pr.set_xlabel("Recall (Sensitivity)", fontsize=FONT_LABEL)
    ax_pr.set_ylabel("Precision (PPV)", fontsize=FONT_LABEL)
    ax_pr.set_xlim(UNIT_MIN, UNIT_MAX)
    ax_pr.set_ylim(UNIT_MIN, AXIS_UPPER_PAD_RATIO)
    _style_clean_axis(ax_pr)
    ax_pr.legend(loc="lower left", fontsize=FONT_LEGEND, frameon=True, framealpha=LEGEND_ALPHA)

    fig.tight_layout()
    return _save_and_close_figure(fig, out_path, valid_dpi)


# ------------------------------------------------------------------------------
# Figure 2: Sensitivity & Specificity Threshold Sweep & Clinical Triage
# ------------------------------------------------------------------------------

def _compute_threshold_curves(
    y_bin: np.ndarray,
    risk_scores: np.ndarray,
    cnn_scores: np.ndarray,
) -> tuple[np.ndarray, list[float], list[float], list[float], list[float]]:
    """Evaluate sensitivity and specificity curves across normalized decision cutoffs."""
    thresholds = np.linspace(THRESHOLD_SWEEP_MIN, THRESHOLD_SWEEP_MAX, THRESHOLD_SWEEP_STEPS)
    sens_fis, spec_fis, sens_cnn, spec_cnn = [], [], [], []
    for t in thresholds:
        m_fis = compute_threshold_metrics(y_bin, risk_scores, threshold=float(t))
        m_cnn = compute_threshold_metrics(y_bin, cnn_scores, threshold=float(t))
        sens_fis.append(m_fis["sensitivity"] * PERCENT_SCALE)
        spec_fis.append(m_fis["specificity"] * PERCENT_SCALE)
        sens_cnn.append(m_cnn["sensitivity"] * PERCENT_SCALE)
        spec_cnn.append(m_cnn["specificity"] * PERCENT_SCALE)
    return thresholds, sens_fis, spec_fis, sens_cnn, spec_cnn


def _annotate_triage_cutoffs(ax: Any, triage_pct: float, std_pct: float, target_sens_pct: float) -> None:
    """Draw Triage 35%, Standard 50%, and 90% Target Sensitivity lines with callouts."""
    ax.axvline(triage_pct, color=COLOR_AMBER, linestyle="--", linewidth=LINEWIDTH_CUTOFF_TRIAGE, label=f"Clinical Triage (R >= {triage_pct:.0f}%)")
    ax.axvline(std_pct, color=COLOR_DARK_TEXT, linestyle="-.", linewidth=LINEWIDTH_CUTOFF_STD, label=f"Standard Cutoff (R >= {std_pct:.0f}%)")
    ax.axhline(target_sens_pct, color=COLOR_CRIMSON, linestyle=":", linewidth=LINEWIDTH_CUTOFF_STD, label=f"Target Sensitivity ({target_sens_pct:.0f}%)")

    ax.annotate(
        f"Clinical Triage\n(R >= {triage_pct:.0f}%)",
        xy=(triage_pct, TRIAGE_ARROW_Y),
        xytext=(triage_pct - TRIAGE_BOX_OFFSET_LEFT, TRIAGE_BOX_Y),
        fontsize=FONT_ANNOT,
        fontweight="bold",
        color=COLOR_AMBER,
        bbox=dict(boxstyle=f"round,pad={BBOX_PAD}", facecolor="white", edgecolor=COLOR_AMBER, alpha=LEGEND_ALPHA),
        arrowprops=dict(arrowstyle="->", color=COLOR_AMBER, lw=ARROW_LINEWIDTH),
    )
    ax.annotate(
        f"Standard\n(R >= {std_pct:.0f}%)",
        xy=(std_pct, TRIAGE_ARROW_Y),
        xytext=(std_pct + STD_BOX_OFFSET_RIGHT, TRIAGE_BOX_Y),
        fontsize=FONT_ANNOT,
        fontweight="bold",
        color=COLOR_DARK_TEXT,
        bbox=dict(boxstyle=f"round,pad={BBOX_PAD}", facecolor="white", edgecolor=COLOR_DARK_TEXT, alpha=LEGEND_ALPHA),
        arrowprops=dict(arrowstyle="->", color=COLOR_DARK_TEXT, lw=ARROW_LINEWIDTH),
    )
    ax.text(
        TARGET_SENS_LABEL_X,
        target_sens_pct + TARGET_SENS_LABEL_DY,
        f"{target_sens_pct:.0f}% Sens Reference",
        fontsize=FONT_ANNOT,
        color=COLOR_CRIMSON,
        ha="right",
        fontweight="bold",
    )


def render_fig_threshold_triage(
    clean_data: Mapping[str, np.ndarray],
    out_path: Path,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """Render Sensitivity & Specificity vs cutoff highlighting Triage 35% and Standard 50%."""
    from matplotlib.figure import Figure

    valid_dpi = _validate_dpi(dpi)
    y_bin = _validate_clean_data_mapping(clean_data)
    risk_scores = np.asarray(clean_data["risk_sqrt"], dtype=np.float64)
    cnn_scores = np.asarray(clean_data["p_cnn"], dtype=np.float64)
    thresholds, sens_fis, spec_fis, sens_cnn, spec_cnn = _compute_threshold_curves(y_bin, risk_scores, cnn_scores)

    fig = Figure(figsize=FIGSIZE_TWO_PANEL, dpi=valid_dpi)
    ax = fig.subplots(1, 1)

    ax.plot(thresholds * PERCENT_SCALE, sens_fis, color=COLOR_NAVY, linewidth=LINEWIDTH_PRIMARY, label="DermaFuzzy-XAI Sensitivity")
    ax.plot(thresholds * PERCENT_SCALE, spec_fis, color=COLOR_TEAL, linewidth=LINEWIDTH_PRIMARY, label="DermaFuzzy-XAI Specificity")
    ax.plot(thresholds * PERCENT_SCALE, sens_cnn, color=COLOR_SLATE, linestyle="--", linewidth=LINEWIDTH_CURVE_SECONDARY, label="MobileNetV2 Sensitivity")
    ax.plot(thresholds * PERCENT_SCALE, spec_cnn, color=COLOR_SLATE, linestyle=":", linewidth=LINEWIDTH_CURVE_SECONDARY, label="MobileNetV2 Specificity")

    _annotate_triage_cutoffs(
        ax,
        CLINICAL_TRIAGE_THRESHOLD * PERCENT_SCALE,
        DEFAULT_THRESHOLD * PERCENT_SCALE,
        TARGET_HIGH_SENSITIVITY * PERCENT_SCALE,
    )

    ax.set_title(
        "Threshold Operating Curve & Clinical Triage Trade-Off (Clean Cohort)",
        fontsize=FONT_TITLE,
        fontweight="bold",
        color=COLOR_DARK_TEXT,
    )
    ax.set_xlabel("Decision Cutoff Threshold (% Normalized Score)", fontsize=FONT_LABEL)
    ax.set_ylabel("Classification Rate (%)", fontsize=FONT_LABEL)
    ax.set_xlim(THRESHOLD_PCT_MIN, THRESHOLD_PCT_MAX)
    ax.set_ylim(UNIT_MIN, THRESHOLD_RATE_YLIM_MAX)
    _style_clean_axis(ax)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=THRESHOLD_LEGEND_ANCHOR,
        ncol=THRESHOLD_LEGEND_NCOL,
        fontsize=FONT_LEGEND,
        frameon=True,
        framealpha=LEGEND_ALPHA_SOLID,
    )

    fig.tight_layout()
    return _save_and_close_figure(fig, out_path, valid_dpi)


# ------------------------------------------------------------------------------
# Figure 3: Saliency Trust Gates & Calibrated Membership Sets
# ------------------------------------------------------------------------------

def _plot_trust_gates_panel(ax: Any) -> None:
    """Render left panel comparing Concave Sqrt, Smooth Hermite, and Linear gates."""
    alpha = np.linspace(UNIT_MIN, UNIT_MAX, GATE_GRID_STEPS)
    g_sqrt = np.minimum(UNIT_MAX, np.sqrt(alpha / SQRT_GATE_TAU_SAT))
    t_hermite = np.clip((alpha - HERMITE_TAU_LOW) / HERMITE_TAU_SPAN, UNIT_MIN, UNIT_MAX)
    g_hermite = (t_hermite ** 2) * (HERMITE_COEFF_A - HERMITE_COEFF_B * t_hermite)

    ax.plot(alpha, g_sqrt, color=COLOR_NAVY, linewidth=LINEWIDTH_GATE_PRIMARY, label=r"Concave Sqrt: $\min(1, \sqrt{\alpha / 0.25})$")
    ax.plot(alpha, g_hermite, color=COLOR_AMBER, linestyle="--", linewidth=LINEWIDTH_GATE_HERMITE, label="Smooth Hermite S-Curve (0.10 / 0.38)")
    ax.plot(alpha, alpha, color=COLOR_SLATE, linestyle=":", linewidth=LINEWIDTH_SECONDARY, label=r"Linear Identity: $g(\alpha) = \alpha$")

    ax.axvline(SQRT_GATE_TAU_SAT, color=COLOR_NAVY, linestyle=":", alpha=GATE_SAT_ALPHA, linewidth=LINEWIDTH_REFERENCE)
    ax.text(
        SQRT_GATE_TAU_SAT + GATE_TEXT_OFFSET_X,
        GATE_TEXT_Y,
        r"$\tau_{sat} = 0.25$",
        fontsize=FONT_ANNOT,
        color=COLOR_NAVY,
        fontweight="bold",
    )
    ax.set_title("A. Saliency Trust Verification Gates", fontsize=FONT_TITLE, fontweight="bold", color=COLOR_DARK_TEXT)
    ax.set_xlabel(r"On-Lesion Grad-CAM Overlap ($\alpha$)", fontsize=FONT_LABEL)
    ax.set_ylabel(r"Confidence Multiplier $g(\alpha)$", fontsize=FONT_LABEL)
    ax.set_xlim(UNIT_MIN, UNIT_MAX)
    ax.set_ylim(UNIT_MIN, GATE_YLIM_MAX)
    _style_clean_axis(ax)
    ax.legend(loc="lower right", fontsize=FONT_LEGEND, frameon=True, framealpha=LEGEND_ALPHA)


def _plot_biomarker_memberships(ax: Any, feat_name: str, cutoffs: tuple[float, float, float]) -> None:
    """Render calibrated Partition-of-Unity Low/Medium/High curves for one biomarker."""
    x_grid = np.linspace(UNIT_MIN, UNIT_MAX, GATE_GRID_STEPS)
    vals = [compute_memberships(float(x), cutoffs) for x in x_grid]
    ax.plot(x_grid, [v.low for v in vals], color=COLOR_TEAL, linewidth=LINEWIDTH_MEMBERSHIP, label="Low")
    ax.plot(x_grid, [v.medium for v in vals], color=COLOR_AMBER, linewidth=LINEWIDTH_MEMBERSHIP, label="Medium")
    ax.plot(x_grid, [v.high for v in vals], color=COLOR_CRIMSON, linewidth=LINEWIDTH_MEMBERSHIP, label="High")

    p25, p50, p75 = cutoffs
    ax.set_title(
        f"Calibrated {feat_name} (P25={p25:.3f}, P50={p50:.3f}, P75={p75:.3f})",
        fontsize=FONT_LABEL,
        fontweight="bold",
        color=COLOR_DARK_TEXT,
    )
    ax.set_xlim(UNIT_MIN, UNIT_MAX)
    ax.set_ylim(UNIT_MIN, AXIS_TALL_PAD_RATIO)
    ax.set_ylabel("Membership", fontsize=FONT_TICK)
    _style_clean_axis(ax)
    ax.legend(loc="best", fontsize=FONT_MEMBER_LEGEND, ncol=MEMBER_LEGEND_NCOL, frameon=True, framealpha=LEGEND_ALPHA)


def _plot_mamdani_output_sets(ax: Any) -> None:
    """Render Room 4 Mamdani consequent risk membership sets over [0, 100]."""
    z = np.linspace(UNIT_MIN, PERCENT_SCALE, RISK_GRID_STEPS)
    _, _, b_l, c_l = RISK_KNOTS_LOW
    mf_low = np.clip(np.minimum(UNIT_MAX, (c_l - z) / (c_l - b_l)), UNIT_MIN, UNIT_MAX)

    a_m, b_m, c_m = RISK_KNOTS_MED
    mf_med = np.maximum(UNIT_MIN, np.minimum((z - a_m) / (b_m - a_m), (c_m - z) / (c_m - b_m)))

    a_h, b_h, _, _ = RISK_KNOTS_HIGH
    mf_high = np.clip((z - a_h) / (b_h - a_h), UNIT_MIN, UNIT_MAX)

    ax.plot(z, mf_low, color=COLOR_TEAL, linewidth=LINEWIDTH_MEMBERSHIP, label="Low Risk")
    ax.plot(z, mf_med, color=COLOR_AMBER, linewidth=LINEWIDTH_MEMBERSHIP, label="Medium Risk")
    ax.plot(z, mf_high, color=COLOR_CRIMSON, linewidth=LINEWIDTH_MEMBERSHIP, label="High Risk")
    ax.set_title("Mamdani Risk Consequent Sets", fontsize=FONT_LABEL, fontweight="bold", color=COLOR_DARK_TEXT)
    ax.set_xlabel("Melanoma Risk Score (%)", fontsize=FONT_TICK)
    ax.set_ylabel("Membership", fontsize=FONT_TICK)
    ax.set_xlim(UNIT_MIN, PERCENT_SCALE)
    ax.set_ylim(UNIT_MIN, AXIS_TALL_PAD_RATIO)
    _style_clean_axis(ax)
    ax.legend(loc="center right", fontsize=FONT_MEMBER_LEGEND, ncol=MEMBER_LEGEND_NCOL, frameon=True, framealpha=LEGEND_ALPHA)


def render_fig_gate_and_memberships(
    calibration: FISCalibration,
    out_path: Path,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """Render 2-panel figure showing trust gates (left) and calibrated/output fuzzy sets (right)."""
    from matplotlib.figure import Figure

    if not isinstance(calibration, FISCalibration):
        raise ValueError(f"calibration must be FISCalibration instance, got {type(calibration).__name__}")
    valid_dpi = _validate_dpi(dpi)

    fig = Figure(figsize=FIGSIZE_GATE_MEMBERSHIP, dpi=valid_dpi)
    gs = fig.add_gridspec(
        GRID_SPEC_NROWS,
        GRID_SPEC_NCOLS,
        width_ratios=[GRID_SPEC_LEFT_WIDTH, GRID_SPEC_RIGHT_WIDTH],
        hspace=GRID_SPEC_HSPACE,
        wspace=GRID_SPEC_WSPACE,
    )

    ax_gate = fig.add_subplot(gs[:, 0])
    _plot_trust_gates_panel(ax_gate)

    ax_b = fig.add_subplot(gs[0, 1])
    _plot_biomarker_memberships(ax_b, "B_fuzzy", calibration.get_cutoffs("B_fuzzy"))

    ax_d = fig.add_subplot(gs[1, 1])
    _plot_biomarker_memberships(ax_d, "D_verified", calibration.get_cutoffs("D_verified"))

    ax_risk = fig.add_subplot(gs[2, 1])
    _plot_mamdani_output_sets(ax_risk)

    fig.subplots_adjust(
        left=FIG3_SUBPLOT_LEFT,
        right=FIG3_SUBPLOT_RIGHT,
        top=FIG3_SUBPLOT_TOP,
        bottom=FIG3_SUBPLOT_BOTTOM,
    )
    return _save_and_close_figure(fig, out_path, valid_dpi)


# ------------------------------------------------------------------------------
# Figure 4: Artifact Resilience Grouped Bar Chart Across Conditions
# ------------------------------------------------------------------------------

def _draw_grouped_metric_panel(
    ax: Any,
    metrics_with_ci: Mapping[str, Any],
    metric_key: str,
    title: str,
    ylabel: str,
    scale: float,
    ylim: tuple[float, float],
) -> None:
    """Draw a multi-condition x 4-tier grouped bar chart with rotated bar value labels."""
    active_conds = [c for c in CONDITIONS if c in metrics_with_ci]
    x_base = np.arange(len(active_conds), dtype=np.float64)
    n_tiers = len(TIER_SPECS_PLOT)
    offsets = (np.arange(n_tiers) - (n_tiers - 1) / 2.0) * RESILIENCE_BAR_WIDTH

    for idx, (tier_key, label, color, _) in enumerate(TIER_SPECS_PLOT):
        vals = [
            float(metrics_with_ci[cond][tier_key][metric_key]["point"]) * scale
            for cond in active_conds
        ]
        bars = ax.bar(
            x_base + offsets[idx],
            vals,
            width=RESILIENCE_BAR_WIDTH,
            color=color,
            edgecolor=COLOR_DARK_TEXT,
            linewidth=BAR_EDGE_LINEWIDTH,
            label=label,
        )
        fmt = "{:.1f}" if scale > UNIT_MAX else "{:.3f}"
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + (ylim[1] - ylim[0]) * RESILIENCE_LABEL_PAD_FRAC,
                fmt.format(val),
                ha="center",
                va="bottom",
                fontsize=BAR_LABEL_FONTSIZE,
                color=COLOR_DARK_TEXT,
                rotation=BAR_TEXT_ROTATION_DEG,
            )

    ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold", color=COLOR_DARK_TEXT)
    ax.set_xticks(x_base)
    ax.set_xticklabels([c.capitalize() for c in active_conds], fontsize=FONT_LABEL)
    ax.set_ylabel(ylabel, fontsize=FONT_LABEL)
    ax.set_ylim(*ylim)
    _style_clean_axis(ax)


def render_fig_artifact_resilience(
    stats_summary: Mapping[str, Any],
    out_path: Path,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """Render 2-panel grouped bar chart of Benign Specificity & ROC-AUC across artifact conditions."""
    from matplotlib.figure import Figure

    valid_dpi = _validate_dpi(dpi)
    metrics_with_ci = _validate_stats_summary_schema(stats_summary)
    fig = Figure(figsize=FIGSIZE_RESILIENCE, dpi=valid_dpi)
    ax_spec, ax_auc = fig.subplots(1, 2)

    _draw_grouped_metric_panel(
        ax_spec,
        metrics_with_ci,
        metric_key="specificity",
        title="A. Benign Specificity Under Peripheral Artifacts",
        ylabel="Benign Specificity (%)",
        scale=PERCENT_SCALE,
        ylim=(UNIT_MIN, RESILIENCE_SPEC_YLIM_MAX),
    )
    _draw_grouped_metric_panel(
        ax_auc,
        metrics_with_ci,
        metric_key="roc_auc",
        title="B. Discriminative ROC-AUC Under Peripheral Artifacts",
        ylabel="ROC-AUC",
        scale=UNIT_MAX,
        ylim=(UNIT_MIN, RESILIENCE_AUC_YLIM_MAX),
    )

    ax_spec.legend(loc="lower left", fontsize=FONT_LEGEND, frameon=True, framealpha=LEGEND_ALPHA)
    fig.tight_layout()
    return _save_and_close_figure(fig, out_path, valid_dpi)


# ------------------------------------------------------------------------------
# Top-Level Orchestration & Selftest
# ------------------------------------------------------------------------------

def render_all_q1_figures(cfg: PlotInputsConfig) -> dict[str, Path]:
    """Validate inputs and render all 4 publication-grade 300 DPI Q1 figures."""
    valid_dpi = _validate_dpi(cfg.dpi)
    if not cfg.cases_csv.is_file():
        raise FileNotFoundError(f"Cases CSV not found: {cfg.cases_csv}")
    if not cfg.stats_json.is_file():
        raise FileNotFoundError(f"Stats JSON not found: {cfg.stats_json}")
    if not cfg.calibration_json.is_file():
        raise FileNotFoundError(f"Calibration JSON not found: {cfg.calibration_json}")

    by_condition = load_cases_table(cfg.cases_csv)
    if "clean" not in by_condition:
        raise ValueError("Cases CSV must contain condition == 'clean'")
    _validate_clean_data_mapping(by_condition["clean"])

    stats_summary = json.loads(cfg.stats_json.read_text(encoding="utf-8"))
    _validate_stats_summary_schema(stats_summary)
    calibration = load_calibration_json(cfg.calibration_json)

    if cfg.cv_summary_json is not None:
        if not cfg.cv_summary_json.is_file():
            raise FileNotFoundError(f"Optional CV summary JSON not found: {cfg.cv_summary_json}")
        cv_obj = json.loads(cfg.cv_summary_json.read_text(encoding="utf-8"))
        if not isinstance(cv_obj, Mapping):
            raise ValueError("Optional CV summary JSON must decode to a JSON object")

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "fig_roc_pr": render_fig_roc_pr(by_condition["clean"], cfg.out_dir / FIGURE_FILENAMES[0], dpi=valid_dpi),
        "fig_threshold_triage": render_fig_threshold_triage(
            by_condition["clean"], cfg.out_dir / FIGURE_FILENAMES[1], dpi=valid_dpi
        ),
        "fig_gate_and_memberships": render_fig_gate_and_memberships(
            calibration, cfg.out_dir / FIGURE_FILENAMES[2], dpi=valid_dpi
        ),
        "fig_artifact_resilience": render_fig_artifact_resilience(
            stats_summary, cfg.out_dir / FIGURE_FILENAMES[3], dpi=valid_dpi
        ),
    }


def _generate_selftest_cases_csv(csv_path: Path) -> Path:
    """Create deterministic synthetic study CSV covering all 4 conditions and 6 tiers."""
    rng = np.random.default_rng(SELFTEST_SEED)
    rows = []
    for cond in ARTIFACT_CONDITIONS:
        for idx in range(2 * SELFTEST_SAMPLES_PER_CLASS):
            lbl = 1 if idx >= SELFTEST_SAMPLES_PER_CLASS else 0
            p_cnn = float(rng.uniform(0.62, 0.94) if lbl == 1 else rng.uniform(0.10, 0.42))
            if cond != "clean" and lbl == 0 and idx < 10:
                p_cnn = float(rng.uniform(0.75, 0.92))
            p_masked = float(rng.uniform(0.58, 0.90) if lbl == 1 else rng.uniform(0.12, 0.40))
            p_crisp = float(rng.choice([0.50, 0.75, 1.0]) if lbl == 1 else rng.choice([0.0, 0.25, 0.50]))
            risk_sqrt = float(rng.uniform(50.0, 92.0) if lbl == 1 else rng.uniform(12.0, 42.0))
            risk_hermite = float(rng.uniform(52.0, 94.0) if lbl == 1 else rng.uniform(10.0, 39.0))
            risk_no_bg = float(rng.uniform(48.0, 88.0) if lbl == 1 else rng.uniform(15.0, 46.0))
            rows.append({
                "image_id": f"SELFTEST_{idx:03d}",
                "label": lbl,
                "condition": cond,
                "p_cnn": f"{p_cnn:.6f}",
                "p_masked": f"{p_masked:.6f}",
                "p_crisp": f"{p_crisp:.6f}",
                "d_ver_sqrt": f"{p_cnn * 0.8:.6f}",
                "d_ver_hermite": f"{p_cnn * 0.85:.6f}",
                "alpha": f"{0.78 if lbl == 1 else 0.15:.6f}",
                "A": f"{0.68 if lbl == 1 else 0.22:.6f}",
                "B_geo": f"{0.70 if lbl == 1 else 0.24:.6f}",
                "B_grad": f"{0.72 if lbl == 1 else 0.26:.6f}",
                "B_fuzzy": f"{0.71 if lbl == 1 else 0.25:.6f}",
                "C": f"{0.65 if lbl == 1 else 0.19:.6f}",
                "risk_sqrt": f"{risk_sqrt:.6f}",
                "risk_hermite": f"{risk_hermite:.6f}",
                "risk_no_bgrad": f"{risk_no_bg:.6f}",
                "fallback_used": 0,
                "mask_iou_vs_clean": f"{1.0 if cond == 'clean' else 0.93:.6f}",
            })
    return write_cases_csv(rows, csv_path)


def run_selftest() -> None:
    """Render all 4 Q1 figures into a temporary directory and verify PNG outputs."""
    calib_path = _REPO_ROOT / DEFAULT_CALIBRATION_PATH
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        cases_csv = _generate_selftest_cases_csv(tmp_dir / "artifact_study_cases.csv")
        stats_json, _ = run_stats_analysis(cases_csv, tmp_dir, n_boot=SELFTEST_N_BOOT, seed=SELFTEST_SEED)
        out_dir = tmp_dir / "figures"

        outputs = render_all_q1_figures(
            PlotInputsConfig(
                cases_csv=cases_csv,
                stats_json=stats_json,
                calibration_json=calib_path,
                out_dir=out_dir,
            )
        )
        for name, fig_path in outputs.items():
            assert fig_path.is_file(), f"Missing generated figure: {name}"
            assert fig_path.stat().st_size > MIN_VALID_PNG_BYTES, f"Empty/tiny PNG: {fig_path}"

    print(f"SELFTEST PASSED: rendered all {len(FIGURE_FILENAMES)} 300 DPI Q1 PNG figures cleanly.")


def main() -> None:
    """CLI entrypoint for rendering publication-grade Q1 figures and offline selftest."""
    parser = argparse.ArgumentParser(
        description="DermaFuzzy-XAI Q1 300 DPI Publication Figures Generator"
    )
    parser.add_argument("--selftest", action="store_true", help="Render 4 PNGs from synthetic data in temp dir")
    parser.add_argument("--cases", type=str, default=DEFAULT_CASES_PATH, help="Path to artifact_study_cases.csv")
    parser.add_argument("--stats", type=str, default=DEFAULT_STATS_PATH, help="Path to stats_summary.json")
    parser.add_argument("--calibration", type=str, default=DEFAULT_CALIBRATION_PATH, help="Path to fis_calibration.json")
    parser.add_argument("--cv-summary", type=str, default=None, help="Optional path to cv_summary.json")
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR, help="Output directory for PNG figures")

    args = parser.parse_args()
    if args.selftest:
        run_selftest()
        return

    outputs = render_all_q1_figures(
        PlotInputsConfig(
            cases_csv=Path(args.cases),
            stats_json=Path(args.stats),
            calibration_json=Path(args.calibration),
            out_dir=Path(args.out_dir),
            cv_summary_json=Path(args.cv_summary) if args.cv_summary else None,
        )
    )
    for fig_path in outputs.values():
        print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
