"""Q1 Empirical Artifact-Perturbation & Ablation Study (Colab & Local Selftest).

Evaluates trained MobileNetV2 + Mamdani FIS across four image conditions:
    ("clean", "ruler", "ink", "gel")
Replicates the exact `train_ham10000.py` Stage 1 test split
(StratifiedGroupKFold n_splits=5, shuffle=True, random_state=42, grouped by
lesion_id, 1:1 balanced test_df) and exact batched @tf.function Pre-Sigmoid
4-Way TTA Grad-CAM (`out_relu` + `logits` + `cam_abs` fallback) so that
condition == "clean" reproduces v1.0-baseline identically.

Also provides a pure NumPy/OpenCV `--selftest` mode with zero TensorFlow or
external dataset dependencies.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import math
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import cv2
import numpy as np

# Ensure packages/ is importable regardless of invocation directory
_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_PACKAGES_DIR: Final[Path] = _REPO_ROOT / "packages"
if str(_PACKAGES_DIR) not in sys.path:
    sys.path.insert(0, str(_PACKAGES_DIR))

from dermafuzzy_core.biomarkers import BiomarkerResult, extract_abcd_biomarkers
from dermafuzzy_core.config import DermaConfig, load_config
from dermafuzzy_core.evaluation import evaluate_crisp_abcd_baseline
from dermafuzzy_core.fis_engine import FISCalibration, evaluate_mamdani_fis, load_calibration_json
from dermafuzzy_core.preprocessing import PreprocessingResult, preprocess_image

# ------------------------------------------------------------------------------
# Module Constants (Zero Magic Numbers Policy)
# ------------------------------------------------------------------------------

ARTIFACT_CONDITIONS: Final[tuple[str, ...]] = ("clean", "ruler", "ink", "gel")
MASKED_NEUTRAL_GRAY_UINT8: Final[int] = 128
HASH_SEED_BYTES: Final[int] = 8
UINT8_MIN: Final[int] = 0
UINT8_MAX: Final[int] = 255
BINARY_LABELS: Final[tuple[int, int]] = (0, 1)

# Overlay parameters: Ruler
RULER_BAND_MIN_PX: Final[int] = 5
RULER_BAND_MAX_PX: Final[int] = 9
RULER_MAX_BAND_DIVISOR: Final[int] = 4
RULER_MIN_STEP_PX: Final[int] = 2
RULER_TONE_MIN: Final[int] = 15
RULER_TONE_MAX: Final[int] = 55
RULER_TICK_MIN_VAL: Final[int] = 205
RULER_TICK_MAX_VAL: Final[int] = 240
RULER_TICK_STEP_MIN_PX: Final[int] = 4
RULER_TICK_STEP_MAX_PX: Final[int] = 7
RULER_MAJOR_TICK_PERIOD: Final[int] = 5
RULER_MAJOR_TICK_RATIO: Final[float] = 0.85
RULER_MINOR_TICK_RATIO: Final[float] = 0.45

# Overlay parameters: Gentian-Violet Surgical Ink
INK_COLOR_RGB: Final[tuple[int, int, int]] = (45, 25, 120)
INK_DILATION_MARGIN_PX: Final[int] = 6
INK_STROKE_LEN_MIN_RATIO: Final[float] = 0.14
INK_STROKE_LEN_MAX_RATIO: Final[float] = 0.28
INK_MIN_STROKE_LEN_PX: Final[float] = 3.0
INK_STROKE_THICKNESS_MIN_PX: Final[int] = 4
INK_STROKE_THICKNESS_MAX_PX: Final[int] = 8
INK_MAX_THICKNESS_DIVISOR: Final[int] = 8
INK_FEATHER_KERNEL_SIZE: Final[int] = 5
INK_FEATHER_SIGMA: Final[float] = 1.5
INK_MAX_ALPHA: Final[float] = 0.85
INK_CORNER_RADIUS_RATIO: Final[float] = 0.08
INK_MIN_CORNER_RADIUS_PX: Final[int] = 2
INK_ALPHA_EPS: Final[float] = 1e-6

# Overlay parameters: Gel Bubble
GEL_ALPHA: Final[float] = 0.35
GEL_WHITE_VAL: Final[float] = 255.0
GEL_AXIS_MIN_RATIO: Final[float] = 0.07
GEL_AXIS_MAX_RATIO: Final[float] = 0.14
GEL_MIN_AXIS_PX: Final[int] = 2
GEL_DIST_TRANSFORM_MASK_SIZE: Final[int] = 3
GEL_ANGLE_MAX_DEG: Final[float] = 180.0
GEL_FULL_CIRCLE_DEG: Final[float] = 360.0

# Saliency Trust Gate Constants (Hermite S-Curve)
HERMITE_TAU_LOW: Final[float] = 0.10
HERMITE_TAU_HIGH: Final[float] = 0.38
HERMITE_TAU_SPAN: Final[float] = HERMITE_TAU_HIGH - HERMITE_TAU_LOW

# Grad-CAM & Splitting Constants (matching train_ham10000.py)
GRADCAM_EPS: Final[float] = 1e-8
OUTER_KFOLD_SPLITS: Final[int] = 5
BENIGN_OVERSAMPLE_FACTOR: Final[int] = 2
TTA_WEIGHT: Final[float] = 0.25

# Default CLI Arguments
DEFAULT_DATA_DIR: Final[str] = "data/HAM10000_images"
DEFAULT_METADATA_CSV: Final[str] = "data/HAM10000_metadata.csv"
DEFAULT_MODEL_PATH: Final[str] = "artifacts/mobilenetv2_ham10000.keras"
DEFAULT_CALIBRATION_PATH: Final[str] = "artifacts/fis_calibration.json"
DEFAULT_OUT_DIR: Final[str] = "artifacts/q1_study"
DEFAULT_BATCH_SIZE: Final[int] = 32
CASES_CSV_FILENAME: Final[str] = "artifact_study_cases.csv"

CSV_FIELDNAMES: Final[tuple[str, ...]] = (
    "image_id",
    "label",
    "condition",
    "p_cnn",
    "p_masked",
    "p_crisp",
    "d_ver_sqrt",
    "d_ver_hermite",
    "alpha",
    "A",
    "B_geo",
    "B_grad",
    "B_fuzzy",
    "C",
    "risk_sqrt",
    "risk_hermite",
    "risk_no_bgrad",
    "fallback_used",
    "mask_iou_vs_clean",
)

FLOAT_CSV_COLUMNS: Final[tuple[str, ...]] = (
    "p_cnn",
    "p_masked",
    "p_crisp",
    "d_ver_sqrt",
    "d_ver_hermite",
    "alpha",
    "A",
    "B_geo",
    "B_grad",
    "B_fuzzy",
    "C",
    "risk_sqrt",
    "risk_hermite",
    "risk_no_bgrad",
    "mask_iou_vs_clean",
)

# Selftest Constants
SELFTEST_IMG_SIZE: Final[int] = 224
SELFTEST_MASK_RADIUS: Final[int] = 48
SELFTEST_SEED: Final[int] = 42
SELFTEST_CASE_COUNT: Final[int] = 6
SELFTEST_SKIN_RGB: Final[tuple[int, int, int]] = (190, 190, 190)
SELFTEST_LESION_RGB: Final[tuple[int, int, int]] = (70, 45, 35)
SELFTEST_HEATMAP_SIZE: Final[int] = 32
SELFTEST_HEATMAP_START: Final[int] = 10
SELFTEST_HEATMAP_END: Final[int] = 22
SELFTEST_POS_PCNN: Final[float] = 0.85
SELFTEST_NEG_PCNN: Final[float] = 0.20
SELFTEST_POS_MASKED: Final[float] = 0.80
SELFTEST_NEG_MASKED: Final[float] = 0.15


@dataclass(frozen=True, slots=True)
class CaseEvaluationInput:
    """Bundled per-case inputs for evaluating all 6 ablation variants."""

    image_id: str
    label: int
    condition: str
    p_cnn: float
    p_masked: float
    heatmap: np.ndarray
    prep_clean: PreprocessingResult
    prep_cond: PreprocessingResult


@dataclass(frozen=True, slots=True)
class StudyRunConfig:
    """Configuration DTO for executing the Colab empirical perturbation study."""

    data_dir: Path
    metadata_csv: Path
    model_path: Path
    calibration_path: Path
    out_dir: Path
    batch_size: int = DEFAULT_BATCH_SIZE
    max_images: int | None = None


@dataclass(frozen=True, slots=True)
class CaseEvaluationResult:
    """Immutable per-case evaluation output across all 6 variants for one condition."""

    image_id: str
    label: int
    condition: str
    p_cnn: float
    p_masked: float
    p_crisp: float
    d_ver_sqrt: float
    d_ver_hermite: float
    alpha: float
    A: float
    B_geo: float
    B_grad: float
    B_fuzzy: float
    C: float
    risk_sqrt: float
    risk_hermite: float
    risk_no_bgrad: float
    fallback_used: bool
    mask_iou_vs_clean: float

    def to_csv_row(self) -> dict[str, Any]:
        """Convert record to dictionary ordered according to CSV_FIELDNAMES."""
        return {
            "image_id": self.image_id,
            "label": int(self.label),
            "condition": self.condition,
            "p_cnn": f"{self.p_cnn:.8f}",
            "p_masked": f"{self.p_masked:.8f}",
            "p_crisp": f"{self.p_crisp:.8f}",
            "d_ver_sqrt": f"{self.d_ver_sqrt:.8f}",
            "d_ver_hermite": f"{self.d_ver_hermite:.8f}",
            "alpha": f"{self.alpha:.8f}",
            "A": f"{self.A:.8f}",
            "B_geo": f"{self.B_geo:.8f}",
            "B_grad": f"{self.B_grad:.8f}",
            "B_fuzzy": f"{self.B_fuzzy:.8f}",
            "C": f"{self.C:.8f}",
            "risk_sqrt": f"{self.risk_sqrt:.8f}",
            "risk_hermite": f"{self.risk_hermite:.8f}",
            "risk_no_bgrad": f"{self.risk_no_bgrad:.8f}",
            "fallback_used": int(self.fallback_used),
            "mask_iou_vs_clean": f"{self.mask_iou_vs_clean:.8f}",
        }


# ------------------------------------------------------------------------------
# Input Validation Helpers
# ------------------------------------------------------------------------------

def _is_bool_like(val: Any) -> bool:
    """Return True if value is a Python bool or NumPy np.bool_."""
    return isinstance(val, (bool, np.bool_))


def _validate_rgb_and_mask(
    rgb: Any,
    mask: Any,
    rng: Any,
) -> tuple[np.ndarray, np.ndarray, np.random.Generator]:
    """Validate 3D uint8 RGB image, matching 2D finite binary mask, and NumPy RNG."""
    if not isinstance(rgb, np.ndarray):
        raise ValueError(f"rgb must be a numpy.ndarray, got {type(rgb).__name__}")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"rgb must have shape (H, W, 3), got {rgb.shape}")
    if rgb.dtype != np.uint8:
        raise ValueError(f"rgb must have dtype uint8, got {rgb.dtype}")
    if rgb.shape[0] <= 0 or rgb.shape[1] <= 0:
        raise ValueError(f"rgb spatial dimensions must be positive, got {rgb.shape}")

    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise ValueError("mask must be a 2D numpy.ndarray")
    if mask.dtype == bool:
        bin_mask = mask
    else:
        if not np.all(np.isfinite(mask)):
            raise ValueError("mask contains NaN or Inf values")
        bin_mask = mask > 0
    if mask.shape[:2] != rgb.shape[:2]:
        raise ValueError(
            f"mask shape {mask.shape[:2]} must match rgb spatial shape {rgb.shape[:2]}"
        )

    if not isinstance(rng, np.random.Generator):
        raise ValueError(f"rng must be a numpy.random.Generator, got {type(rng).__name__}")

    return rgb, bin_mask, rng


def _validate_unit_float(val: Any, name: str) -> float:
    """Validate finite float in [0.0, 1.0], rejecting bool, NaN, and Inf."""
    if _is_bool_like(val) or not isinstance(val, (int, float, np.floating, np.integer)):
        raise ValueError(f"{name} must be a numeric float, got {type(val).__name__}")
    fval = float(val)
    if not math.isfinite(fval):
        raise ValueError(f"{name} must be finite, got {fval}")
    if not (0.0 <= fval <= 1.0):
        raise ValueError(f"{name} must be in [0.0, 1.0], got {fval}")
    return fval


def _make_deterministic_rng(seed: int, image_id: str, condition: str) -> np.random.Generator:
    """Derive a cross-run deterministic RNG uniquely keyed by (seed, image_id, condition)."""
    digest = hashlib.sha256(f"{seed}:{image_id}:{condition}".encode("utf-8")).digest()
    seed_int = int.from_bytes(digest[:HASH_SEED_BYTES], byteorder="little", signed=False)
    return np.random.default_rng(seed_int)


# ------------------------------------------------------------------------------
# Mask-Aware Peripheral Artifact Overlays
# ------------------------------------------------------------------------------

def _draw_ruler_ticks(
    strip: np.ndarray,
    edge: str,
    band: int,
    step: int,
    tick_val: int,
) -> None:
    """Draw millimeter tick marks (major every 5th tick) along a border ruler strip."""
    h, w = strip.shape[:2]
    major_len = max(1, min(band, int(round(band * RULER_MAJOR_TICK_RATIO))))
    minor_len = max(1, min(band, int(round(band * RULER_MINOR_TICK_RATIO))))
    length = w if edge in ("top", "bottom") else h

    for idx, pos in enumerate(range(step // 2, length, step)):
        t_len = major_len if (idx % RULER_MAJOR_TICK_PERIOD == 0) else minor_len
        if edge == "top":
            strip[band - t_len : band, pos] = tick_val
        elif edge == "bottom":
            strip[h - band : h - band + t_len, pos] = tick_val
        elif edge == "left":
            strip[pos, band - t_len : band] = tick_val
        else:
            strip[pos, w - band : w - band + t_len] = tick_val


def overlay_ruler(
    rgb: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Overlay a dark dermoscopic calibration ruler with millimeter ticks on lowest-overlap edge.

    Parameters
    ----------
    rgb : np.ndarray
        Input RGB image of shape (H, W, 3), dtype uint8 (not mutated).
    mask : np.ndarray
        2D lesion mask of shape (H, W).
    rng : np.random.Generator
        Deterministic NumPy random generator.

    Returns
    -------
    np.ndarray
        New uint8 RGB image with peripheral ruler strip overlaid.
    """
    src, bin_mask, gen = _validate_rgb_and_mask(rgb, mask, rng)
    h, w = src.shape[:2]
    out = src.copy()

    max_band = max(1, min(h, w) // RULER_MAX_BAND_DIVISOR)
    band = min(max_band, int(gen.integers(RULER_BAND_MIN_PX, RULER_BAND_MAX_PX + 1)))
    tone = int(gen.integers(RULER_TONE_MIN, RULER_TONE_MAX + 1))
    tick_val = int(gen.integers(RULER_TICK_MIN_VAL, RULER_TICK_MAX_VAL + 1))
    step = min(
        max(RULER_MIN_STEP_PX, max_band),
        int(gen.integers(RULER_TICK_STEP_MIN_PX, RULER_TICK_STEP_MAX_PX + 1)),
    )

    edge_overlaps = {
        "top": int(bin_mask[:band, :].sum()),
        "bottom": int(bin_mask[-band:, :].sum()),
        "left": int(bin_mask[:, :band].sum()),
        "right": int(bin_mask[:, -band:].sum()),
    }
    min_overlap = min(edge_overlaps.values())
    candidate_edges = [edge for edge, ov in edge_overlaps.items() if ov == min_overlap]
    chosen_edge = candidate_edges[int(gen.integers(0, len(candidate_edges)))]

    if chosen_edge == "top":
        out[:band, :, :] = tone
    elif chosen_edge == "bottom":
        out[-band:, :, :] = tone
    elif chosen_edge == "left":
        out[:, :band, :] = tone
    else:
        out[:, -band:, :] = tone

    _draw_ruler_ticks(out, chosen_edge, band, step, tick_val)
    return out


def _draw_feathered_corner_patch(
    h: int,
    w: int,
    gen: np.random.Generator,
    bin_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Create a normalized alpha mask at a corner (preferring skin pixels if available)."""
    alpha = np.zeros((h, w), dtype=np.float32)
    radius = max(INK_MIN_CORNER_RADIUS_PX, int(round(min(h, w) * INK_CORNER_RADIUS_RATIO)))
    corners = ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1))

    if bin_mask is not None and np.any(~bin_mask):
        bg_mask = ~bin_mask
        best_cov = -1
        best_corners: list[tuple[int, int]] = []
        for cx, cy in corners:
            y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
            x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
            cov = int(bg_mask[y0:y1, x0:x1].sum())
            if cov > best_cov:
                best_cov = cov
                best_corners = [(cx, cy)]
            elif cov == best_cov:
                best_corners.append((cx, cy))
        cx, cy = best_corners[int(gen.integers(0, len(best_corners)))]
    else:
        cx, cy = corners[int(gen.integers(0, len(corners)))]

    cv2.circle(alpha, (cx, cy), radius, 1.0, thickness=-1)
    k = INK_FEATHER_KERNEL_SIZE
    alpha = cv2.GaussianBlur(alpha, (k, k), INK_FEATHER_SIGMA)
    max_a = float(alpha.max())
    if max_a > INK_ALPHA_EPS:
        alpha = alpha / max_a
    return alpha * INK_MAX_ALPHA


def _draw_peripheral_ink_stroke(
    h: int,
    w: int,
    safe_region: np.ndarray,
    gen: np.random.Generator,
) -> np.ndarray:
    """Render feathered surgical ink stroke anchored strictly inside dilated-safe skin."""
    alpha_map = np.zeros((h, w), dtype=np.float32)
    safe_coords = np.argwhere(safe_region)
    anchor_y, anchor_x = safe_coords[int(gen.integers(0, len(safe_coords)))]

    stroke_len = max(
        INK_MIN_STROKE_LEN_PX,
        float(gen.uniform(INK_STROKE_LEN_MIN_RATIO, INK_STROKE_LEN_MAX_RATIO)) * float(min(h, w)),
    )
    angle_rad = float(gen.uniform(0.0, 2.0 * math.pi))
    end_x = int(np.clip(round(anchor_x + stroke_len * math.cos(angle_rad)), 0, w - 1))
    end_y = int(np.clip(round(anchor_y + stroke_len * math.sin(angle_rad)), 0, h - 1))
    thickness = max(
        1,
        min(
            max(1, min(h, w) // INK_MAX_THICKNESS_DIVISOR),
            int(gen.integers(INK_STROKE_THICKNESS_MIN_PX, INK_STROKE_THICKNESS_MAX_PX + 1)),
        ),
    )

    cv2.line(alpha_map, (int(anchor_x), int(anchor_y)), (end_x, end_y), 1.0, thickness=thickness)
    alpha_map = cv2.GaussianBlur(
        alpha_map,
        (INK_FEATHER_KERNEL_SIZE, INK_FEATHER_KERNEL_SIZE),
        INK_FEATHER_SIGMA,
    )
    # Strictly zero any blur bleed inside the 6px lesion margin
    alpha_map[~safe_region] = 0.0
    max_val = float(alpha_map.max())
    if max_val > INK_ALPHA_EPS:
        return (alpha_map / max_val) * INK_MAX_ALPHA
    alpha_map[int(anchor_y), int(anchor_x)] = INK_MAX_ALPHA
    return alpha_map


def overlay_ink(
    rgb: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Overlay gentian-violet surgical marking ink strictly on peripheral skin outside lesion.

    Maintains at least 6 px margin from `mask > 0` via morphological dilation;
    falls back to a feathered corner patch if the lesion fills the entire frame.

    Parameters
    ----------
    rgb : np.ndarray
        Input RGB image of shape (H, W, 3), dtype uint8 (not mutated).
    mask : np.ndarray
        2D lesion mask of shape (H, W).
    rng : np.random.Generator
        Deterministic NumPy random generator.

    Returns
    -------
    np.ndarray
        New uint8 RGB image with peripheral surgical ink overlaid.
    """
    src, bin_mask, gen = _validate_rgb_and_mask(rgb, mask, rng)
    h, w = src.shape[:2]

    dil_ksize = 2 * INK_DILATION_MARGIN_PX + 1
    k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil_ksize, dil_ksize))
    safe_region = ~(cv2.dilate(bin_mask.astype(np.uint8), k_dil) > 0)

    if np.any(safe_region):
        alpha_map = _draw_peripheral_ink_stroke(h, w, safe_region, gen)
    else:
        alpha_map = _draw_feathered_corner_patch(h, w, gen, bin_mask=bin_mask)
        if np.any(~bin_mask):
            alpha_map[bin_mask] = 0.0

    ink_color = np.array(INK_COLOR_RGB, dtype=np.float32)
    blended = src.astype(np.float32) * (1.0 - alpha_map[..., None]) + ink_color * alpha_map[..., None]
    return np.clip(np.round(blended), UINT8_MIN, UINT8_MAX).astype(np.uint8)


def overlay_gel(
    rgb: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Overlay translucent white immersion gel bubble (alpha=0.35) on peripheral skin.

    Parameters
    ----------
    rgb : np.ndarray
        Input RGB image of shape (H, W, 3), dtype uint8 (not mutated).
    mask : np.ndarray
        2D lesion mask of shape (H, W).
    rng : np.random.Generator
        Deterministic NumPy random generator.

    Returns
    -------
    np.ndarray
        New uint8 RGB image with translucent peripheral gel bubble overlaid.
    """
    src, bin_mask, gen = _validate_rgb_and_mask(rgb, mask, rng)
    h, w = src.shape[:2]

    bg_mask = ~bin_mask
    bg_coords = np.argwhere(bg_mask)

    ax_w = max(GEL_MIN_AXIS_PX, int(round(float(gen.uniform(GEL_AXIS_MIN_RATIO, GEL_AXIS_MAX_RATIO)) * w)))
    ax_h = max(GEL_MIN_AXIS_PX, int(round(float(gen.uniform(GEL_AXIS_MIN_RATIO, GEL_AXIS_MAX_RATIO)) * h)))
    angle = float(gen.uniform(0.0, GEL_ANGLE_MAX_DEG))

    if len(bg_coords) > 0:
        dist_map = cv2.distanceTransform(bg_mask.astype(np.uint8), cv2.DIST_L2, GEL_DIST_TRANSFORM_MASK_SIZE)
        outer_thresh = max(1.0, float(min(ax_w, ax_h)))
        outer_coords = np.argwhere(dist_map >= outer_thresh)
        pool = outer_coords if len(outer_coords) > 0 else bg_coords
        cy, cx = pool[int(gen.integers(0, len(pool)))]
    else:
        corners = ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1))
        cx, cy = corners[int(gen.integers(0, len(corners)))]

    gel_region = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(
        gel_region,
        (int(cx), int(cy)),
        (ax_w, ax_h),
        angle,
        0.0,
        GEL_FULL_CIRCLE_DEG,
        1,
        thickness=-1,
    )
    gel_bool = gel_region > 0
    if np.any(bg_mask):
        gel_bool = gel_bool & bg_mask
        if not np.any(gel_bool) and len(bg_coords) > 0:
            gel_bool[int(cy), int(cx)] = True

    blend_mask = gel_bool.astype(np.float32)[..., None] * GEL_ALPHA
    blended = src.astype(np.float32) * (1.0 - blend_mask) + GEL_WHITE_VAL * blend_mask
    return np.clip(np.round(blended), UINT8_MIN, UINT8_MAX).astype(np.uint8)


def apply_artifact_condition(
    rgb: np.ndarray,
    mask: np.ndarray,
    condition: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Dispatch deterministic peripheral artifact overlay for a valid study condition."""
    if condition not in ARTIFACT_CONDITIONS:
        raise ValueError(f"Unsupported condition '{condition}'; choose from {ARTIFACT_CONDITIONS}")
    src, bin_mask, gen = _validate_rgb_and_mask(rgb, mask, rng)
    if condition == "clean":
        return src.copy()
    if condition == "ruler":
        return overlay_ruler(src, bin_mask, gen)
    if condition == "ink":
        return overlay_ink(src, bin_mask, gen)
    return overlay_gel(src, bin_mask, gen)


# ------------------------------------------------------------------------------
# Ablation Variant Helpers & Metrics
# ------------------------------------------------------------------------------

def compute_hermite_verified_confidence(p_cnn: Any, alpha: Any) -> float:
    """Compute Smooth Hermite S-Curve saliency trust gate confidence.

    Uses tau_low = 0.10 and tau_high = 0.38:
        t = clip((alpha - 0.10) / 0.28, 0.0, 1.0)
        g(alpha) = t^2 * (3 - 2*t)
        D_verified = clip(p_cnn * g(alpha), 0.0, 1.0)
    """
    fp = _validate_unit_float(p_cnn, "p_cnn")
    fa = _validate_unit_float(alpha, "alpha")
    t = float(np.clip((fa - HERMITE_TAU_LOW) / HERMITE_TAU_SPAN, 0.0, 1.0))
    gate = t * t * (3.0 - 2.0 * t)
    return float(np.clip(fp * gate, 0.0, 1.0))


def compute_mask_iou(mask_a: Any, mask_b: Any) -> float:
    """Compute Intersection-over-Union between two binary 2D lesion masks."""
    if not isinstance(mask_a, np.ndarray) or not isinstance(mask_b, np.ndarray):
        raise ValueError("Both masks must be numpy.ndarray instances")
    if mask_a.ndim != 2 or mask_b.ndim != 2 or mask_a.shape != mask_b.shape:
        raise ValueError(f"Masks must be matching 2D arrays, got {mask_a.shape} and {mask_b.shape}")
    if mask_a.dtype != bool and not np.all(np.isfinite(mask_a)):
        raise ValueError("mask_a contains non-finite values")
    if mask_b.dtype != bool and not np.all(np.isfinite(mask_b)):
        raise ValueError("mask_b contains non-finite values")

    a_bin = mask_a > 0
    b_bin = mask_b > 0
    inter = int(np.logical_and(a_bin, b_bin).sum())
    union = int(np.logical_or(a_bin, b_bin).sum())
    return 1.0 if union == 0 else float(inter / union)


def neutralize_off_lesion_pixels(clean_rgb: np.ndarray, lesion_mask: np.ndarray) -> np.ndarray:
    """Neutralize off-lesion pixels (mask == 0) to mid-gray 128 without mutating input."""
    if not isinstance(clean_rgb, np.ndarray) or clean_rgb.ndim != 3 or clean_rgb.shape[2] != 3:
        raise ValueError("clean_rgb must be a 3D (H, W, 3) numpy.ndarray")
    if not isinstance(lesion_mask, np.ndarray) or lesion_mask.shape != clean_rgb.shape[:2]:
        raise ValueError("lesion_mask shape must match clean_rgb spatial dimensions")
    out = clean_rgb.copy()
    out[lesion_mask == 0] = MASKED_NEUTRAL_GRAY_UINT8
    return out


def evaluate_case_variants(
    case_input: CaseEvaluationInput,
    calibration: FISCalibration,
    config: DermaConfig,
) -> CaseEvaluationResult:
    """Evaluate all 6 clinical inference variants for a single (image, condition) pair."""
    if case_input.condition not in ARTIFACT_CONDITIONS:
        raise ValueError(f"Invalid condition '{case_input.condition}'")
    if _is_bool_like(case_input.label) or case_input.label not in BINARY_LABELS:
        raise ValueError(f"label must be binary 0 or 1, got {case_input.label!r}")

    fp_cnn = _validate_unit_float(case_input.p_cnn, "p_cnn")
    fp_masked = _validate_unit_float(case_input.p_masked, "p_masked")
    prep_cond = case_input.prep_cond

    bio_sqrt: BiomarkerResult = extract_abcd_biomarkers(
        clean_rgb=prep_cond.clean_rgb,
        mask=prep_cond.lesion_mask,
        boundary_ring=prep_cond.boundary_ring,
        config=config,
        p_cnn=fp_cnn,
        heatmap=case_input.heatmap,
    )

    p_crisp = float(evaluate_crisp_abcd_baseline([bio_sqrt], calibration)[0])
    risk_sqrt = evaluate_mamdani_fis(bio_sqrt, calibration, config=config).risk_score

    d_ver_hermite = compute_hermite_verified_confidence(fp_cnn, bio_sqrt.alpha)
    risk_hermite = evaluate_mamdani_fis(
        (bio_sqrt.A, bio_sqrt.B_fuzzy, bio_sqrt.C, d_ver_hermite),
        calibration,
        config=config,
    ).risk_score

    # Ablation testing B_grad contribution: substitute pure geometric B_geo for hybrid B_fuzzy
    risk_no_bgrad = evaluate_mamdani_fis(
        (bio_sqrt.A, bio_sqrt.B_geo, bio_sqrt.C, bio_sqrt.D_verified),
        calibration,
        config=config,
    ).risk_score

    mask_iou = (
        1.0
        if case_input.condition == "clean"
        else compute_mask_iou(prep_cond.lesion_mask, case_input.prep_clean.lesion_mask)
    )

    return CaseEvaluationResult(
        image_id=str(case_input.image_id),
        label=int(case_input.label),
        condition=case_input.condition,
        p_cnn=fp_cnn,
        p_masked=fp_masked,
        p_crisp=p_crisp,
        d_ver_sqrt=bio_sqrt.D_verified,
        d_ver_hermite=d_ver_hermite,
        alpha=bio_sqrt.alpha,
        A=bio_sqrt.A,
        B_geo=bio_sqrt.B_geo,
        B_grad=bio_sqrt.B_grad,
        B_fuzzy=bio_sqrt.B_fuzzy,
        C=bio_sqrt.C,
        risk_sqrt=risk_sqrt,
        risk_hermite=risk_hermite,
        risk_no_bgrad=risk_no_bgrad,
        fallback_used=bool(prep_cond.fallback_used),
        mask_iou_vs_clean=mask_iou,
    )


def write_cases_csv(rows: Sequence[CaseEvaluationResult | Mapping[str, Any]], csv_path: str | Path) -> Path:
    """Persist per-case study records to CSV matching CSV_FIELDNAMES."""
    target = Path(csv_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            if isinstance(row, CaseEvaluationResult):
                writer.writerow(row.to_csv_row())
            else:
                writer.writerow({k: row[k] for k in CSV_FIELDNAMES})
    return target


def read_cases_csv(csv_path: str | Path) -> list[dict[str, Any]]:
    """Read and parse artifact_study_cases.csv rows into typed dictionaries."""
    src = Path(csv_path)
    if not src.is_file():
        raise FileNotFoundError(f"Cases CSV not found at {src}")

    records: list[dict[str, Any]] = []
    with src.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or set(CSV_FIELDNAMES) - set(reader.fieldnames):
            missing = set(CSV_FIELDNAMES) - set(reader.fieldnames or [])
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")

        for raw in reader:
            parsed: dict[str, Any] = {
                "image_id": str(raw["image_id"]),
                "label": int(raw["label"]),
                "condition": str(raw["condition"]),
                "fallback_used": raw["fallback_used"].strip().lower() in ("1", "true"),
            }
            for col in FLOAT_CSV_COLUMNS:
                fval = float(raw[col])
                if not math.isfinite(fval):
                    raise ValueError(f"Non-finite value in CSV column '{col}': {raw[col]}")
                parsed[col] = fval
            records.append(parsed)
    return records


# ------------------------------------------------------------------------------
# Colab Full Study Pipeline (Lazy ML Imports, Slice-Wise Low-VRAM TTA)
# ------------------------------------------------------------------------------

def _load_test_split(
    data_dir: Path,
    metadata_csv: Path,
    config: DermaConfig,
    max_images: int | None,
) -> Any:
    """Replicate train_ham10000.py Stage 1 StratifiedGroupKFold 1:1 balanced test split."""
    import pandas as pd
    from sklearn.model_selection import StratifiedGroupKFold

    df = pd.read_csv(metadata_csv)
    mel_code = config.dataset.target_classes["malignant"]
    nv_code = config.dataset.target_classes["benign"]
    seed = config.dataset.random_state

    df_mel = df[df["dx"] == mel_code].copy()
    n_nv_target = min(len(df_mel) * BENIGN_OVERSAMPLE_FACTOR, len(df[df["dx"] == nv_code]))
    df_nv = df[df["dx"] == nv_code].sample(n=n_nv_target, random_state=seed).copy()
    df_all = (
        pd.concat([df_mel, df_nv], ignore_index=True)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    df_all["label"] = (df_all["dx"] == mel_code).astype(int)

    image_index = {p.stem: p for p in data_dir.rglob("*.jpg")}
    df_all["path"] = df_all["image_id"].map(image_index)
    df_all = df_all.dropna(subset=["path"]).reset_index(drop=True)

    groups = df_all["lesion_id"] if "lesion_id" in df_all.columns else df_all["image_id"]
    sgkf_outer = StratifiedGroupKFold(n_splits=OUTER_KFOLD_SPLITS, shuffle=True, random_state=seed)
    _, test_idx = next(sgkf_outer.split(df_all, df_all["label"], groups))
    raw_test_df = df_all.iloc[test_idx].reset_index(drop=True)

    test_mel = raw_test_df[raw_test_df["label"] == 1]
    test_nv = raw_test_df[raw_test_df["label"] == 0].sample(
        n=min(len(test_mel), len(raw_test_df[raw_test_df["label"] == 0])),
        random_state=seed,
    )
    test_df = (
        pd.concat([test_mel, test_nv], ignore_index=True)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    if max_images is not None:
        test_df = test_df.iloc[:max_images].reset_index(drop=True)
    return test_df


def _resize_to_config(rgb: np.ndarray, config: DermaConfig) -> np.ndarray:
    """Resize RGB image deterministically matching preprocess_image interpolation."""
    target_w, target_h = config.dataset.image_size
    h, w = rgb.shape[:2]
    if (w, h) != (target_w, target_h):
        interp = cv2.INTER_AREA if (w > target_w or h > target_h) else cv2.INTER_LINEAR
        return cv2.resize(rgb, (target_w, target_h), interpolation=interp)
    return rgb


def _load_clean_test_records(
    test_df: Any,
    config: DermaConfig,
) -> list[tuple[str, int, np.ndarray, PreprocessingResult]]:
    """Read raw images from disk, resize deterministically, and preprocess clean baselines."""
    base_records: list[tuple[str, int, np.ndarray, PreprocessingResult]] = []
    for row in test_df.itertuples():
        bgr = cv2.imread(str(row.path))
        if bgr is None:
            continue
        rgb = _resize_to_config(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), config)
        prep_clean = preprocess_image(rgb, config)
        base_records.append((str(row.image_id), int(row.label), rgb, prep_clean))
    if not base_records:
        raise ValueError("No readable test images loaded from HAM10000 split")
    return base_records


def _build_tf_evaluators(model: Any) -> tuple[Any, Any]:
    """Build batched @tf.function Pre-Sigmoid Grad-CAM and prob-only forward step callables."""
    import tensorflow as tf

    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[
            model.get_layer("out_relu").output,
            model.get_layer("logits").output,
            model.output,
        ],
    )

    @tf.function
    def _gradcam_batch_step(batch_t: Any) -> tuple[Any, Any]:
        with tf.GradientTape() as tape:
            conv_out, logit_out, prob_out = grad_model(batch_t, training=False)
            target = logit_out[:, 0]
        grads = tape.gradient(target, conv_out)
        pooled = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
        cam_pos = tf.nn.relu(tf.reduce_sum(conv_out * pooled, axis=-1))
        cam_abs = tf.reduce_sum(conv_out * tf.abs(pooled), axis=-1)
        max_pos = tf.reduce_max(cam_pos, axis=(1, 2), keepdims=True)
        cam = tf.where(max_pos > GRADCAM_EPS, cam_pos, cam_abs)
        cam_max = tf.reduce_max(cam, axis=(1, 2), keepdims=True)
        cam = tf.where(cam_max > GRADCAM_EPS, cam / cam_max, tf.zeros_like(cam))
        return prob_out[:, 0], cam

    @tf.function
    def _probs_batch_step(batch_t: Any) -> Any:
        prob_out = model(batch_t, training=False)
        return prob_out[:, 0]

    return _gradcam_batch_step, _probs_batch_step


def _run_batched_gradcam_tta(
    x_arr: np.ndarray,
    gradcam_step: Any,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 4-way TTA Grad-CAM batch-by-batch to bound Colab T4 GPU VRAM."""
    import tensorflow as tf

    probs_chunks, hm_chunks = [], []
    for start in range(0, len(x_arr), batch_size):
        b0 = x_arr[start : start + batch_size]
        p0, h0 = gradcam_step(tf.convert_to_tensor(b0))
        p1, h1 = gradcam_step(tf.convert_to_tensor(np.ascontiguousarray(b0[:, :, ::-1, :])))
        p2, h2 = gradcam_step(tf.convert_to_tensor(np.ascontiguousarray(b0[:, ::-1, :, :])))
        p3, h3 = gradcam_step(tf.convert_to_tensor(np.ascontiguousarray(b0[:, ::-1, ::-1, :])))

        p_tta = TTA_WEIGHT * (p0.numpy() + p1.numpy() + p2.numpy() + p3.numpy())
        h_tta = TTA_WEIGHT * (
            h0.numpy()
            + h1.numpy()[:, :, ::-1]
            + h2.numpy()[:, ::-1, :]
            + h3.numpy()[:, ::-1, ::-1]
        )
        probs_chunks.append(p_tta)
        hm_chunks.append(h_tta)

    probs = np.clip(np.concatenate(probs_chunks, axis=0).astype(np.float64), 0.0, 1.0)
    hms = np.concatenate(hm_chunks, axis=0)
    hm_max = np.max(hms, axis=(1, 2), keepdims=True)
    hms = np.where(hm_max > GRADCAM_EPS, hms / np.maximum(hm_max, GRADCAM_EPS), 0.0)
    return probs, hms


def _run_batched_probs_tta(
    x_arr: np.ndarray,
    probs_step: Any,
    batch_size: int,
) -> np.ndarray:
    """Compute 4-way TTA probability predictions batch-by-batch without gradients."""
    import tensorflow as tf

    probs_chunks = []
    for start in range(0, len(x_arr), batch_size):
        b0 = x_arr[start : start + batch_size]
        p0 = probs_step(tf.convert_to_tensor(b0)).numpy()
        p1 = probs_step(tf.convert_to_tensor(np.ascontiguousarray(b0[:, :, ::-1, :]))).numpy()
        p2 = probs_step(tf.convert_to_tensor(np.ascontiguousarray(b0[:, ::-1, :, :]))).numpy()
        p3 = probs_step(tf.convert_to_tensor(np.ascontiguousarray(b0[:, ::-1, ::-1, :]))).numpy()
        probs_chunks.append(TTA_WEIGHT * (p0 + p1 + p2 + p3))
    return np.clip(np.concatenate(probs_chunks, axis=0).astype(np.float64), 0.0, 1.0)


def _prepare_condition_arrays(
    cond: str,
    base_records: Sequence[tuple[str, int, np.ndarray, PreprocessingResult]],
    seed: int,
    config: DermaConfig,
) -> tuple[list[PreprocessingResult], np.ndarray, np.ndarray]:
    """Apply condition artifact, preprocess via Room 2, and prepare MobileNetV2 batches."""
    import tensorflow as tf

    cond_preps: list[PreprocessingResult] = []
    cnn_raw: list[np.ndarray] = []
    masked_raw: list[np.ndarray] = []

    for img_id, _, rgb_base, prep_clean in base_records:
        if cond == "clean":
            prep_cond = prep_clean
        else:
            rng = _make_deterministic_rng(seed, img_id, cond)
            corrupted = apply_artifact_condition(rgb_base, prep_clean.lesion_mask, cond, rng)
            prep_cond = preprocess_image(corrupted, config)
        cond_preps.append(prep_cond)
        cnn_raw.append(prep_cond.clean_rgb.astype(np.float32))
        masked_raw.append(
            neutralize_off_lesion_pixels(prep_cond.clean_rgb, prep_cond.lesion_mask).astype(np.float32)
        )

    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    x_cnn = preprocess_input(np.asarray(cnn_raw, dtype=np.float32))
    x_masked = preprocess_input(np.asarray(masked_raw, dtype=np.float32))
    return cond_preps, x_cnn, x_masked


def run_perturbation_study(study_config: StudyRunConfig) -> Path:
    """Execute full empirical perturbation study against trained MobileNetV2 & FIS calibration."""
    if _is_bool_like(study_config.batch_size) or study_config.batch_size <= 0:
        raise ValueError(f"batch_size must be positive integer, got {study_config.batch_size}")
    if study_config.max_images is not None and (
        _is_bool_like(study_config.max_images) or study_config.max_images <= 0
    ):
        raise ValueError(f"max_images must be positive integer or None, got {study_config.max_images}")
    if not study_config.data_dir.is_dir():
        raise FileNotFoundError(f"HAM10000 image directory not found: {study_config.data_dir}")
    if not study_config.metadata_csv.is_file():
        raise FileNotFoundError(f"HAM10000 metadata CSV not found: {study_config.metadata_csv}")
    if not study_config.model_path.is_file():
        raise FileNotFoundError(f"Trained Keras model not found: {study_config.model_path}")

    import tensorflow as tf

    config = load_config()
    calibration = load_calibration_json(study_config.calibration_path)
    study_config.out_dir.mkdir(parents=True, exist_ok=True)
    seed = config.dataset.random_state

    test_df = _load_test_split(
        study_config.data_dir,
        study_config.metadata_csv,
        config,
        max_images=study_config.max_images,
    )
    print(f"Loaded Stage 1 balanced test split: N={len(test_df)} images")

    base_records = _load_clean_test_records(test_df, config)
    model = tf.keras.models.load_model(study_config.model_path, compile=False)
    gradcam_step, probs_step = _build_tf_evaluators(model)

    all_rows: list[CaseEvaluationResult] = []
    for cond in ARTIFACT_CONDITIONS:
        print(f"Evaluating condition '{cond}' across N={len(base_records)} test images...")
        cond_preps, x_cnn, x_masked = _prepare_condition_arrays(cond, base_records, seed, config)
        probs_cnn, hms = _run_batched_gradcam_tta(x_cnn, gradcam_step, study_config.batch_size)
        probs_masked = _run_batched_probs_tta(x_masked, probs_step, study_config.batch_size)

        for idx, (img_id, lbl, _, prep_clean) in enumerate(base_records):
            case_in = CaseEvaluationInput(
                image_id=img_id,
                label=lbl,
                condition=cond,
                p_cnn=float(probs_cnn[idx]),
                p_masked=float(probs_masked[idx]),
                heatmap=hms[idx],
                prep_clean=prep_clean,
                prep_cond=cond_preps[idx],
            )
            all_rows.append(evaluate_case_variants(case_in, calibration, config))

        del cond_preps, x_cnn, x_masked, hms
        gc.collect()

    csv_out = study_config.out_dir / CASES_CSV_FILENAME
    write_cases_csv(all_rows, csv_out)
    print(f"Saved {len(all_rows)} case evaluations to {csv_out}")
    return csv_out


# ------------------------------------------------------------------------------
# Self-Test Mode (Zero TensorFlow / Zero External Dataset)
# ------------------------------------------------------------------------------

def _build_selftest_phantom(size: int, radius: int) -> tuple[np.ndarray, np.ndarray]:
    """Create deterministic synthetic RGB skin frame with centered circular lesion mask."""
    rgb = np.full((size, size, 3), SELFTEST_SKIN_RGB, dtype=np.uint8)
    y_grid, x_grid = np.ogrid[:size, :size]
    center = size // 2
    mask = ((x_grid - center) ** 2 + (y_grid - center) ** 2 <= radius ** 2).astype(np.uint8)
    rgb[mask > 0] = SELFTEST_LESION_RGB
    rgb.flags.writeable = False
    mask.flags.writeable = False
    return rgb, mask


def _verify_selftest_overlays(base_rgb: np.ndarray, base_mask: np.ndarray) -> None:
    """Verify overlay determinism, non-mutation, peripheral masks, and full-frame fallbacks."""
    orig_snapshot = base_rgb.copy()
    for cond in ARTIFACT_CONDITIONS:
        out_a = apply_artifact_condition(base_rgb, base_mask, cond, np.random.default_rng(SELFTEST_SEED))
        out_b = apply_artifact_condition(base_rgb, base_mask, cond, np.random.default_rng(SELFTEST_SEED))

        assert np.array_equal(base_rgb, orig_snapshot), "Input RGB was mutated"
        assert out_a.shape == base_rgb.shape and out_a.dtype == np.uint8
        assert np.array_equal(out_a, out_b), f"Non-deterministic overlay for '{cond}'"

        if cond in ("ink", "gel"):
            assert np.array_equal(out_a[base_mask > 0], base_rgb[base_mask > 0])
            assert not np.array_equal(out_a, base_rgb)

    full_mask = np.ones((SELFTEST_IMG_SIZE, SELFTEST_IMG_SIZE), dtype=np.uint8)
    ink_fb = overlay_ink(base_rgb, full_mask, np.random.default_rng(SELFTEST_SEED))
    gel_fb = overlay_gel(base_rgb, full_mask, np.random.default_rng(SELFTEST_SEED))
    assert ink_fb.dtype == np.uint8 and not np.array_equal(ink_fb, base_rgb)
    assert gel_fb.dtype == np.uint8 and not np.array_equal(gel_fb, base_rgb)


def run_selftest() -> None:
    """Verify overlay bounds, mask-awareness, determinism, variants, and CSV round-trip."""
    config = load_config()
    calibration = load_calibration_json(_REPO_ROOT / DEFAULT_CALIBRATION_PATH)
    base_rgb, base_mask = _build_selftest_phantom(SELFTEST_IMG_SIZE, SELFTEST_MASK_RADIUS)
    _verify_selftest_overlays(base_rgb, base_mask)

    heatmap = np.zeros((SELFTEST_HEATMAP_SIZE, SELFTEST_HEATMAP_SIZE), dtype=np.float32)
    heatmap[SELFTEST_HEATMAP_START:SELFTEST_HEATMAP_END, SELFTEST_HEATMAP_START:SELFTEST_HEATMAP_END] = 1.0
    prep_clean = preprocess_image(base_rgb, config)

    rows: list[CaseEvaluationResult] = []
    for idx in range(SELFTEST_CASE_COUNT):
        lbl = idx % 2
        p_cnn = SELFTEST_POS_PCNN if lbl == 1 else SELFTEST_NEG_PCNN
        p_masked = SELFTEST_POS_MASKED if lbl == 1 else SELFTEST_NEG_MASKED
        for cond in ARTIFACT_CONDITIONS:
            rng = _make_deterministic_rng(SELFTEST_SEED, f"SELFTEST_{idx:03d}", cond)
            corrupted = apply_artifact_condition(base_rgb, prep_clean.lesion_mask, cond, rng)
            prep_cond = preprocess_image(corrupted, config) if cond != "clean" else prep_clean
            rows.append(
                evaluate_case_variants(
                    CaseEvaluationInput(
                        image_id=f"SELFTEST_{idx:03d}",
                        label=lbl,
                        condition=cond,
                        p_cnn=p_cnn,
                        p_masked=p_masked,
                        heatmap=heatmap,
                        prep_clean=prep_clean,
                        prep_cond=prep_cond,
                    ),
                    calibration,
                    config,
                )
            )

    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / CASES_CSV_FILENAME
        write_cases_csv(rows, csv_path)
        loaded = read_cases_csv(csv_path)
        assert len(loaded) == SELFTEST_CASE_COUNT * len(ARTIFACT_CONDITIONS)

    print(
        f"SELFTEST PASSED: verified {len(ARTIFACT_CONDITIONS)} conditions, "
        f"mask-awareness, determinism, variants, and CSV round-trip ({len(rows)} rows)."
    )


def main() -> None:
    """CLI entrypoint for artifact perturbation study and offline selftest."""
    parser = argparse.ArgumentParser(
        description="DermaFuzzy-XAI Q1 Artifact Perturbation & Ablation Study"
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="Run local pure-NumPy/OpenCV verification with zero TensorFlow or dataset",
    )
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR, help="Path to HAM10000 images")
    parser.add_argument("--metadata-csv", type=str, default=DEFAULT_METADATA_CSV, help="Path to metadata CSV")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH, help="Path to trained .keras model")
    parser.add_argument(
        "--calibration",
        type=str,
        default=DEFAULT_CALIBRATION_PATH,
        help="Path to fis_calibration.json",
    )
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR, help="Output directory for CSV results")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for Grad-CAM TTA")
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional cap on test split size for debugging",
    )

    args = parser.parse_args()
    if args.selftest:
        run_selftest()
        return

    run_perturbation_study(
        StudyRunConfig(
            data_dir=Path(args.data_dir),
            metadata_csv=Path(args.metadata_csv),
            model_path=Path(args.model_path),
            calibration_path=Path(args.calibration),
            out_dir=Path(args.out_dir),
            batch_size=args.batch_size,
            max_images=args.max_images,
        )
    )


if __name__ == "__main__":
    main()
