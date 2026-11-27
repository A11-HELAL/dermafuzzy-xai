"""Q1 5-Fold Lesion-Grouped Cross-Validation Pipeline (Colab T4 & Offline Selftest).

Replicates the exact `v1.0-baseline` (`e2ac0e9`) training, pre-sigmoid batched
Grad-CAM, 1:1 balanced train-only Mamdani FIS calibration, and 4-tier evaluation
across all 5 outer StratifiedGroupKFold splits grouped strictly by `lesion_id`.

Evaluates both:
- Full Unseen Fold (`raw_test_df`, retaining all benign nevus images in the fold)
- 1:1 Balanced Fold (`bal_test_df`, matched melanoma:nevus ratio)

Supports Colab T4 checkpoint resume (`--resume`) via per-fold OOF CSV files
(`cv_fold_{k}_oof.csv`), guaranteed `try/finally` TensorFlow session clearing,
and offline `--selftest` verification with zero TensorFlow or dataset dependencies.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import cv2
import numpy as np

# Ensure repo packages and sibling colab_trainer modules resolve from any CWD
_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_PACKAGES_DIR: Final[Path] = _REPO_ROOT / "packages"
if str(_PACKAGES_DIR) not in sys.path:
    sys.path.insert(0, str(_PACKAGES_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dermafuzzy_core.biomarkers import BiomarkerResult, extract_abcd_biomarkers
from dermafuzzy_core.config import DermaConfig, load_config
from dermafuzzy_core.evaluation import evaluate_crisp_abcd_baseline
from dermafuzzy_core.fis_engine import (
    FISCalibration,
    calibrate_fis_percentiles,
    evaluate_mamdani_fis,
)
from dermafuzzy_core.preprocessing import PreprocessingResult, preprocess_image
from services.colab_trainer.artifact_perturbation_study import (
    BENIGN_OVERSAMPLE_FACTOR,
    OUTER_KFOLD_SPLITS,
    _build_tf_evaluators,
    _resize_to_config,
    _run_batched_gradcam_tta,
)
from services.colab_trainer.stats_analysis import (
    CLINICAL_TRIAGE_THRESHOLD,
    DEFAULT_THRESHOLD,
    PERCENT_SCALE,
    RISK_NORMALIZATION_DIVISOR,
    compute_pr_auc,
    compute_roc_auc_exact,
    compute_threshold_metrics,
    delong_roc_test,
    mcnemar_exact_test,
)

# ------------------------------------------------------------------------------
# Module Constants (Zero Magic Numbers Policy)
# ------------------------------------------------------------------------------

N_OUTER_FOLDS: Final[int] = 5
INNER_VAL_FOLDS: Final[int] = 5
MIN_FOLDS_FOR_SD: Final[int] = 2
MAX_LEAKAGE_SAMPLE_DISPLAY: Final[int] = 5
RGB_CHANNELS: Final[int] = 3
BINARY_CLASS_WEIGHT_NUMERATOR: Final[float] = 2.0

BINARY_LABELS_SET: Final[frozenset[int]] = frozenset({0, 1})
VALID_TRUE_STRINGS: Final[frozenset[str]] = frozenset({"1", "true"})
VALID_FALSE_STRINGS: Final[frozenset[str]] = frozenset({"0", "false"})
TEXT_BOOL_WORDS: Final[frozenset[str]] = frozenset({"true", "false"})
NAN_TEXT_SENTINELS: Final[frozenset[str]] = frozenset({"", "nan", "none", "null"})

CV_TIERS: Final[tuple[str, ...]] = (
    "tier1_cnn",
    "tier2_crisp",
    "tier3_standard",
    "tier3_triage",
)
EVAL_METRICS: Final[tuple[str, ...]] = (
    "accuracy",
    "sensitivity",
    "specificity",
    "precision",
    "f1_score",
    "roc_auc",
    "pr_auc",
)
COHORT_MODES: Final[tuple[str, ...]] = ("full_unseen", "balanced_1to1")

# Model & Training Hyperparameters (exact v1.0-baseline e2ac0e9 parity)
HEAD_DROPOUT_RATE: Final[float] = 0.35
LOGITS_L2_WEIGHT: Final[float] = 1e-4
STAGE3A_EPOCHS: Final[int] = 5
STAGE3A_LR: Final[float] = 1e-3
UNFREEZE_TOP_LAYERS: Final[int] = 45
STAGE3B_LR: Final[float] = 1e-4
EARLY_STOP_PATIENCE: Final[int] = 6
REDUCE_LR_FACTOR: Final[float] = 0.5
REDUCE_LR_PATIENCE: Final[int] = 2
REDUCE_LR_MIN_LR: Final[float] = 1e-6

AUG_BRIGHTNESS_DELTA: Final[float] = 0.08
AUG_CONTRAST_LOWER: Final[float] = 0.92
AUG_CONTRAST_UPPER: Final[float] = 1.08
AUG_ROT90_MAX: Final[int] = 4
CNN_CLIP_MIN: Final[float] = -1.0
CNN_CLIP_MAX: Final[float] = 1.0

DEFAULT_DATA_DIR: Final[str] = "data/HAM10000_images"
DEFAULT_METADATA_CSV: Final[str] = "data/HAM10000_metadata.csv"
DEFAULT_OUTPUT_DIR: Final[str] = "artifacts/cv_5fold"
DEFAULT_EPOCHS: Final[int] = 20
DEFAULT_BATCH_SIZE: Final[int] = 32

OOF_FIELDNAMES: Final[tuple[str, ...]] = (
    "fold",
    "image_id",
    "lesion_id",
    "label",
    "in_balanced_test",
    "p_cnn",
    "p_crisp",
    "d_ver_sqrt",
    "alpha",
    "A",
    "B_geo",
    "B_grad",
    "B_fuzzy",
    "C",
    "risk_score",
    "fallback_used",
)
FLOAT_OOF_COLS: Final[tuple[str, ...]] = (
    "p_cnn",
    "p_crisp",
    "d_ver_sqrt",
    "alpha",
    "A",
    "B_geo",
    "B_grad",
    "B_fuzzy",
    "C",
    "risk_score",
)

MIN_CALIBRATION_ROWS: Final[int] = 4
UNIT_MIN: Final[float] = 0.0
UNIT_MAX: Final[float] = 1.0

# Selftest Constants
SELFTEST_SEED: Final[int] = 42
SELFTEST_SAMPLES_PER_FOLD: Final[int] = 24
SELFTEST_BALANCED_PER_CLASS: Final[int] = 8


@dataclass(frozen=True, slots=True)
class CVRunConfig:
    """Configuration DTO for 5-fold cross-validation execution."""

    data_dir: Path
    metadata_csv: Path
    output_dir: Path
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    resume: bool = False


@dataclass(frozen=True, slots=True)
class _SplitTensors:
    """Bundled preprocessed train/validation image tensors and labels for one fold."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray


@dataclass(frozen=True, slots=True)
class _FoldEvalContext:
    """Bundled test predictions and biomarkers needed to serialize one fold's OOF rows."""

    fold_idx: int
    balanced_ids: frozenset[str]
    test_preps: Sequence[PreprocessingResult]
    y_test: np.ndarray
    p_test: np.ndarray
    p_crisp_test: np.ndarray
    test_bios: Sequence[BiomarkerResult]
    calibration: FISCalibration


# ------------------------------------------------------------------------------
# Scalar, Split & CSV Validation Guards
# ------------------------------------------------------------------------------

def _is_bool_like(val: Any) -> bool:
    """Return True if val is Python bool or NumPy np.bool_."""
    return isinstance(val, (bool, np.bool_))


def _validate_finite_number(val: Any, name: str) -> float:
    """Validate finite real scalar rejecting bool, np.bool_, strings, NaN, and Inf."""
    if _is_bool_like(val) or not isinstance(val, (int, float, np.integer, np.floating)):
        raise ValueError(f"'{name}' must be a finite numeric scalar, got {type(val).__name__}")
    fval = float(val)
    if not math.isfinite(fval):
        raise ValueError(f"'{name}' must be finite (no NaN or Inf), got {val}")
    return fval


def _validate_strict_int(val: Any, name: str, min_val: int | None = None) -> int:
    """Validate genuine integer rejecting bool, np.bool_, floats, and strings."""
    if _is_bool_like(val) or not isinstance(val, (int, np.integer)):
        raise ValueError(f"'{name}' must be an integer, got {type(val).__name__}")
    ival = int(val)
    if min_val is not None and ival < min_val:
        raise ValueError(f"'{name}' must be >= {min_val}, got {ival}")
    return ival


def _validate_binary_label(val: Any, name: str = "label") -> int:
    """Validate strict binary label in {0, 1}, rejecting bool, np.bool_, floats, NaN."""
    ival = _validate_strict_int(val, name)
    if ival not in BINARY_LABELS_SET:
        raise ValueError(f"'{name}' must be binary 0 or 1, got {ival}")
    return ival


def validate_binary_split_labels(labels: Sequence[Any] | np.ndarray, split_name: str) -> None:
    """Ensure a split is non-empty, contains strictly binary labels, and includes both {0, 1}."""
    if len(labels) == 0:
        raise ValueError(f"Split '{split_name}' cannot be empty")
    parsed = {_validate_binary_label(lbl, f"{split_name}.label") for lbl in labels}
    if parsed != set(BINARY_LABELS_SET):
        raise ValueError(
            f"Split '{split_name}' must contain both binary classes {{0, 1}} (single-class rejected), got {sorted(parsed)}"
        )


def resolve_lesion_groups(
    image_ids: Sequence[Any],
    lesion_ids: Sequence[Any] | None = None,
) -> list[str]:
    """Resolve non-empty lesion group strings, warning explicitly on missing lesion_id fallback."""
    if len(image_ids) == 0:
        raise ValueError("image_ids cannot be empty when resolving lesion groups")

    cleaned_images: list[str] = []
    for idx, img_id in enumerate(image_ids):
        if _is_bool_like(img_id) or img_id is None:
            raise ValueError(f"Invalid image_id at row {idx}: {img_id!r}")
        text = str(img_id).strip()
        if text.lower() in NAN_TEXT_SENTINELS:
            raise ValueError(f"Empty or NaN image_id at row {idx}")
        cleaned_images.append(text)

    if lesion_ids is None:
        warnings.warn(
            "Metadata lacks 'lesion_id' column; falling back to 'image_id' grouping.",
            UserWarning,
            stacklevel=2,
        )
        return cleaned_images

    if len(lesion_ids) != len(cleaned_images):
        raise ValueError(
            f"lesion_ids length ({len(lesion_ids)}) does not match image_ids length ({len(cleaned_images)})"
        )

    resolved: list[str] = []
    backfilled = 0
    for img_text, raw_lesion in zip(cleaned_images, lesion_ids):
        is_nan_float = isinstance(raw_lesion, (float, np.floating)) and not math.isfinite(float(raw_lesion))
        les_text = "" if (raw_lesion is None or is_nan_float) else str(raw_lesion).strip()
        if les_text.lower() in NAN_TEXT_SENTINELS or is_nan_float:
            resolved.append(img_text)
            backfilled += 1
        else:
            resolved.append(les_text)

    if backfilled > 0:
        warnings.warn(
            f"Backfilled {backfilled} missing/NaN lesion_id entries using image_id.",
            UserWarning,
            stacklevel=2,
        )
    return resolved


# ------------------------------------------------------------------------------
# Leakage Guard & Per-Fold CSV Persistence
# ------------------------------------------------------------------------------

def assert_lesion_disjoint(train_lesions: Sequence[Any], test_lesions: Sequence[Any]) -> None:
    """Strictly verify zero lesion_id overlap between training and test splits.

    Raises
    ------
    ValueError
        If either lesion collection is empty.
    RuntimeError
        If any lesion_id appears in both training and test splits.
    """
    if len(train_lesions) == 0 or len(test_lesions) == 0:
        raise ValueError("Train and test lesion collections must both be non-empty")
    train_set = {str(x).strip() for x in train_lesions}
    test_set = {str(x).strip() for x in test_lesions}
    overlap = train_set & test_set
    if overlap:
        sample = sorted(overlap)[:MAX_LEAKAGE_SAMPLE_DISPLAY]
        raise RuntimeError(
            f"DATA LEAKAGE DETECTED: {len(overlap)} overlapping lesion_id(s) across train/test splits: {sample}"
        )


def write_fold_oof_csv(rows: Sequence[Mapping[str, Any]], csv_path: str | Path) -> Path:
    """Persist fold out-of-fold predictions with standardized OOF_FIELDNAMES schema."""
    target = Path(csv_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OOF_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in OOF_FIELDNAMES})
    return target


def _parse_csv_int(raw_val: Any, col: str, line_no: int) -> int:
    """Parse strict integer string cell from CSV row, raising ValueError on malformed text."""
    if raw_val is None or not isinstance(raw_val, str) or not raw_val.strip():
        raise ValueError(f"Line {line_no}: empty or missing integer column '{col}'")
    text = raw_val.strip()
    if text.lower() in TEXT_BOOL_WORDS:
        raise ValueError(f"Line {line_no}: boolean text invalid in integer column '{col}': {text!r}")
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(f"Line {line_no}: invalid integer in column '{col}': {text!r}") from exc


def _parse_csv_float(raw_val: Any, col: str, line_no: int) -> float:
    """Parse finite float cell from CSV row and enforce column numeric range."""
    if raw_val is None or not isinstance(raw_val, str) or not raw_val.strip():
        raise ValueError(f"Line {line_no}: empty or missing float column '{col}'")
    text = raw_val.strip()
    try:
        val = float(text)
    except ValueError as exc:
        raise ValueError(f"Line {line_no}: invalid float in column '{col}': {text!r}") from exc
    if not math.isfinite(val):
        raise ValueError(f"Line {line_no}: non-finite value in OOF column '{col}': {text!r}")
    if col == "risk_score" and not (UNIT_MIN <= val <= RISK_NORMALIZATION_DIVISOR):
        raise ValueError(f"Line {line_no}: risk_score must be in [0, 100], got {val}")
    if col != "risk_score" and not (UNIT_MIN <= val <= UNIT_MAX):
        raise ValueError(f"Line {line_no}: OOF feature '{col}' must be in [0, 1], got {val}")
    return val


def _parse_csv_bool_flag(raw_val: Any, col: str, line_no: int) -> bool:
    """Parse strict boolean flag ('0'/'1'/'true'/'false') from CSV cell."""
    if raw_val is None or not isinstance(raw_val, str):
        raise ValueError(f"Line {line_no}: missing boolean flag column '{col}'")
    text = raw_val.strip().lower()
    if text in VALID_TRUE_STRINGS:
        return True
    if text in VALID_FALSE_STRINGS:
        return False
    raise ValueError(f"Line {line_no}: invalid boolean flag in '{col}': {raw_val!r}")


def _parse_oof_row(raw: Mapping[Any, Any], line_no: int) -> dict[str, Any]:
    """Validate and parse a single DictReader row into a typed OOF record."""
    if None in raw:
        raise ValueError(f"Line {line_no}: ragged CSV row has unexpected extra column(s)")

    img_id = str(raw.get("image_id") or "").strip()
    lesion_id = str(raw.get("lesion_id") or "").strip()
    if not img_id or not lesion_id:
        raise ValueError(f"Line {line_no}: image_id and lesion_id must be non-empty")

    lbl = _parse_csv_int(raw.get("label"), "label", line_no)
    if lbl not in BINARY_LABELS_SET:
        raise ValueError(f"Line {line_no}: non-binary label in OOF CSV: {lbl}")

    in_bal = _parse_csv_int(raw.get("in_balanced_test"), "in_balanced_test", line_no)
    if in_bal not in BINARY_LABELS_SET:
        raise ValueError(f"Line {line_no}: invalid in_balanced_test flag in OOF CSV: {in_bal}")

    rec: dict[str, Any] = {
        "fold": _parse_csv_int(raw.get("fold"), "fold", line_no),
        "image_id": img_id,
        "lesion_id": lesion_id,
        "label": lbl,
        "in_balanced_test": in_bal,
        "fallback_used": _parse_csv_bool_flag(raw.get("fallback_used"), "fallback_used", line_no),
    }
    for col in FLOAT_OOF_COLS:
        rec[col] = _parse_csv_float(raw.get(col), col, line_no)
    return rec


def load_fold_oof_csv(csv_path: str | Path) -> list[dict[str, Any]]:
    """Read and validate fold or pooled OOF CSV rows into typed Python dicts."""
    src = Path(csv_path)
    if not src.is_file():
        raise FileNotFoundError(f"OOF predictions CSV not found: {src}")

    parsed_rows: list[dict[str, Any]] = []
    with src.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or (set(OOF_FIELDNAMES) - set(reader.fieldnames)):
            missing = sorted(set(OOF_FIELDNAMES) - set(reader.fieldnames or []))
            raise ValueError(f"OOF CSV missing required columns: {missing}")

        for line_no, raw in enumerate(reader, start=2):
            parsed_rows.append(_parse_oof_row(raw, line_no))

    if not parsed_rows:
        raise ValueError(f"OOF CSV at '{src}' contains 0 rows")
    return parsed_rows


def load_and_verify_fold_checkpoint(csv_path: str | Path, expected_fold: int) -> list[dict[str, Any]]:
    """Load fold checkpoint CSV during --resume and verify fold index and binary class coverage."""
    fold_k = _validate_strict_int(expected_fold, "expected_fold", min_val=0)
    rows = load_fold_oof_csv(csv_path)
    folds_found = {int(r["fold"]) for r in rows}
    if folds_found != {fold_k}:
        raise ValueError(
            f"Resume checkpoint '{csv_path}' contains mismatched fold index(es): {sorted(folds_found)} (expected [{fold_k}])"
        )
    validate_binary_split_labels([r["label"] for r in rows], f"resume_fold_{fold_k}")
    return rows


# ------------------------------------------------------------------------------
# Cross-Validation Metric & Pooled Statistical Aggregation
# ------------------------------------------------------------------------------

def _extract_tier_arrays(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, dict[str, tuple[np.ndarray, float]]]:
    """Validate OOF rows and extract binary labels + normalized score/cutoff pairs."""
    labels_list: list[int] = []
    cnn_list: list[float] = []
    crisp_list: list[float] = []
    risk_list: list[float] = []

    for idx, r in enumerate(rows):
        lbl = _validate_binary_label(r["label"], f"rows[{idx}].label")
        p_cnn = _validate_finite_number(r["p_cnn"], f"rows[{idx}].p_cnn")
        p_crisp = _validate_finite_number(r["p_crisp"], f"rows[{idx}].p_crisp")
        risk = _validate_finite_number(r["risk_score"], f"rows[{idx}].risk_score")

        if not (UNIT_MIN <= p_cnn <= UNIT_MAX):
            raise ValueError(f"rows[{idx}].p_cnn must be in [0, 1], got {p_cnn}")
        if not (UNIT_MIN <= p_crisp <= UNIT_MAX):
            raise ValueError(f"rows[{idx}].p_crisp must be in [0, 1], got {p_crisp}")
        if not (UNIT_MIN <= risk <= RISK_NORMALIZATION_DIVISOR):
            raise ValueError(f"rows[{idx}].risk_score must be in [0, 100], got {risk}")

        labels_list.append(lbl)
        cnn_list.append(p_cnn)
        crisp_list.append(p_crisp)
        risk_list.append(risk / RISK_NORMALIZATION_DIVISOR)

    y_bin = np.asarray(labels_list, dtype=np.int32)
    unique_lbls = set(np.unique(y_bin))
    if unique_lbls != set(BINARY_LABELS_SET):
        raise ValueError(f"Evaluation subset must contain both classes {{0, 1}}, got {sorted(unique_lbls)}")

    risk_norm = np.asarray(risk_list, dtype=np.float64)
    tiers = {
        "tier1_cnn": (np.asarray(cnn_list, dtype=np.float64), DEFAULT_THRESHOLD),
        "tier2_crisp": (np.asarray(crisp_list, dtype=np.float64), DEFAULT_THRESHOLD),
        "tier3_standard": (risk_norm, DEFAULT_THRESHOLD),
        "tier3_triage": (risk_norm, CLINICAL_TRIAGE_THRESHOLD),
    }
    return y_bin, tiers


def evaluate_rows_4tiers(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Compute Accuracy, Sensitivity, Specificity, Precision, F1, ROC-AUC, PR-AUC for 4 tiers."""
    if not rows:
        raise ValueError("Cannot evaluate empty prediction row collection")
    y_bin, tiers = _extract_tier_arrays(rows)
    out: dict[str, dict[str, Any]] = {}
    for tier_name, (scores, thresh) in tiers.items():
        tm = compute_threshold_metrics(y_bin, scores, threshold=thresh)
        out[tier_name] = {
            **tm,
            "roc_auc": compute_roc_auc_exact(y_bin, scores),
            "pr_auc": compute_pr_auc(y_bin, scores),
        }
    return out


def compute_fold_mean_sd(
    fold_metrics_list: Sequence[Mapping[str, Mapping[str, Any]]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Compute Mean +/- sample SD (ddof=1) across folds rejecting bool, NaN, and Inf."""
    if len(fold_metrics_list) < MIN_FOLDS_FOR_SD:
        raise ValueError(
            f"Mean +/- SD aggregation requires at least {MIN_FOLDS_FOR_SD} folds, got {len(fold_metrics_list)}"
        )

    summary: dict[str, dict[str, dict[str, float]]] = {}
    for tier in CV_TIERS:
        tier_summary: dict[str, dict[str, float]] = {}
        for metric in EVAL_METRICS:
            validated_vals = [
                _validate_finite_number(fm[tier][metric], f"fold[{idx}].{tier}.{metric}")
                for idx, fm in enumerate(fold_metrics_list)
            ]
            vals = np.asarray(validated_vals, dtype=np.float64)
            tier_summary[metric] = {
                "mean": float(np.mean(vals)),
                "sd": float(np.std(vals, ddof=1)),
                "values": [float(v) for v in vals],
            }
        summary[tier] = tier_summary
    return summary


def compute_pooled_significance(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Run DeLong ROC and McNemar exact tests on pooled OOF predictions."""
    if not rows:
        raise ValueError("Cannot compute pooled significance on empty prediction collection")
    y_bin, tiers = _extract_tier_arrays(rows)
    s_t3, t3_cut = tiers["tier3_standard"]
    s_t3_tri, tri_cut = tiers["tier3_triage"]
    s_t2, t2_cut = tiers["tier2_crisp"]
    s_t1, t1_cut = tiers["tier1_cnn"]

    pred_t3_tri = (s_t3_tri >= tri_cut).astype(np.float64)
    pred_t2 = (s_t2 >= t2_cut).astype(np.float64)
    pred_t1 = (s_t1 >= t1_cut).astype(np.float64)

    return {
        "tier3_standard_vs_tier2_crisp": {
            "delong": delong_roc_test(y_bin, s_t3, s_t2),
            "mcnemar": mcnemar_exact_test(y_bin, s_t3, s_t2, threshold=t3_cut),
        },
        "tier3_standard_vs_tier1_cnn": {
            "delong": delong_roc_test(y_bin, s_t3, s_t1),
            "mcnemar": mcnemar_exact_test(y_bin, s_t3, s_t1, threshold=t1_cut),
        },
        "tier3_triage_vs_tier2_crisp": {
            "delong": delong_roc_test(y_bin, s_t3_tri, s_t2),
            "mcnemar": mcnemar_exact_test(y_bin, pred_t3_tri, pred_t2, threshold=DEFAULT_THRESHOLD),
        },
        "tier3_triage_vs_tier1_cnn": {
            "delong": delong_roc_test(y_bin, s_t3_tri, s_t1),
            "mcnemar": mcnemar_exact_test(y_bin, pred_t3_tri, pred_t1, threshold=DEFAULT_THRESHOLD),
        },
    }


def aggregate_cv_results(oof_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate per-fold Mean ± SD and pooled OOF metrics across full and balanced cohorts."""
    if not oof_rows:
        raise ValueError("Cannot aggregate empty OOF predictions")

    row_folds = [_validate_strict_int(r["fold"], f"oof_rows[{idx}].fold", min_val=0) for idx, r in enumerate(oof_rows)]
    in_bal_flags = [
        _validate_binary_label(r["in_balanced_test"], f"oof_rows[{idx}].in_balanced_test")
        for idx, r in enumerate(oof_rows)
    ]

    folds_present = sorted(set(row_folds))
    if len(folds_present) != N_OUTER_FOLDS:
        raise ValueError(f"Expected {N_OUTER_FOLDS} folds in OOF results, found {folds_present}")

    cohort_rows: dict[str, list[tuple[int, Mapping[str, Any]]]] = {
        "full_unseen": list(zip(row_folds, oof_rows)),
        "balanced_1to1": [(f, r) for f, b, r in zip(row_folds, in_bal_flags, oof_rows) if b == 1],
    }

    cohorts_out: dict[str, Any] = {}
    for cohort_name, c_items in cohort_rows.items():
        if not c_items:
            raise ValueError(f"Cohort '{cohort_name}' has zero rows")
        c_rows = [r for _, r in c_items]
        per_fold = [
            evaluate_rows_4tiers([r for f, r in c_items if f == k])
            for k in folds_present
        ]
        cohorts_out[cohort_name] = {
            "n_total": len(c_rows),
            "n_mel": int(sum(_validate_binary_label(r["label"]) == 1 for r in c_rows)),
            "n_nv": int(sum(_validate_binary_label(r["label"]) == 0 for r in c_rows)),
            "per_fold_metrics": {str(k): per_fold[idx] for idx, k in enumerate(folds_present)},
            "mean_sd": compute_fold_mean_sd(per_fold),
            "pooled_oof": evaluate_rows_4tiers(c_rows),
            "pooled_significance": compute_pooled_significance(c_rows),
        }

    return {
        "n_folds": N_OUTER_FOLDS,
        "folds": folds_present,
        "tiers": list(CV_TIERS),
        "cohorts": cohorts_out,
    }


def _fmt_mean_sd(entry: Mapping[str, float], pct: bool = True) -> str:
    """Format Mean ± SD cleanly for Markdown report tables."""
    scale = PERCENT_SCALE if pct else UNIT_MAX
    fmt = ".1f" if pct else ".4f"
    suffix = "%" if pct else ""
    return f"{entry['mean']*scale:{fmt}}{suffix} ± {entry['sd']*scale:{fmt}}{suffix}"


def _format_cohort_markdown(cohort_name: str, cohort_data: Mapping[str, Any]) -> list[str]:
    """Render Mean ± SD, Pooled OOF, and significance sections for one evaluation cohort."""
    lines = [
        f"## Cohort: `{cohort_name}` (N={cohort_data['n_total']}: {cohort_data['n_mel']} mel, {cohort_data['n_nv']} nv)",
        "",
        "### Mean ± SD Across 5 Folds",
        "",
        "| Tier | Accuracy | Sensitivity | Specificity | Precision | F1-Score | ROC-AUC | PR-AUC |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    mean_sd = cohort_data["mean_sd"]
    for tier in CV_TIERS:
        m = mean_sd[tier]
        lines.append(
            f"| `{tier}` | {_fmt_mean_sd(m['accuracy'])} | {_fmt_mean_sd(m['sensitivity'])} | "
            f"{_fmt_mean_sd(m['specificity'])} | {_fmt_mean_sd(m['precision'])} | "
            f"{_fmt_mean_sd(m['f1_score'])} | {_fmt_mean_sd(m['roc_auc'], pct=False)} | "
            f"{_fmt_mean_sd(m['pr_auc'], pct=False)} |"
        )

    lines.extend([
        "",
        "### Pooled Out-of-Fold (OOF) Evaluation & Paired Tests",
        "",
        "| Tier | Accuracy | Sensitivity | Specificity | Precision | F1-Score | ROC-AUC | PR-AUC |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])
    pooled = cohort_data["pooled_oof"]
    for tier in CV_TIERS:
        p = pooled[tier]
        lines.append(
            f"| `{tier}` | {p['accuracy']*PERCENT_SCALE:.1f}% | {p['sensitivity']*PERCENT_SCALE:.1f}% | "
            f"{p['specificity']*PERCENT_SCALE:.1f}% | {p['precision']*PERCENT_SCALE:.1f}% | "
            f"{p['f1_score']*PERCENT_SCALE:.1f}% | {p['roc_auc']:.4f} | {p['pr_auc']:.4f} |"
        )

    lines.extend([
        "",
        "| Comparison | ΔROC-AUC | DeLong z | DeLong p-value | McNemar (b/c) | McNemar p-value |",
        "|---|:---:|:---:|:---:|:---:|:---:|",
    ])
    for pair_key, sig in cohort_data["pooled_significance"].items():
        dl, mc = sig["delong"], sig["mcnemar"]
        lines.append(
            f"| `{pair_key}` | {dl['auc_diff']:+.4f} | {dl['z_stat']:.3f} | {dl['p_value']:.4e} | "
            f"{mc['b_a_correct_b_wrong']}/{mc['c_a_wrong_b_correct']} | {mc['p_value']:.4e} |"
        )
    lines.append("")
    return lines


def format_cv_markdown(summary: Mapping[str, Any]) -> str:
    """Format 5-fold CV summary into a publication-grade Markdown report."""
    lines = [
        "# DermaFuzzy-XAI 5-Fold Lesion-Grouped Cross-Validation Report",
        "",
        f"- **Outer Splits**: `StratifiedGroupKFold(n_splits={summary['n_folds']}, shuffle=True, random_state=42)` grouped strictly by `lesion_id`",
        "- **Evaluated Tiers**: `tier1_cnn` (>=0.50), `tier2_crisp` (>=0.50), `tier3_standard` (>=50.0), `tier3_triage` (>=35.0)",
        "",
    ]
    for cohort_name in COHORT_MODES:
        lines.extend(_format_cohort_markdown(cohort_name, summary["cohorts"][cohort_name]))
    return "\n".join(lines)


def save_cv_outputs(summary: Mapping[str, Any], oof_rows: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
    """Persist pooled OOF CSV, JSON summary, and Markdown report to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    write_fold_oof_csv(oof_rows, output_dir / "cv_oof_predictions.csv")
    (output_dir / "cv_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "cv_report.md").write_text(format_cv_markdown(summary) + "\n", encoding="utf-8")


# ------------------------------------------------------------------------------
# Colab Training & Grad-CAM Execution Helpers (Lazy TF/Sklearn/Pandas Imports)
# ------------------------------------------------------------------------------

def _load_grouped_ham10000(data_dir: Path, metadata_csv: Path, config: DermaConfig) -> Any:
    """Load balanced HAM10000 cohort (all mel + 2x nv) with existing image paths."""
    import pandas as pd

    df = pd.read_csv(metadata_csv)
    required_cols = {"dx", "image_id"}
    if not required_cols.issubset(set(df.columns)):
        missing = sorted(required_cols - set(df.columns))
        raise ValueError(f"Metadata CSV missing required columns: {missing}")

    mel_code = config.dataset.target_classes["malignant"]
    nv_code = config.dataset.target_classes["benign"]
    seed = config.dataset.random_state

    df_mel = df[df["dx"] == mel_code].copy()
    n_nv_target = min(len(df_mel) * BENIGN_OVERSAMPLE_FACTOR, len(df[df["dx"] == nv_code]))
    if len(df_mel) == 0 or n_nv_target == 0:
        raise ValueError(f"Insufficient samples in metadata for classes '{mel_code}' and '{nv_code}'")

    df_nv = df[df["dx"] == nv_code].sample(n=n_nv_target, random_state=seed).copy()
    df_all = (
        pd.concat([df_mel, df_nv], ignore_index=True)
        .sample(frac=UNIT_MAX, random_state=seed)
        .reset_index(drop=True)
    )
    df_all["label"] = (df_all["dx"] == mel_code).astype(int)

    image_index = {p.stem: p for p in data_dir.rglob("*.jpg")}
    df_all["path"] = df_all["image_id"].map(image_index)
    df_all = df_all.dropna(subset=["path"]).reset_index(drop=True)
    if df_all.empty:
        raise ValueError(f"No matching .jpg images found in {data_dir} for selected metadata rows")

    lesion_col = df_all["lesion_id"].tolist() if "lesion_id" in df_all.columns else None
    df_all["lesion_id"] = resolve_lesion_groups(df_all["image_id"].tolist(), lesion_col)
    return df_all


def _preprocess_split_cached(
    split_df: Any,
    prep_cache: dict[str, PreprocessingResult],
    config: DermaConfig,
) -> tuple[list[PreprocessingResult], np.ndarray, np.ndarray]:
    """Preprocess split images once via Room 2 OpenCV pipeline using in-memory uint8 cache."""
    preps, cnn_imgs, labels = [], [], []
    for row in split_df.itertuples():
        img_id = str(row.image_id)
        if img_id not in prep_cache:
            bgr = cv2.imread(str(row.path))
            if bgr is None:
                raise RuntimeError(f"Failed to decode HAM10000 image file: {row.path}")
            rgb = _resize_to_config(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), config)
            prep_cache[img_id] = preprocess_image(rgb, config)
        prep = prep_cache[img_id]
        preps.append(prep)
        cnn_imgs.append(prep.clean_rgb.astype(np.float32))
        labels.append(int(row.label))
    return preps, np.asarray(cnn_imgs, dtype=np.float32), np.asarray(labels, dtype=np.int32)


def _build_mobilenetv2_classifier(config: DermaConfig) -> tuple[Any, Any]:
    """Construct ImageNet-pretrained MobileNetV2 with Clean Linear Logit Head."""
    import tensorflow as tf

    h, w = config.dataset.image_size[1], config.dataset.image_size[0]
    base_model = tf.keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(h, w, RGB_CHANNELS),
    )
    base_model.trainable = False

    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dropout(HEAD_DROPOUT_RATE)(x)
    logits = tf.keras.layers.Dense(
        1,
        activation=None,
        kernel_regularizer=tf.keras.regularizers.l2(LOGITS_L2_WEIGHT),
        name="logits",
    )(x)
    out = tf.keras.layers.Activation("sigmoid", name="probs")(logits)
    return tf.keras.Model(inputs=base_model.input, outputs=out), base_model


def _build_training_dataset(
    x_train_cnn: np.ndarray,
    y_train: np.ndarray,
    batch_size: int,
    seed: int,
) -> Any:
    """Build GPU-prefetched augmented tf.data pipeline matching v1.0-baseline."""
    import tensorflow as tf

    def _augment(imgs: Any, lbls: Any) -> tuple[Any, Any]:
        imgs = tf.image.random_flip_left_right(imgs)
        imgs = tf.image.random_flip_up_down(imgs)
        k = tf.random.uniform([], minval=0, maxval=AUG_ROT90_MAX, dtype=tf.int32)
        imgs = tf.image.rot90(imgs, k=k)
        imgs = tf.image.random_brightness(imgs, max_delta=AUG_BRIGHTNESS_DELTA)
        imgs = tf.image.random_contrast(imgs, lower=AUG_CONTRAST_LOWER, upper=AUG_CONTRAST_UPPER)
        imgs = tf.clip_by_value(imgs, CNN_CLIP_MIN, CNN_CLIP_MAX)
        return imgs, lbls

    return (
        tf.data.Dataset.from_tensor_slices((x_train_cnn, y_train))
        .shuffle(len(x_train_cnn), seed=seed)
        .batch(batch_size)
        .map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )


def _train_fold_mobilenetv2(
    split_tensors: _SplitTensors,
    config: DermaConfig,
    epochs: int,
    batch_size: int,
) -> Any:
    """Two-stage fine-tune MobileNetV2 (Stage 3a warmup + Stage 3b frozen-BN unfreeze)."""
    import tensorflow as tf

    model, base_model = _build_mobilenetv2_classifier(config)
    y_train = split_tensors.y_train
    n_neg, n_pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    class_weight = {
        0: len(y_train) / (BINARY_CLASS_WEIGHT_NUMERATOR * max(n_neg, 1)),
        1: len(y_train) / (BINARY_CLASS_WEIGHT_NUMERATOR * max(n_pos, 1)),
    }
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    aug_ds = _build_training_dataset(split_tensors.x_train, y_train, batch_size, config.dataset.random_state)
    val_data = (split_tensors.x_val, split_tensors.y_val)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=STAGE3A_LR),
        loss=loss_fn,
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    model.fit(aug_ds, validation_data=val_data, epochs=STAGE3A_EPOCHS, class_weight=class_weight, verbose=1)

    # Keep all BatchNormalization layers frozen during top-layer fine-tuning to stabilize T4 stats.
    base_model.trainable = True
    for layer in base_model.layers[:-UNFREEZE_TOP_LAYERS]:
        layer.trainable = False
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=STAGE3B_LR),
        loss=loss_fn,
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_auc", mode="max", patience=EARLY_STOP_PATIENCE, restore_best_weights=True
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=REDUCE_LR_FACTOR, patience=REDUCE_LR_PATIENCE, min_lr=REDUCE_LR_MIN_LR
    )
    model.fit(
        aug_ds,
        validation_data=val_data,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=[early_stop, reduce_lr],
        verbose=1,
    )
    return model


def _run_batched_gradcam_single(
    x_arr: np.ndarray,
    gradcam_step: Any,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run batched Pre-Sigmoid Grad-CAM without TTA (for train split calibration)."""
    import tensorflow as tf

    if len(x_arr) == 0:
        raise ValueError("Input batch x_arr cannot be empty for Grad-CAM inference")
    valid_bs = _validate_strict_int(batch_size, "batch_size", min_val=1)

    probs_chunks, hm_chunks = [], []
    for start in range(0, len(x_arr), valid_bs):
        bt = tf.convert_to_tensor(x_arr[start : start + valid_bs])
        pb, hb = gradcam_step(bt)
        probs_chunks.append(pb.numpy())
        hm_chunks.append(hb.numpy())
    probs = np.clip(np.concatenate(probs_chunks, axis=0).astype(np.float64), UNIT_MIN, UNIT_MAX)
    hms = np.concatenate(hm_chunks, axis=0)
    return probs, hms


def _compute_split_biomarkers(
    preps: Sequence[PreprocessingResult],
    probs: np.ndarray,
    hms: np.ndarray,
    config: DermaConfig,
) -> list[BiomarkerResult]:
    """Compute Room 3 ABCD biomarkers + Concave Sqrt saliency-verified D_verified."""
    return [
        extract_abcd_biomarkers(
            clean_rgb=prep.clean_rgb,
            mask=prep.lesion_mask,
            boundary_ring=prep.boundary_ring,
            config=config,
            p_cnn=float(probs[idx]),
            heatmap=hms[idx],
        )
        for idx, prep in enumerate(preps)
    ]


def _calibrate_fold_fis(
    train_bios: Sequence[BiomarkerResult],
    y_train: np.ndarray,
    fold_idx: int,
    config: DermaConfig,
) -> FISCalibration:
    """Strictly calibrate Mamdani FIS percentiles on the 1:1 balanced train subset only."""
    pos_idx = [i for i, lbl in enumerate(y_train) if int(lbl) == 1]
    neg_idx = [i for i, lbl in enumerate(y_train) if int(lbl) == 0][: len(pos_idx)]
    bal_train_idx = pos_idx + neg_idx
    if len(bal_train_idx) < MIN_CALIBRATION_ROWS or not pos_idx or not neg_idx:
        raise RuntimeError(
            f"Fold {fold_idx} has insufficient balanced training samples (pos={len(pos_idx)}, neg={len(neg_idx)})"
        )
    train_matrix = np.asarray([train_bios[i].as_vector() for i in bal_train_idx], dtype=np.float64)
    return calibrate_fis_percentiles(train_matrix, config)


def _build_fold_oof_rows(
    raw_test_df: Any,
    ctx: _FoldEvalContext,
    config: DermaConfig,
) -> list[dict[str, Any]]:
    """Serialize all 4 tiers and biomarkers for one fold into OOF dict records."""
    fold_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(raw_test_df.itertuples()):
        bio = ctx.test_bios[idx]
        fis_pred = evaluate_mamdani_fis(bio, ctx.calibration, config=config)
        img_id = str(row.image_id)
        fold_rows.append({
            "fold": ctx.fold_idx,
            "image_id": img_id,
            "lesion_id": str(row.lesion_id),
            "label": int(ctx.y_test[idx]),
            "in_balanced_test": 1 if img_id in ctx.balanced_ids else 0,
            "p_cnn": f"{float(ctx.p_test[idx]):.8f}",
            "p_crisp": f"{float(ctx.p_crisp_test[idx]):.8f}",
            "d_ver_sqrt": f"{bio.D_verified:.8f}",
            "alpha": f"{bio.alpha:.8f}",
            "A": f"{bio.A:.8f}",
            "B_geo": f"{bio.B_geo:.8f}",
            "B_grad": f"{bio.B_grad:.8f}",
            "B_fuzzy": f"{bio.B_fuzzy:.8f}",
            "C": f"{bio.C:.8f}",
            "risk_score": f"{fis_pred.risk_score:.8f}",
            "fallback_used": int(ctx.test_preps[idx].fallback_used),
        })
    return fold_rows


def _execute_single_fold(
    fold_idx: int,
    splits: tuple[Any, Any],
    df_all: Any,
    prep_cache: dict[str, PreprocessingResult],
    cv_cfg: CVRunConfig,
    config: DermaConfig,
) -> list[dict[str, Any]]:
    """Train, calibrate strictly on 1:1 balanced train split, and evaluate fold_idx."""
    import tensorflow as tf
    from sklearn.model_selection import StratifiedGroupKFold

    trainval_idx, test_idx = splits
    trainval_df = df_all.iloc[trainval_idx].reset_index(drop=True)
    raw_test_df = df_all.iloc[test_idx].reset_index(drop=True)
    assert_lesion_disjoint(trainval_df["lesion_id"].tolist(), raw_test_df["lesion_id"].tolist())
    validate_binary_split_labels(raw_test_df["label"].tolist(), f"fold_{fold_idx}_raw_test")

    seed = config.dataset.random_state
    sgkf_inner = StratifiedGroupKFold(n_splits=INNER_VAL_FOLDS, shuffle=True, random_state=seed)
    tr_idx, va_idx = next(sgkf_inner.split(trainval_df, trainval_df["label"], trainval_df["lesion_id"]))
    train_df = trainval_df.iloc[tr_idx].reset_index(drop=True)
    val_df = trainval_df.iloc[va_idx].reset_index(drop=True)
    assert_lesion_disjoint(train_df["lesion_id"].tolist(), val_df["lesion_id"].tolist())
    validate_binary_split_labels(train_df["label"].tolist(), f"fold_{fold_idx}_train")
    validate_binary_split_labels(val_df["label"].tolist(), f"fold_{fold_idx}_val")

    test_mel = raw_test_df[raw_test_df["label"] == 1]
    test_nv = raw_test_df[raw_test_df["label"] == 0].sample(
        n=min(len(test_mel), len(raw_test_df[raw_test_df["label"] == 0])),
        random_state=seed,
    )
    balanced_ids = frozenset(test_mel["image_id"].astype(str)).union(frozenset(test_nv["image_id"].astype(str)))

    train_preps, x_train_raw, y_train = _preprocess_split_cached(train_df, prep_cache, config)
    _, x_val_raw, y_val = _preprocess_split_cached(val_df, prep_cache, config)
    test_preps, x_test_raw, y_test = _preprocess_split_cached(raw_test_df, prep_cache, config)

    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    model: Any = None
    gradcam_step: Any = None
    x_train_cnn: np.ndarray | None = None
    x_val_cnn: np.ndarray | None = None
    x_test_cnn: np.ndarray | None = None
    hm_train: np.ndarray | None = None
    hm_test: np.ndarray | None = None

    try:
        x_train_cnn = preprocess_input(x_train_raw)
        x_val_cnn = preprocess_input(x_val_raw)
        x_test_cnn = preprocess_input(x_test_raw)

        model = _train_fold_mobilenetv2(
            _SplitTensors(x_train=x_train_cnn, y_train=y_train, x_val=x_val_cnn, y_val=y_val),
            config,
            cv_cfg.epochs,
            cv_cfg.batch_size,
        )
        gradcam_step, _ = _build_tf_evaluators(model)
        p_train, hm_train = _run_batched_gradcam_single(x_train_cnn, gradcam_step, cv_cfg.batch_size)
        p_test, hm_test = _run_batched_gradcam_tta(x_test_cnn, gradcam_step, cv_cfg.batch_size)

        train_bios = _compute_split_biomarkers(train_preps, p_train, hm_train, config)
        test_bios = _compute_split_biomarkers(test_preps, p_test, hm_test, config)
        calibration = _calibrate_fold_fis(train_bios, y_train, fold_idx, config)
        p_crisp_test = evaluate_crisp_abcd_baseline(test_bios, calibration)

        return _build_fold_oof_rows(
            raw_test_df,
            _FoldEvalContext(
                fold_idx=fold_idx,
                balanced_ids=balanced_ids,
                test_preps=test_preps,
                y_test=y_test,
                p_test=p_test,
                p_crisp_test=p_crisp_test,
                test_bios=test_bios,
                calibration=calibration,
            ),
            config,
        )
    finally:
        # Always release Keras graph and GPU/CPU buffers even if a fold errors mid-run.
        del model, gradcam_step, x_train_cnn, x_val_cnn, x_test_cnn, hm_train, hm_test
        tf.keras.backend.clear_session()
        gc.collect()


def run_5fold_cv(cv_cfg: CVRunConfig) -> Path:
    """Execute complete 5-fold lesion-grouped cross-validation and export aggregated reports."""
    _validate_strict_int(cv_cfg.epochs, "epochs", min_val=1)
    _validate_strict_int(cv_cfg.batch_size, "batch_size", min_val=1)
    if not cv_cfg.data_dir.is_dir():
        raise FileNotFoundError(f"HAM10000 image directory not found: {cv_cfg.data_dir}")
    if not cv_cfg.metadata_csv.is_file():
        raise FileNotFoundError(f"HAM10000 metadata CSV not found: {cv_cfg.metadata_csv}")

    from sklearn.model_selection import StratifiedGroupKFold

    config = load_config()
    cv_cfg.output_dir.mkdir(parents=True, exist_ok=True)
    seed = config.dataset.random_state

    df_all = _load_grouped_ham10000(cv_cfg.data_dir, cv_cfg.metadata_csv, config)
    sgkf_outer = StratifiedGroupKFold(n_splits=OUTER_KFOLD_SPLITS, shuffle=True, random_state=seed)
    prep_cache: dict[str, PreprocessingResult] = {}

    all_oof_parsed: list[dict[str, Any]] = []
    for fold_idx, splits in enumerate(sgkf_outer.split(df_all, df_all["label"], df_all["lesion_id"])):
        fold_csv = cv_cfg.output_dir / f"cv_fold_{fold_idx}_oof.csv"
        if cv_cfg.resume and fold_csv.is_file():
            print(f"[FOLD {fold_idx + 1}/{OUTER_KFOLD_SPLITS}] Resuming from verified checkpoint: {fold_csv}")
            all_oof_parsed.extend(load_and_verify_fold_checkpoint(fold_csv, fold_idx))
            continue

        print(f"\n[FOLD {fold_idx + 1}/{OUTER_KFOLD_SPLITS}] Training MobileNetV2 & Calibrating Fold FIS...")
        fold_raw_rows = _execute_single_fold(fold_idx, splits, df_all, prep_cache, cv_cfg, config)
        write_fold_oof_csv(fold_raw_rows, fold_csv)
        all_oof_parsed.extend(load_and_verify_fold_checkpoint(fold_csv, fold_idx))

    summary = aggregate_cv_results(all_oof_parsed)
    save_cv_outputs(summary, all_oof_parsed, cv_cfg.output_dir)
    print(f"5-Fold CV complete. Reports saved in {cv_cfg.output_dir}")
    return cv_cfg.output_dir / "cv_summary.json"


# ------------------------------------------------------------------------------
# Offline Self-Test Mode (Zero TensorFlow / Zero External Dataset)
# ------------------------------------------------------------------------------

def _build_selftest_oof_rows(rng: np.random.Generator) -> list[dict[str, Any]]:
    """Build deterministic 5-fold synthetic OOF rows for offline CV verification."""
    rows: list[dict[str, Any]] = []
    for k in range(N_OUTER_FOLDS):
        n_mel = SELFTEST_BALANCED_PER_CLASS
        n_nv = BENIGN_OVERSAMPLE_FACTOR * SELFTEST_BALANCED_PER_CLASS
        for idx in range(n_mel + n_nv):
            lbl = 1 if idx < n_mel else 0
            in_bal = 1 if (lbl == 1 or idx < n_mel + SELFTEST_BALANCED_PER_CLASS) else 0
            p_cnn = float(rng.uniform(0.60, 0.92) if lbl == 1 else rng.uniform(0.10, 0.42))
            p_crisp = float(rng.choice([0.50, 0.75, 1.0]) if lbl == 1 else rng.choice([0.0, 0.25, 0.50]))
            risk_sc = float(rng.uniform(52.0, 90.0) if lbl == 1 else rng.uniform(12.0, 44.0))
            rows.append({
                "fold": k,
                "image_id": f"ISIC_F{k}_{idx:03d}",
                "lesion_id": f"HAM_L{k}_{idx:03d}",
                "label": lbl,
                "in_balanced_test": in_bal,
                "p_cnn": p_cnn,
                "p_crisp": p_crisp,
                "d_ver_sqrt": p_cnn * 0.85,
                "alpha": 0.75 if lbl == 1 else 0.18,
                "A": 0.65 if lbl == 1 else 0.22,
                "B_geo": 0.68 if lbl == 1 else 0.25,
                "B_grad": 0.70 if lbl == 1 else 0.26,
                "B_fuzzy": 0.69 if lbl == 1 else 0.255,
                "C": 0.64 if lbl == 1 else 0.20,
                "risk_score": risk_sc,
                "fallback_used": 0,
            })
    return rows


def run_selftest() -> None:
    """Verify lesion-ID leakage guard, 5-fold Mean ± SD, pooled OOF stats, resume skip, and exports."""
    assert_lesion_disjoint(["L1", "L2"], ["L3", "L4"])
    try:
        assert_lesion_disjoint(["L1", "L2"], ["L2", "L3"])
        raise AssertionError("Expected RuntimeError on overlapping lesion_id")
    except RuntimeError:
        pass

    rng = np.random.default_rng(SELFTEST_SEED)
    syn_rows = _build_selftest_oof_rows(rng)

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        resumed_rows: list[dict[str, Any]] = []
        for k in range(N_OUTER_FOLDS):
            fold_rows = [r for r in syn_rows if r["fold"] == k]
            fold_csv = write_fold_oof_csv(fold_rows, out_dir / f"cv_fold_{k}_oof.csv")
            resumed_rows.extend(load_and_verify_fold_checkpoint(fold_csv, k))

        summary = aggregate_cv_results(resumed_rows)
        save_cv_outputs(summary, resumed_rows, out_dir)

        assert (out_dir / "cv_summary.json").is_file()
        assert (out_dir / "cv_report.md").is_file()
        assert (out_dir / "cv_oof_predictions.csv").is_file()
        parsed_json = json.loads((out_dir / "cv_summary.json").read_text(encoding="utf-8"))
        assert parsed_json["cohorts"]["full_unseen"]["n_total"] == N_OUTER_FOLDS * SELFTEST_SAMPLES_PER_FOLD
        assert parsed_json["cohorts"]["balanced_1to1"]["n_total"] == N_OUTER_FOLDS * 2 * SELFTEST_BALANCED_PER_CLASS

    print(
        f"SELFTEST PASSED: verified lesion_id disjointness guard, {N_OUTER_FOLDS}-fold Mean+/-SD, "
        "pooled OOF DeLong/McNemar stats, resume checkpointing, and JSON/MD exports."
    )


def main() -> None:
    """CLI entrypoint for 5-fold cross-validation pipeline and offline selftest."""
    parser = argparse.ArgumentParser(
        description="DermaFuzzy-XAI 5-Fold Lesion-Grouped Cross-Validation Pipeline"
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="Run offline verification with zero TensorFlow or dataset",
    )
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR, help="Path to HAM10000 images")
    parser.add_argument("--metadata-csv", type=str, default=DEFAULT_METADATA_CSV, help="Path to metadata CSV")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory for CV artifacts")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Stage 3b fine-tuning max epochs")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for training & TTA")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip completed folds when cv_fold_{k}_oof.csv already exists",
    )

    args = parser.parse_args()
    if args.selftest:
        run_selftest()
        return

    run_5fold_cv(
        CVRunConfig(
            data_dir=Path(args.data_dir),
            metadata_csv=Path(args.metadata_csv),
            output_dir=Path(args.output_dir),
            epochs=args.epochs,
            batch_size=args.batch_size,
            resume=bool(args.resume),
        )
    )


if __name__ == "__main__":
    main()
