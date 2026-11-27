"""Q1 Pure NumPy + SciPy Statistical Analysis Engine for DermaFuzzy-XAI.

Consumes `artifact_study_cases.csv` produced by `artifact_perturbation_study.py`
across conditions ("clean", "ruler", "ink", "gel") and 6 model tiers:
    ("p_cnn", "p_masked", "p_crisp", "risk_sqrt", "risk_hermite", "risk_no_bgrad")

CRITICAL SCALE INVARIANT:
    `risk_*` columns (`risk_sqrt`, `risk_hermite`, `risk_no_bgrad`) are reported
    on [0.0, 100.0] and MUST be normalized by `/ 100.0` into [0.0, 1.0] prior to
    metric computation, using `threshold = 0.50` across all 6 normalized tiers.

Outputs:
    - `stats_summary.json`: Machine-readable metrics, CIs, DeLong/McNemar p-values,
      operating points, Bayes PPV/NPV tables, and artifact resilience deltas.
    - `stats_report.md`: Publication-grade formatted Markdown statistical report.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Final, Mapping

import numpy as np
from scipy import stats as scipy_stats

# ------------------------------------------------------------------------------
# Module Constants (Zero Magic Numbers Policy)
# ------------------------------------------------------------------------------

CONDITIONS: Final[tuple[str, ...]] = ("clean", "ruler", "ink", "gel")
PERTURBED_CONDITIONS: Final[tuple[str, ...]] = ("ruler", "ink", "gel")
TIER_COLUMNS: Final[tuple[str, ...]] = (
    "p_cnn",
    "p_masked",
    "p_crisp",
    "risk_sqrt",
    "risk_hermite",
    "risk_no_bgrad",
)
RISK_SCALE_TIERS: Final[frozenset[str]] = frozenset(
    {"risk_sqrt", "risk_hermite", "risk_no_bgrad"}
)

RISK_NORMALIZATION_DIVISOR: Final[float] = 100.0
PERCENT_SCALE: Final[float] = 100.0
DEFAULT_THRESHOLD: Final[float] = 0.50
CLINICAL_TRIAGE_THRESHOLD: Final[float] = 0.35
TARGET_HIGH_SENSITIVITY: Final[float] = 0.90
BINARY_SPLIT_MIDPOINT: Final[float] = 0.5
MIN_YOUDEN_SENTINEL: Final[float] = -2.0
SENSITIVITY_TARGET_TOL: Final[float] = 1e-12

DEFAULT_N_BOOT: Final[int] = 2000
DEFAULT_SEED: Final[int] = 42
CI_LOWER_PERCENTILE: Final[float] = 2.5
CI_UPPER_PERCENTILE: Final[float] = 97.5

AUC_TIE_FALLBACK: Final[float] = 0.50
ZERO_VARIANCE_EPS: Final[float] = 1e-15
DELONG_MIN_VARIANCE_FLOOR: Final[float] = 1e-12
MIN_DELONG_CLASS_SAMPLES: Final[int] = 2
EXPECTED_BINARY_CLASSES: Final[frozenset[float]] = frozenset({0.0, 1.0})

PAIRED_COMPARISONS: Final[tuple[tuple[str, str], ...]] = (
    ("risk_sqrt", "p_cnn"),
    ("risk_sqrt", "p_crisp"),
    ("risk_hermite", "p_cnn"),
    ("risk_hermite", "p_crisp"),
    ("risk_hermite", "risk_sqrt"),
    ("risk_sqrt", "risk_no_bgrad"),
)

PREVALENCE_RATIOS: Final[tuple[tuple[str, float], ...]] = (
    ("2:1", 2.0),
    ("6:1", 6.0),
    ("10:1", 10.0),
)

DEFAULT_CASES_CSV: Final[str] = "artifacts/q1_study/artifact_study_cases.csv"
DEFAULT_OUT_DIR: Final[str] = "artifacts/q1_study"
SUMMARY_JSON_FILENAME: Final[str] = "stats_summary.json"
REPORT_MD_FILENAME: Final[str] = "stats_report.md"


# ------------------------------------------------------------------------------
# Validation & Normalization Helpers
# ------------------------------------------------------------------------------

def _is_bool_like(val: Any) -> bool:
    """Return True if value is a Python bool or NumPy np.bool_."""
    return isinstance(val, (bool, np.bool_))


def _validate_int_param(val: Any, name: str, min_val: int = 0) -> int:
    """Validate genuine integer parameter rejecting bool/np.bool_ and floats."""
    if _is_bool_like(val) or not isinstance(val, (int, np.integer)):
        raise ValueError(f"'{name}' must be an integer, got {type(val).__name__}")
    ival = int(val)
    if ival < min_val:
        raise ValueError(f"'{name}' must be >= {min_val}, got {ival}")
    return ival


def _validate_numeric_1d(arr: Any, name: str) -> np.ndarray:
    """Validate 1D finite numeric array rejecting bools, NaNs, Infs, and empty inputs."""
    if _is_bool_like(arr):
        raise ValueError(f"Boolean input is invalid for '{name}'")
    if isinstance(arr, (list, tuple)):
        for item in arr:
            if _is_bool_like(item):
                raise ValueError(f"Boolean element found in '{name}'")

    np_arr = np.asarray(arr)
    if np_arr.dtype == bool or np_arr.dtype == np.bool_:
        raise ValueError(f"Boolean array is invalid for '{name}'")
    if np_arr.ndim != 1:
        raise ValueError(f"'{name}' must be 1D, got shape {np_arr.shape}")
    if np_arr.size == 0:
        raise ValueError(f"'{name}' cannot be empty")
    if not np.issubdtype(np_arr.dtype, np.number):
        raise ValueError(f"'{name}' must contain numeric values, got dtype {np_arr.dtype}")

    f_arr = np_arr.astype(np.float64)
    if not np.all(np.isfinite(f_arr)):
        raise ValueError(f"'{name}' contains NaN or Inf values")
    return f_arr


def _validate_binary_and_scores(
    y_true: Any,
    y_score: Any,
    score_name: str = "y_score",
    require_both_classes: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate matching binary ground truth labels in {0, 1} and finite scores."""
    yt = _validate_numeric_1d(y_true, "y_true")
    ys = _validate_numeric_1d(y_score, score_name)
    if len(yt) != len(ys):
        raise ValueError(f"Length mismatch: y_true={len(yt)}, {score_name}={len(ys)}")

    unique_yt = set(np.unique(yt))
    if not unique_yt.issubset(EXPECTED_BINARY_CLASSES):
        raise ValueError(f"y_true must contain only binary labels {{0, 1}}, got {sorted(unique_yt)}")
    if require_both_classes and unique_yt != set(EXPECTED_BINARY_CLASSES):
        raise ValueError(
            f"y_true must contain both binary classes {{0, 1}} (single-class rejected), got {sorted(unique_yt)}"
        )

    return (yt >= BINARY_SPLIT_MIDPOINT).astype(np.int32), ys


def _validate_unit_float(val: Any, name: str) -> float:
    """Validate scalar float in [0.0, 1.0], rejecting bool, NaN, Inf."""
    if _is_bool_like(val) or not isinstance(val, (int, float, np.floating, np.integer)):
        raise ValueError(f"'{name}' must be float in [0.0, 1.0], got {type(val).__name__}")
    fval = float(val)
    if not math.isfinite(fval) or not (0.0 <= fval <= 1.0):
        raise ValueError(f"'{name}' must be finite and in [0.0, 1.0], got {val}")
    return fval


def _validate_unit_score_array(scores: np.ndarray, name: str) -> np.ndarray:
    """Ensure every probability/normalized risk element lies within [0.0, 1.0]."""
    if np.min(scores) < 0.0 or np.max(scores) > 1.0:
        raise ValueError(
            f"'{name}' must contain normalized values in [0.0, 1.0], got [{np.min(scores)}, {np.max(scores)}]"
        )
    return scores


def normalize_tier_scores(values: Any, tier_name: str) -> np.ndarray:
    """Normalize a raw score column into [0.0, 1.0], scaling risk_* columns by / 100.0."""
    arr = _validate_numeric_1d(values, tier_name)
    if tier_name in RISK_SCALE_TIERS:
        if np.min(arr) < 0.0 or np.max(arr) > RISK_NORMALIZATION_DIVISOR:
            raise ValueError(
                f"Risk column '{tier_name}' must be in [0.0, 100.0], got [{np.min(arr)}, {np.max(arr)}]"
            )
        arr = arr / RISK_NORMALIZATION_DIVISOR
    else:
        _validate_unit_score_array(arr, tier_name)
    return np.clip(arr, 0.0, 1.0)


# ------------------------------------------------------------------------------
# Core Biostatistical & Ranking Metrics
# ------------------------------------------------------------------------------

def _roc_auc_binary(y_bin: np.ndarray, scores: np.ndarray) -> float:
    """Fast internal tie-aware ROC-AUC helper on pre-validated arrays."""
    pos_mask = y_bin == 1
    n_pos = int(np.sum(pos_mask))
    n_neg = len(y_bin) - n_pos

    if n_pos == 0 or n_neg == 0 or np.all(scores == scores[0]):
        return AUC_TIE_FALLBACK

    ranks = scipy_stats.rankdata(scores, method="average")
    auc = (float(np.sum(ranks[pos_mask])) - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)
    return float(np.clip(auc, 0.0, 1.0))


def _pr_auc_binary(y_bin: np.ndarray, scores: np.ndarray) -> float:
    """Fast internal step-function PR-AUC helper on pre-validated two-class arrays."""
    n_pos = int(np.sum(y_bin == 1))
    order = np.argsort(-scores, kind="mergesort")
    y_sorted = y_bin[order]
    s_sorted = scores[order]

    # Group ties at distinct descending score cutoffs to match sklearn average_precision_score
    distinct_idx = np.flatnonzero(np.r_[np.diff(s_sorted) != 0.0, True])
    tps = np.cumsum(y_sorted, dtype=np.float64)[distinct_idx]
    fps = (distinct_idx.astype(np.float64) + 1.0) - tps
    precisions = tps / (tps + fps)
    recalls = tps / float(n_pos)
    prev_recalls = np.r_[0.0, recalls[:-1]]

    return float(np.clip(float(np.sum((recalls - prev_recalls) * precisions)), 0.0, 1.0))


def compute_roc_auc_exact(y_true: Any, y_score: Any) -> float:
    """Compute exact tie-aware ROC-AUC via fractional average midranks (Mann-Whitney U)."""
    y_bin, scores = _validate_binary_and_scores(y_true, y_score, require_both_classes=True)
    return _roc_auc_binary(y_bin, scores)


def compute_pr_auc(y_true: Any, y_score: Any) -> float:
    """Compute exact step-function Precision-Recall AUC (Average Precision) with tie grouping."""
    y_bin, scores = _validate_binary_and_scores(y_true, y_score, require_both_classes=True)
    return _pr_auc_binary(y_bin, scores)


def compute_threshold_metrics(
    y_bin: np.ndarray,
    scores: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    """Compute binary confusion counts and classification rates at a scalar cutoff."""
    thresh = _validate_unit_float(threshold, "threshold")
    y_pred = scores >= thresh
    pos_actual = y_bin == 1
    neg_actual = y_bin == 0

    tp = int(np.sum(pos_actual & y_pred))
    tn = int(np.sum(neg_actual & (~y_pred)))
    fp = int(np.sum(neg_actual & y_pred))
    fn = int(np.sum(pos_actual & (~y_pred)))

    n_pos = tp + fn
    n_neg = tn + fp
    total = n_pos + n_neg

    sens = float(tp / n_pos) if n_pos > 0 else 0.0
    spec = float(tn / n_neg) if n_neg > 0 else 0.0
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    acc = float((tp + tn) / total) if total > 0 else 0.0
    f1 = float(2.0 * prec * sens / (prec + sens)) if (prec + sens) > 0.0 else 0.0

    return {
        "threshold": thresh,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": acc,
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "f1_score": f1,
    }


def _compute_all_point_metrics(
    y_bin: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Compute the 7 core evaluation metrics as floats."""
    m = compute_threshold_metrics(y_bin, scores, threshold=threshold)
    return {
        "accuracy": m["accuracy"],
        "sensitivity": m["sensitivity"],
        "specificity": m["specificity"],
        "precision": m["precision"],
        "f1_score": m["f1_score"],
        "roc_auc": _roc_auc_binary(y_bin, scores),
        "pr_auc": _pr_auc_binary(y_bin, scores),
    }


# ------------------------------------------------------------------------------
# Stratified Bootstrap Confidence Intervals
# ------------------------------------------------------------------------------

def _generate_stratified_boot_indices(
    y_bin: np.ndarray,
    n_boot: int,
    seed: int,
) -> np.ndarray:
    """Sample stratified bootstrap index matrix of shape (n_boot, N) preserving class counts."""
    valid_n_boot = _validate_int_param(n_boot, "n_boot", min_val=1)
    valid_seed = _validate_int_param(seed, "seed", min_val=0)

    pos_idx = np.flatnonzero(y_bin == 1)
    neg_idx = np.flatnonzero(y_bin == 0)
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError("Stratified bootstrap requires at least 1 positive and 1 negative sample")

    rng = np.random.default_rng(valid_seed)
    boot_pos = rng.choice(pos_idx, size=(valid_n_boot, len(pos_idx)), replace=True)
    boot_neg = rng.choice(neg_idx, size=(valid_n_boot, len(neg_idx)), replace=True)
    return np.concatenate([boot_pos, boot_neg], axis=1)


def _summarize_bootstrap_samples(
    point_metrics: Mapping[str, float],
    boot_samples: Mapping[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """Aggregate bootstrap arrays into point, ci_low (2.5%), ci_high (97.5%), and std."""
    summary: dict[str, dict[str, float]] = {}
    for metric_name, point_val in point_metrics.items():
        vals = boot_samples[metric_name]
        summary[metric_name] = {
            "point": float(point_val),
            "ci_low": float(np.percentile(vals, CI_LOWER_PERCENTILE)),
            "ci_high": float(np.percentile(vals, CI_UPPER_PERCENTILE)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        }
    return summary


def _stratified_bootstrap_from_indices(
    y_bin: np.ndarray,
    scores: np.ndarray,
    boot_indices: np.ndarray,
    threshold: float,
) -> dict[str, dict[str, float]]:
    """Compute bootstrap CIs reusing pre-sampled stratified index rows."""
    point_metrics = _compute_all_point_metrics(y_bin, scores, threshold=threshold)
    metric_keys = tuple(point_metrics.keys())
    boot_arrays: dict[str, np.ndarray] = {k: np.empty(len(boot_indices), dtype=np.float64) for k in metric_keys}

    for b_idx, row_idx in enumerate(boot_indices):
        res = _compute_all_point_metrics(y_bin[row_idx], scores[row_idx], threshold=threshold)
        for k in metric_keys:
            boot_arrays[k][b_idx] = res[k]

    return _summarize_bootstrap_samples(point_metrics, boot_arrays)


def _paired_bootstrap_delta_from_indices(
    y_bin: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    boot_indices: np.ndarray,
    threshold: float,
) -> dict[str, dict[str, float]]:
    """Compute paired metric difference CIs reusing pre-sampled stratified index rows."""
    ma_point = _compute_all_point_metrics(y_bin, scores_a, threshold=threshold)
    mb_point = _compute_all_point_metrics(y_bin, scores_b, threshold=threshold)
    point_deltas = {k: ma_point[k] - mb_point[k] for k in ma_point}

    boot_deltas: dict[str, np.ndarray] = {k: np.empty(len(boot_indices), dtype=np.float64) for k in point_deltas}
    for b_idx, row_idx in enumerate(boot_indices):
        yb = y_bin[row_idx]
        ma = _compute_all_point_metrics(yb, scores_a[row_idx], threshold=threshold)
        mb = _compute_all_point_metrics(yb, scores_b[row_idx], threshold=threshold)
        for k in point_deltas:
            boot_deltas[k][b_idx] = ma[k] - mb[k]

    return _summarize_bootstrap_samples(point_deltas, boot_deltas)


def stratified_bootstrap_ci(
    y_true: Any,
    y_score: Any,
    threshold: float = DEFAULT_THRESHOLD,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> dict[str, dict[str, float]]:
    """Compute point estimates and stratified percentile 95% CIs across 7 metrics."""
    y_bin, scores = _validate_binary_and_scores(y_true, y_score, require_both_classes=True)
    _validate_unit_score_array(scores, "y_score")
    thresh = _validate_unit_float(threshold, "threshold")
    indices = _generate_stratified_boot_indices(y_bin, n_boot=n_boot, seed=seed)
    return _stratified_bootstrap_from_indices(y_bin, scores, indices, thresh)


def paired_bootstrap_delta_ci(
    y_true: Any,
    scores_a: Any,
    scores_b: Any,
    threshold: float = DEFAULT_THRESHOLD,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> dict[str, dict[str, float]]:
    """Compute paired bootstrap 95% CIs for metric differences (Model A - Model B)."""
    y_bin, sa = _validate_binary_and_scores(y_true, scores_a, "scores_a", require_both_classes=True)
    _, sb = _validate_binary_and_scores(y_true, scores_b, "scores_b", require_both_classes=True)
    _validate_unit_score_array(sa, "scores_a")
    _validate_unit_score_array(sb, "scores_b")
    thresh = _validate_unit_float(threshold, "threshold")
    indices = _generate_stratified_boot_indices(y_bin, n_boot=n_boot, seed=seed)
    return _paired_bootstrap_delta_from_indices(y_bin, sa, sb, indices, thresh)


# ------------------------------------------------------------------------------
# Paired Significance Tests: Sun & Xu (2014) DeLong & Exact McNemar
# ------------------------------------------------------------------------------

def _delong_structural_components(
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute exact tie-aware DeLong structural components V10, V01 via midranks (Sun & Xu, 2014)."""
    m = len(pos_scores)
    n = len(neg_scores)
    tz = scipy_stats.rankdata(np.concatenate([pos_scores, neg_scores]), method="average")
    tx = scipy_stats.rankdata(pos_scores, method="average")
    ty = scipy_stats.rankdata(neg_scores, method="average")

    v10 = (tz[:m] - tx) / float(n)
    v01 = 1.0 - (tz[m:] - ty) / float(m)
    theta = float(np.mean(v10))
    return v10, v01, theta


def delong_roc_test(
    y_true: Any,
    scores_a: Any,
    scores_b: Any,
) -> dict[str, float]:
    """Compare two correlated ROC-AUCs using Sun & Xu (2014) DeLong two-sided z-test.

    Raises
    ------
    ValueError
        If inputs are invalid or either class has fewer than 2 samples.
    """
    y_bin, sa = _validate_binary_and_scores(y_true, scores_a, "scores_a", require_both_classes=True)
    _, sb = _validate_binary_and_scores(y_true, scores_b, "scores_b", require_both_classes=True)

    pos_mask = y_bin == 1
    neg_mask = y_bin == 0
    m = int(np.sum(pos_mask))
    n = int(np.sum(neg_mask))
    if m < MIN_DELONG_CLASS_SAMPLES or n < MIN_DELONG_CLASS_SAMPLES:
        raise ValueError(
            f"DeLong test requires at least {MIN_DELONG_CLASS_SAMPLES} positive and "
            f"{MIN_DELONG_CLASS_SAMPLES} negative samples, got m={m}, n={n}"
        )

    v10_a, v01_a, theta_a = _delong_structural_components(sa[pos_mask], sa[neg_mask])
    v10_b, v01_b, theta_b = _delong_structural_components(sb[pos_mask], sb[neg_mask])

    s10 = np.cov(np.vstack([v10_a, v10_b]), ddof=1)
    s01 = np.cov(np.vstack([v01_a, v01_b]), ddof=1)
    cov_auc = s10 / float(m) + s01 / float(n)

    var_a = max(0.0, float(cov_auc[0, 0]))
    var_b = max(0.0, float(cov_auc[1, 1]))
    var_diff = max(0.0, float(cov_auc[0, 0] + cov_auc[1, 1] - 2.0 * cov_auc[0, 1]))
    auc_diff = theta_a - theta_b

    # Guard against degenerate zero-variance subtraction artifacts or deterministic class separation
    if abs(auc_diff) <= ZERO_VARIANCE_EPS:
        z_stat = 0.0
        p_value = 1.0
    elif var_diff <= ZERO_VARIANCE_EPS:
        z_stat = auc_diff / math.sqrt(DELONG_MIN_VARIANCE_FLOOR)
        p_value = float(2.0 * scipy_stats.norm.sf(abs(z_stat)))
    else:
        z_stat = auc_diff / math.sqrt(var_diff)
        p_value = float(2.0 * scipy_stats.norm.sf(abs(z_stat)))

    return {
        "auc_a": float(theta_a),
        "auc_b": float(theta_b),
        "auc_diff": float(auc_diff),
        "var_a": var_a,
        "var_b": var_b,
        "var_diff": var_diff,
        "z_stat": float(z_stat),
        "p_value": float(np.clip(p_value, 0.0, 1.0)),
    }


def mcnemar_exact_test(
    y_true: Any,
    scores_or_preds_a: Any,
    scores_or_preds_b: Any,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    """Compute exact two-sided McNemar binomial test on paired binary classification outcomes."""
    y_bin, sa = _validate_binary_and_scores(y_true, scores_or_preds_a, "scores_or_preds_a", require_both_classes=True)
    _, sb = _validate_binary_and_scores(y_true, scores_or_preds_b, "scores_or_preds_b", require_both_classes=True)
    _validate_unit_score_array(sa, "scores_or_preds_a")
    _validate_unit_score_array(sb, "scores_or_preds_b")
    thresh = _validate_unit_float(threshold, "threshold")

    pred_a = (sa >= thresh).astype(np.int32)
    pred_b = (sb >= thresh).astype(np.int32)

    corr_a = pred_a == y_bin
    corr_b = pred_b == y_bin

    b_count = int(np.sum(corr_a & (~corr_b)))
    c_count = int(np.sum((~corr_a) & corr_b))
    n_disc = b_count + c_count

    if n_disc == 0:
        p_value = 1.0
        odds_ratio: float | None = 1.0
    else:
        p_value = float(scipy_stats.binomtest(b_count, n=n_disc, p=0.5, alternative="two-sided").pvalue)
        odds_ratio = float(b_count / c_count) if c_count > 0 else None

    return {
        "b_a_correct_b_wrong": b_count,
        "c_a_wrong_b_correct": c_count,
        "n_discordant": n_disc,
        "odds_ratio": odds_ratio,
        "p_value": float(np.clip(p_value, 0.0, 1.0)),
    }


# ------------------------------------------------------------------------------
# Clinical Operating Points & Bayes Prevalence Adjustment
# ------------------------------------------------------------------------------

def compute_operating_points(
    y_true: Any,
    y_score: Any,
) -> dict[str, dict[str, Any]]:
    """Evaluate clinical operating points: 0.50, Youden's J, Triage 0.35, and Spec@Sens>=0.90."""
    y_bin, scores = _validate_binary_and_scores(y_true, y_score, require_both_classes=True)
    _validate_unit_score_array(scores, "y_score")

    standard_050 = compute_threshold_metrics(y_bin, scores, threshold=DEFAULT_THRESHOLD)
    triage_035 = compute_threshold_metrics(y_bin, scores, threshold=CLINICAL_TRIAGE_THRESHOLD)

    unique_desc = np.sort(np.unique(scores))[::-1]
    best_youden_j = MIN_YOUDEN_SENTINEL
    best_youden_metrics: dict[str, Any] = {}
    spec90_metrics: dict[str, Any] | None = None

    for t in unique_desc:
        m = compute_threshold_metrics(y_bin, scores, threshold=float(t))
        j_val = m["sensitivity"] + m["specificity"] - 1.0
        if j_val > best_youden_j + ZERO_VARIANCE_EPS:
            best_youden_j = j_val
            best_youden_metrics = {**m, "youden_j": float(j_val)}
        elif abs(j_val - best_youden_j) <= ZERO_VARIANCE_EPS:
            prev_dist = abs(best_youden_metrics.get("threshold", DEFAULT_THRESHOLD) - DEFAULT_THRESHOLD)
            curr_dist = abs(float(t) - DEFAULT_THRESHOLD)
            if curr_dist < prev_dist:
                best_youden_metrics = {**m, "youden_j": float(j_val)}

        if spec90_metrics is None and m["sensitivity"] >= TARGET_HIGH_SENSITIVITY - SENSITIVITY_TARGET_TOL:
            spec90_metrics = m

    if spec90_metrics is None:
        spec90_metrics = compute_threshold_metrics(y_bin, scores, threshold=float(unique_desc[-1]))

    return {
        "standard_0_50": standard_050,
        "clinical_triage_0_35": triage_035,
        "youden_j_optimal": best_youden_metrics,
        "spec_at_sens_90": spec90_metrics,
    }


def prevalence_adjusted_ppv_npv(
    sensitivity: float,
    specificity: float,
    nevus_to_melanoma_ratio: float,
) -> dict[str, float]:
    """Compute Bayes-adjusted PPV and NPV at a specified benign:malignant ratio."""
    sens = _validate_unit_float(sensitivity, "sensitivity")
    spec = _validate_unit_float(specificity, "specificity")
    if (
        _is_bool_like(nevus_to_melanoma_ratio)
        or not isinstance(nevus_to_melanoma_ratio, (int, float, np.floating, np.integer))
        or not math.isfinite(float(nevus_to_melanoma_ratio))
        or float(nevus_to_melanoma_ratio) <= 0.0
    ):
        raise ValueError(f"nevus_to_melanoma_ratio must be finite > 0, got {nevus_to_melanoma_ratio}")

    ratio = float(nevus_to_melanoma_ratio)
    prevalence = 1.0 / (1.0 + ratio)
    neg_prev = 1.0 - prevalence

    tp_rate = sens * prevalence
    fp_rate = (1.0 - spec) * neg_prev
    tn_rate = spec * neg_prev
    fn_rate = (1.0 - sens) * prevalence

    ppv = float(tp_rate / (tp_rate + fp_rate)) if (tp_rate + fp_rate) > ZERO_VARIANCE_EPS else 0.0
    npv = float(tn_rate / (tn_rate + fn_rate)) if (tn_rate + fn_rate) > ZERO_VARIANCE_EPS else 0.0

    return {
        "nevus_to_melanoma_ratio": ratio,
        "prevalence": float(prevalence),
        "ppv": float(np.clip(ppv, 0.0, 1.0)),
        "npv": float(np.clip(npv, 0.0, 1.0)),
    }


def compute_prevalence_table(sensitivity: float, specificity: float) -> dict[str, dict[str, float]]:
    """Compute prevalence-adjusted PPV & NPV at 2:1, 6:1, and 10:1 nevus:melanoma ratios."""
    return {
        label: prevalence_adjusted_ppv_npv(sensitivity, specificity, ratio)
        for label, ratio in PREVALENCE_RATIOS
    }


# ------------------------------------------------------------------------------
# CSV Loading & Full Study Aggregation
# ------------------------------------------------------------------------------

def _parse_csv_grouped_rows(src: Path) -> dict[str, list[dict[str, Any]]]:
    """Read raw CSV rows and validate schema + finite numeric entries grouped by condition."""
    grouped_raw: dict[str, list[dict[str, Any]]] = {}
    with src.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"image_id", "label", "condition", "alpha", "mask_iou_vs_clean", *TIER_COLUMNS}
        if reader.fieldnames is None or (required - set(reader.fieldnames)):
            missing = required - set(reader.fieldnames or [])
            raise ValueError(f"Cases CSV missing required columns: {sorted(missing)}")

        for raw in reader:
            cond = str(raw["condition"]).strip()
            row: dict[str, Any] = {
                "image_id": str(raw["image_id"]),
                "label": int(raw["label"]),
                "alpha": float(raw["alpha"]),
                "mask_iou_vs_clean": float(raw["mask_iou_vs_clean"]),
            }
            for tier in TIER_COLUMNS:
                val = float(raw[tier])
                if not math.isfinite(val):
                    raise ValueError(f"Non-finite value in column '{tier}': {raw[tier]}")
                row[tier] = val
            grouped_raw.setdefault(cond, []).append(row)
    return grouped_raw


def load_cases_table(csv_path: str | Path) -> dict[str, dict[str, np.ndarray]]:
    """Load `artifact_study_cases.csv`, validate bounds, and normalize risk_* by / 100.0."""
    src = Path(csv_path)
    if not src.is_file():
        raise FileNotFoundError(f"Cases CSV not found at {src}")

    grouped_raw = _parse_csv_grouped_rows(src)
    if not grouped_raw:
        raise ValueError(f"Cases CSV at '{src}' contains 0 rows")

    by_condition: dict[str, dict[str, np.ndarray]] = {}
    for cond, rows in grouped_raw.items():
        # Deterministic sort by image_id so paired cross-condition comparisons always align 1:1
        rows_sorted = sorted(rows, key=lambda r: r["image_id"])
        lbl_arr = _validate_numeric_1d([r["label"] for r in rows_sorted], "label")
        unique_lbls = set(np.unique(lbl_arr))
        if not unique_lbls.issubset(EXPECTED_BINARY_CLASSES):
            raise ValueError(f"Condition '{cond}' has non-binary labels: {unique_lbls}")

        cond_dict: dict[str, np.ndarray] = {
            "image_id": np.array([r["image_id"] for r in rows_sorted], dtype=object),
            "label": (lbl_arr >= BINARY_SPLIT_MIDPOINT).astype(np.int32),
            "alpha": _validate_numeric_1d([r["alpha"] for r in rows_sorted], "alpha"),
            "mask_iou_vs_clean": _validate_numeric_1d(
                [r["mask_iou_vs_clean"] for r in rows_sorted],
                "mask_iou_vs_clean",
            ),
        }
        for tier in TIER_COLUMNS:
            cond_dict[tier] = normalize_tier_scores([r[tier] for r in rows_sorted], tier)
        by_condition[cond] = cond_dict

    return by_condition


def compute_artifact_resilience(
    by_condition: Mapping[str, Mapping[str, np.ndarray]],
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, dict[str, Any]]:
    """Compare each artifact-perturbed condition against clean baseline on matched image_ids."""
    if "clean" not in by_condition:
        raise ValueError("Baseline condition 'clean' is required for artifact resilience analysis")

    thresh = _validate_unit_float(threshold, "threshold")
    clean = by_condition["clean"]
    clean_ids, clean_y = clean["image_id"], clean["label"]
    neg_mask = clean_y == 0
    mean_alpha_clean = float(np.mean(clean["alpha"]))

    resilience: dict[str, dict[str, Any]] = {}
    for cond in PERTURBED_CONDITIONS:
        if cond not in by_condition:
            continue
        pert = by_condition[cond]
        if not np.array_equal(clean_ids, pert["image_id"]) or not np.array_equal(clean_y, pert["label"]):
            raise ValueError(f"Condition '{cond}' image_id/label alignment mismatch vs 'clean'")

        mean_alpha_pert = float(np.mean(pert["alpha"]))
        per_tier: dict[str, dict[str, Any]] = {}
        for tier in TIER_COLUMNS:
            pred_clean = clean[tier] >= thresh
            pred_pert = pert[tier] >= thresh
            flips = int(np.sum(pred_clean != pred_pert))
            spec_clean = float(np.mean(~pred_clean[neg_mask])) if np.any(neg_mask) else 0.0
            spec_pert = float(np.mean(~pred_pert[neg_mask])) if np.any(neg_mask) else 0.0
            per_tier[tier] = {
                "flip_count": flips,
                "flip_rate": float(flips / len(clean_y)),
                "n_samples": len(clean_y),
                "benign_spec_clean": spec_clean,
                "benign_spec_perturbed": spec_pert,
                "delta_specificity": spec_pert - spec_clean,
                "delta_fpr": (1.0 - spec_pert) - (1.0 - spec_clean),
            }

        resilience[cond] = {
            "mean_alpha_clean": mean_alpha_clean,
            "mean_alpha_perturbed": mean_alpha_pert,
            "delta_mean_alpha": mean_alpha_pert - mean_alpha_clean,
            "mean_mask_iou_vs_clean": float(np.mean(pert["mask_iou_vs_clean"])),
            "tiers": per_tier,
        }
    return resilience


def _evaluate_condition_block(
    data: Mapping[str, np.ndarray],
    boot_idx: np.ndarray,
    thresh: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute bootstrap CIs and paired DeLong/McNemar significance tests for one condition."""
    y_bin = data["label"]
    cond_metrics = {
        tier: _stratified_bootstrap_from_indices(y_bin, data[tier], boot_idx, thresh)
        for tier in TIER_COLUMNS
    }
    cond_paired = {
        f"{tier_a}_vs_{tier_b}": {
            "model_a": tier_a,
            "model_b": tier_b,
            "bootstrap_delta_ci": _paired_bootstrap_delta_from_indices(
                y_bin, data[tier_a], data[tier_b], boot_idx, thresh
            ),
            "delong": delong_roc_test(y_bin, data[tier_a], data[tier_b]),
            "mcnemar": mcnemar_exact_test(y_bin, data[tier_a], data[tier_b], threshold=thresh),
        }
        for tier_a, tier_b in PAIRED_COMPARISONS
    }
    return cond_metrics, cond_paired


def analyze_study_tables(
    by_condition: Mapping[str, Mapping[str, np.ndarray]],
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    """Run complete statistical evaluation across all conditions, tiers, and significance tests."""
    thresh = _validate_unit_float(threshold, "threshold")
    valid_n_boot = _validate_int_param(n_boot, "n_boot", min_val=1)
    valid_seed = _validate_int_param(seed, "seed", min_val=0)
    active_conditions = [c for c in CONDITIONS if c in by_condition]
    if not active_conditions:
        raise ValueError("No valid conditions present for analysis")

    metrics_by_condition: dict[str, Any] = {}
    paired_tests_by_condition: dict[str, Any] = {}
    for cond in active_conditions:
        data = by_condition[cond]
        boot_idx = _generate_stratified_boot_indices(data["label"], n_boot=valid_n_boot, seed=valid_seed)
        c_metrics, c_paired = _evaluate_condition_block(data, boot_idx, thresh)
        metrics_by_condition[cond] = c_metrics
        paired_tests_by_condition[cond] = c_paired

    operating_points_clean: dict[str, Any] = {}
    prevalence_clean: dict[str, Any] = {}
    if "clean" in by_condition:
        clean = by_condition["clean"]
        y_clean = clean["label"]
        for tier in TIER_COLUMNS:
            ops = compute_operating_points(y_clean, clean[tier])
            operating_points_clean[tier] = ops
            std_m = ops["standard_0_50"]
            prevalence_clean[tier] = compute_prevalence_table(
                std_m["sensitivity"], std_m["specificity"]
            )

    resilience = compute_artifact_resilience(by_condition, threshold=thresh) if "clean" in by_condition else {}

    return {
        "metadata": {
            "n_boot": valid_n_boot,
            "seed": valid_seed,
            "threshold_normalized": thresh,
            "risk_normalization_divisor": RISK_NORMALIZATION_DIVISOR,
            "conditions": active_conditions,
            "tiers": list(TIER_COLUMNS),
        },
        "metrics_with_ci": metrics_by_condition,
        "paired_significance": paired_tests_by_condition,
        "operating_points_clean": operating_points_clean,
        "prevalence_adjusted_clean": prevalence_clean,
        "artifact_resilience": resilience,
    }


# ------------------------------------------------------------------------------
# Markdown Report Formatting & File Persistence
# ------------------------------------------------------------------------------

def _fmt_ci(stat: Mapping[str, float], pct: bool = True) -> str:
    """Format point [ci_low, ci_high] cleanly for Markdown tables."""
    scale = PERCENT_SCALE if pct else 1.0
    fmt = ".1f" if pct else ".4f"
    suffix = "%" if pct else ""
    return (
        f"{stat['point'] * scale:{fmt}}{suffix} "
        f"[{stat['ci_low'] * scale:{fmt}}, {stat['ci_high'] * scale:{fmt}}]"
    )


def _format_metrics_section(metrics_with_ci: Mapping[str, Any]) -> list[str]:
    """Render Section 1 (metrics & 95% CIs across conditions)."""
    lines = ["## 1. Diagnostic Performance & Stratified 95% CIs Across Conditions", ""]
    for cond, tiers in metrics_with_ci.items():
        lines.extend([
            f"### Condition: `{cond}`",
            "",
            "| Tier | Accuracy | Sensitivity | Specificity | Precision | F1-Score | ROC-AUC | PR-AUC |",
            "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
        ])
        for tier in TIER_COLUMNS:
            if tier not in tiers:
                continue
            m = tiers[tier]
            lines.append(
                f"| `{tier}` | {_fmt_ci(m['accuracy'])} | {_fmt_ci(m['sensitivity'])} | "
                f"{_fmt_ci(m['specificity'])} | {_fmt_ci(m['precision'])} | {_fmt_ci(m['f1_score'])} | "
                f"{_fmt_ci(m['roc_auc'], pct=False)} | {_fmt_ci(m['pr_auc'], pct=False)} |"
            )
        lines.append("")
    return lines


def _format_paired_section(paired_sig: Mapping[str, Any]) -> list[str]:
    """Render Section 2 (paired DeLong, McNemar, and bootstrap deltas)."""
    lines = ["## 2. Paired Statistical Significance (DeLong, McNemar & Paired Bootstrap)", ""]
    for cond, pairs in paired_sig.items():
        lines.extend([
            f"### Condition: `{cond}`",
            "",
            "| Comparison | ΔROC-AUC [95% CI] | DeLong z | DeLong p-value | ΔSpec [95% CI] | McNemar (b/c) | McNemar p-value |",
            "|---|:---:|:---:|:---:|:---:|:---:|:---:|",
        ])
        for res in pairs.values():
            dl, mc = res["delong"], res["mcnemar"]
            bd_auc = res["bootstrap_delta_ci"]["roc_auc"]
            bd_spec = res["bootstrap_delta_ci"]["specificity"]
            lines.append(
                f"| `{res['model_a']}` vs `{res['model_b']}` | "
                f"{_fmt_ci(bd_auc, pct=False)} | {dl['z_stat']:.3f} | {dl['p_value']:.4e} | "
                f"{_fmt_ci(bd_spec, pct=True)} | {mc['b_a_correct_b_wrong']}/{mc['c_a_wrong_b_correct']} | "
                f"{mc['p_value']:.4e} |"
            )
        lines.append("")
    return lines


def _format_operating_and_prevalence(summary: Mapping[str, Any]) -> list[str]:
    """Render Sections 3 & 4 (clean operating points and Bayes PPV/NPV tables)."""
    if not summary.get("operating_points_clean"):
        return []
    lines = [
        "## 3. Clinical Operating Points (`clean`)",
        "",
        "| Tier | Operating Point | Cutoff | Sensitivity | Specificity | Accuracy | F1-Score |",
        "|---|---|:---:|:---:|:---:|:---:|:---:|",
    ]
    for tier, ops in summary["operating_points_clean"].items():
        for op_name, om in ops.items():
            lines.append(
                f"| `{tier}` | `{op_name}` | {om['threshold']:.4f} | "
                f"{om['sensitivity']*PERCENT_SCALE:.1f}% | {om['specificity']*PERCENT_SCALE:.1f}% | "
                f"{om['accuracy']*PERCENT_SCALE:.1f}% | {om['f1_score']*PERCENT_SCALE:.1f}% |"
            )
    lines.extend([
        "",
        "## 4. Bayes Prevalence-Adjusted PPV & NPV (`clean`, cutoff=0.50)",
        "",
        "| Tier | Nevus:Melanoma | Prevalence | PPV | NPV |",
        "|---|:---:|:---:|:---:|:---:|",
    ])
    for tier, pmap in summary["prevalence_adjusted_clean"].items():
        for r_label, pv in pmap.items():
            lines.append(
                f"| `{tier}` | {r_label} | {pv['prevalence']*PERCENT_SCALE:.1f}% | "
                f"{pv['ppv']*PERCENT_SCALE:.1f}% | {pv['npv']*PERCENT_SCALE:.1f}% |"
            )
    lines.append("")
    return lines


def _format_resilience_section(resilience: Mapping[str, Any]) -> list[str]:
    """Render Section 5 (cross-condition artifact resilience vs clean)."""
    if not resilience:
        return []
    lines = [
        "## 5. Artifact Resilience vs `clean`",
        "",
        "| Condition | Tier | Flips (Rate) | Benign Spec | ΔSpec | ΔFPR | Mean α Shift | Mean Mask IoU |",
        "|---|---|---|:---:|:---:|:---:|:---:|:---:|",
    ]
    for cond, cres in resilience.items():
        for tier, tres in cres["tiers"].items():
            lines.append(
                f"| `{cond}` | `{tier}` | {tres['flip_count']}/{tres['n_samples']} ({tres['flip_rate']*PERCENT_SCALE:.1f}%) | "
                f"{tres['benign_spec_perturbed']*PERCENT_SCALE:.1f}% | {tres['delta_specificity']*PERCENT_SCALE:+.1f}% | "
                f"{tres['delta_fpr']*PERCENT_SCALE:+.1f}% | {cres['delta_mean_alpha']:+.4f} | "
                f"{cres['mean_mask_iou_vs_clean']:.4f} |"
            )
    lines.append("")
    return lines


def format_stats_markdown(summary: Mapping[str, Any]) -> str:
    """Format full statistical summary into a publication-grade Markdown report."""
    meta = summary["metadata"]
    lines: list[str] = [
        "# DermaFuzzy-XAI Q1 Statistical Analysis & Artifact Perturbation Report",
        "",
        f"- **Stratified Bootstrap Replicates**: `{meta['n_boot']}` (`seed={meta['seed']}`)",
        f"- **Normalized Decision Cutoff**: `{meta['threshold_normalized']:.2f}` (`risk_*` normalized by `/ {meta['risk_normalization_divisor']:.0f}`)",
        "",
    ]
    lines.extend(_format_metrics_section(summary["metrics_with_ci"]))
    lines.extend(_format_paired_section(summary["paired_significance"]))
    lines.extend(_format_operating_and_prevalence(summary))
    lines.extend(_format_resilience_section(summary.get("artifact_resilience", {})))
    return "\n".join(lines)


def run_stats_analysis(
    cases_csv: str | Path,
    out_dir: str | Path,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[Path, Path]:
    """Execute end-to-end statistical analysis from CSV and persist JSON + Markdown outputs."""
    by_condition = load_cases_table(cases_csv)
    summary = analyze_study_tables(by_condition, n_boot=n_boot, seed=seed, threshold=threshold)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    json_path = out_path / SUMMARY_JSON_FILENAME
    md_path = out_path / REPORT_MD_FILENAME

    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md_path.write_text(format_stats_markdown(summary) + "\n", encoding="utf-8")
    return json_path, md_path


def main() -> None:
    """CLI entrypoint for Q1 statistical analysis."""
    parser = argparse.ArgumentParser(
        description="DermaFuzzy-XAI Q1 Statistical Evaluation (DeLong, McNemar, Bootstrap CIs, Bayes PPV/NPV)"
    )
    parser.add_argument("--cases-csv", type=str, default=DEFAULT_CASES_CSV, help="Path to artifact_study_cases.csv")
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR, help="Output directory for JSON/MD reports")
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT, help="Stratified bootstrap replicates")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Normalized cutoff in [0, 1]")

    args = parser.parse_args()
    json_p, md_p = run_stats_analysis(
        cases_csv=Path(args.cases_csv),
        out_dir=Path(args.out_dir),
        n_boot=args.n_boot,
        seed=args.seed,
        threshold=args.threshold,
    )
    print(f"Saved statistical summary JSON: {json_p}")
    print(f"Saved statistical report Markdown: {md_p}")


if __name__ == "__main__":
    main()
