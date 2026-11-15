"""Room 5: Evaluation & 3-Tier Ablation Study Engine for DermaFuzzy-XAI.

Provides:
- ClinicalMetrics: Immutable biostatistical metrics container (TP, TN, FP, FN, Sensitivity,
  Specificity, Precision, Accuracy, F1-Score, AUC-ROC).
- AblationReport: Immutable 3-tier comparative ablation container with Markdown and CSV export.
- compute_clinical_metrics: Pure NumPy confusion matrix and exact tied-score trapezoidal AUC-ROC.
- evaluate_crisp_abcd_baseline: Baseline 2 step indicators at median P50.
- run_ablation_study: 3-tier ablation study comparing Pure CNN Black-Box, Crisp ABCD, and DermaFuzzy-XAI.

Strictly adheres to Constitution Principle I (Pure Core: zero tensorflow/torch/sklearn imports).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from dermafuzzy_core.biomarkers import BiomarkerResult
from dermafuzzy_core.config import DermaConfig, load_config
from dermafuzzy_core.fis_engine import FISCalibration, evaluate_mamdani_fis


@dataclass(frozen=True, slots=True)
class ClinicalMetrics:
    """Immutable container storing biostatistical diagnostic classification metrics."""

    tp: int
    tn: int
    fp: int
    fn: int
    sensitivity: float
    specificity: float
    precision: float
    accuracy: float
    f1_score: float
    auc_roc: float
    threshold: float = 0.50

    def __post_init__(self) -> None:
        """Validate counts, bounds, and immutability invariants."""
        # Validate confusion matrix integer counts (reject bool)
        for name in ("tp", "tn", "fp", "fn"):
            val = getattr(self, name)
            if type(val) is bool or not isinstance(val, (int, np.integer)):
                raise ValueError(f"Field '{name}' must be non-negative integer, got {type(val).__name__}: {val}")
            if val < 0:
                raise ValueError(f"Field '{name}' must be non-negative integer, got {val}")

        # Validate rates and probabilities
        rates = {
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "precision": self.precision,
            "accuracy": self.accuracy,
            "f1_score": self.f1_score,
            "auc_roc": self.auc_roc,
            "threshold": self.threshold,
        }
        for name, val in rates.items():
            if type(val) is bool or not isinstance(val, (int, float, np.floating, np.integer)):
                raise ValueError(f"Field '{name}' must be finite float, got {type(val).__name__}: {val}")
            f_val = float(val)
            if not math.isfinite(f_val):
                raise ValueError(f"Field '{name}' must be finite, got {f_val}")
            if not (0.0 <= f_val <= 1.0):
                raise ValueError(f"Field '{name}' must be in [0.0, 1.0], got {f_val}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize metrics to a standard Python dictionary."""
        return {
            "tp": int(self.tp),
            "tn": int(self.tn),
            "fp": int(self.fp),
            "fn": int(self.fn),
            "sensitivity": float(self.sensitivity),
            "specificity": float(self.specificity),
            "precision": float(self.precision),
            "accuracy": float(self.accuracy),
            "f1_score": float(self.f1_score),
            "auc_roc": float(self.auc_roc),
            "threshold": float(self.threshold),
        }


@dataclass(frozen=True, slots=True)
class AblationReport:
    """Immutable container holding 3-tier comparative ablation study results."""

    tiers: Mapping[str, ClinicalMetrics]
    sample_count: int
    generated_at: str

    def __init__(
        self,
        tiers: Mapping[str, ClinicalMetrics],
        sample_count: int,
        generated_at: str | None = None,
    ) -> None:
        if type(sample_count) is bool or not isinstance(sample_count, (int, np.integer)) or sample_count < 0:
            raise ValueError(f"sample_count must be non-negative int, got {sample_count}")

        if not isinstance(tiers, (dict, MappingProxyType, Mapping)):
            raise ValueError(f"tiers must be a mapping, got {type(tiers).__name__}")

        validated_tiers: dict[str, ClinicalMetrics] = {}
        for k, v in tiers.items():
            if not isinstance(k, str) or not k:
                raise ValueError(f"Tier name must be non-empty string, got {k!r}")
            if not isinstance(v, ClinicalMetrics):
                raise ValueError(f"Tier '{k}' value must be ClinicalMetrics, got {type(v).__name__}")
            validated_tiers[k] = v

        if generated_at is None:
            generated_at = datetime.now(timezone.utc).isoformat()
        elif not isinstance(generated_at, str) or not generated_at:
            raise ValueError(f"generated_at must be non-empty string, got {generated_at!r}")

        object.__setattr__(self, "tiers", MappingProxyType(validated_tiers))
        object.__setattr__(self, "sample_count", int(sample_count))
        object.__setattr__(self, "generated_at", str(generated_at))

    def to_dict(self) -> dict[str, Any]:
        """Serialize ablation report to dictionary."""
        return {
            "tiers": {k: v.to_dict() for k, v in self.tiers.items()},
            "sample_count": self.sample_count,
            "generated_at": self.generated_at,
        }

    def to_markdown_table(self) -> str:
        """Format ablation study results as a publication-grade GitHub Markdown table."""
        tier_display_names = {
            "1_pure_mobilenetv2_blackbox": "Tier 1: Pure MobileNetV2 Black-Box",
            "2_crisp_abcd_fixed_thresholds": "Tier 2: Crisp ABCD Fixed Thresholds",
            "3_proposed_dermafuzzy_xai": "Tier 3: Proposed DermaFuzzy-XAI",
        }

        lines = [
            "| Architecture Tier | Accuracy | Sensitivity | Specificity | Precision | F1-Score | AUC-ROC |",
            "|---|:---:|:---:|:---:|:---:|:---:|:---:|",
        ]

        for key, metrics in self.tiers.items():
            name = tier_display_names.get(key, key)
            acc_str = f"{metrics.accuracy * 100:.1f}%"
            sens_str = f"{metrics.sensitivity * 100:.1f}%"
            spec_str = f"{metrics.specificity * 100:.1f}%"
            prec_str = f"{metrics.precision * 100:.1f}%"
            f1_str = f"{metrics.f1_score * 100:.1f}%"
            auc_str = f"{metrics.auc_roc:.4f}"
            lines.append(f"| {name} | {acc_str} | {sens_str} | {spec_str} | {prec_str} | {f1_str} | {auc_str} |")

        return "\n".join(lines)

    def to_csv(self) -> str:
        """Format ablation study metrics as CSV."""
        headers = [
            "tier",
            "tp",
            "tn",
            "fp",
            "fn",
            "sensitivity",
            "specificity",
            "precision",
            "accuracy",
            "f1_score",
            "auc_roc",
            "threshold",
        ]
        lines = [",".join(headers)]
        for key, m in self.tiers.items():
            row = [
                key,
                str(m.tp),
                str(m.tn),
                str(m.fp),
                str(m.fn),
                f"{m.sensitivity:.6f}",
                f"{m.specificity:.6f}",
                f"{m.precision:.6f}",
                f"{m.accuracy:.6f}",
                f"{m.f1_score:.6f}",
                f"{m.auc_roc:.6f}",
                f"{m.threshold:.4f}",
            ]
            lines.append(",".join(row))
        return "\n".join(lines)


def _validate_1d_array(arr: Any, name: str) -> np.ndarray:
    """Validate 1D numerical array rejecting bool, NaN, Inf, and empty arrays."""
    if type(arr) is bool:
        raise ValueError(f"Boolean input is invalid for '{name}'")
    if isinstance(arr, (list, tuple)):
        for item in arr:
            if type(item) is bool:
                raise ValueError(f"Boolean element found in '{name}'")

    np_arr = np.asarray(arr)
    if np_arr.dtype == bool:
        raise ValueError(f"Boolean array is invalid for '{name}'")

    if np_arr.ndim != 1:
        raise ValueError(f"'{name}' must be 1-dimensional, got shape {np_arr.shape}")

    if np_arr.size == 0:
        raise ValueError(f"'{name}' cannot be empty")

    if not np.all(np.isfinite(np_arr)):
        raise ValueError(f"'{name}' contains NaN or Inf values")

    return np_arr


def compute_clinical_metrics(
    y_true: np.ndarray | Sequence[int],
    y_prob: np.ndarray | Sequence[float],
    threshold: float = 0.50,
) -> ClinicalMetrics:
    """Compute biostatistical clinical classification metrics and pure NumPy AUC-ROC.

    Args:
        y_true: 1D array-like of binary ground truth labels in {0, 1}.
        y_prob: 1D array-like of continuous prediction probabilities in [0.0, 1.0]
                (auto-normalized if supplied in [0.0, 100.0]).
        threshold: Decision cutoff for binary classification (default 0.50).

    Returns:
        Immutable ClinicalMetrics dataclass.

    Raises:
        ValueError: If inputs have mismatched lengths, invalid shapes, non-finite values,
                    or invalid labels outside {0, 1}.
    """
    if type(threshold) is bool or not isinstance(threshold, (int, float, np.floating, np.integer)):
        raise ValueError(f"threshold must be float in [0.0, 1.0], got {threshold}")
    f_thresh = float(threshold)
    if not (0.0 <= f_thresh <= 1.0):
        raise ValueError(f"threshold must be in [0.0, 1.0], got {f_thresh}")

    yt = _validate_1d_array(y_true, "y_true")
    yp = _validate_1d_array(y_prob, "y_prob").astype(np.float64)

    if len(yt) != len(yp):
        raise ValueError(f"Length mismatch: y_true has length {len(yt)}, y_prob has length {len(yp)}")

    # Verify binary ground truth
    unique_yt = set(np.unique(yt))
    if not unique_yt.issubset({0, 1, 0.0, 1.0}):
        raise ValueError(f"y_true must contain only binary labels {0, 1}, found {unique_yt}")
    yt_bin = (yt >= 0.5).astype(np.int32)

    # Auto-normalize probability if provided on [0, 100] scale
    if np.max(yp) > 1.0 and np.max(yp) <= 100.0:
        yp = yp / 100.0

    if np.min(yp) < 0.0 or np.max(yp) > 1.0:
        raise ValueError(f"y_prob values must be in [0.0, 1.0], got range [{np.min(yp)}, {np.max(yp)}]")

    # Binary classification decisions at threshold
    y_pred = (yp >= f_thresh).astype(np.int32)

    tp = int(np.sum((yt_bin == 1) & (y_pred == 1)))
    tn = int(np.sum((yt_bin == 0) & (y_pred == 0)))
    fp = int(np.sum((yt_bin == 0) & (y_pred == 1)))
    fn = int(np.sum((yt_bin == 1) & (y_pred == 0)))

    # Compute rates with epsilon stabilization for clinical safety
    pos_cases = tp + fn
    neg_cases = tn + fp
    total_cases = tp + tn + fp + fn

    sens = float(tp / pos_cases) if pos_cases > 0 else 0.0
    spec = float(tn / neg_cases) if neg_cases > 0 else 0.0
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    acc = float((tp + tn) / total_cases) if total_cases > 0 else 0.0

    if (prec + sens) > 0.0:
        f1 = float(2.0 * prec * sens / (prec + sens))
    else:
        f1 = 0.0

    # Exact Pure NumPy Tied-Score Trapezoidal AUC-ROC
    if pos_cases == 0 or neg_cases == 0 or np.all(yp == yp[0]):
        auc_roc = 0.50
    else:
        # Sort distinct thresholds in descending order
        distinct_v = np.sort(np.unique(yp))[::-1]
        tpr_list = [0.0]
        fpr_list = [0.0]

        for v in distinct_v:
            pred_k = yp >= v
            tp_k = np.sum((pred_k == 1) & (yt_bin == 1))
            fp_k = np.sum((pred_k == 1) & (yt_bin == 0))
            tpr_list.append(float(tp_k / pos_cases))
            fpr_list.append(float(fp_k / neg_cases))

        # Trapezoidal integration along FPR axis
        _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
        if _trapz is not None:
            raw_auc = float(_trapz(tpr_list, fpr_list))
        else:
            # Fallback manual trapezoidal integration
            raw_auc = 0.0
            for i in range(1, len(fpr_list)):
                dx = fpr_list[i] - fpr_list[i - 1]
                avg_y = (tpr_list[i] + tpr_list[i - 1]) / 2.0
                raw_auc += avg_y * dx

        auc_roc = float(np.clip(raw_auc, 0.0, 1.0))

    return ClinicalMetrics(
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        sensitivity=float(np.clip(sens, 0.0, 1.0)),
        specificity=float(np.clip(spec, 0.0, 1.0)),
        precision=float(np.clip(prec, 0.0, 1.0)),
        accuracy=float(np.clip(acc, 0.0, 1.0)),
        f1_score=float(np.clip(f1, 0.0, 1.0)),
        auc_roc=auc_roc,
        threshold=f_thresh,
    )


def _extract_biomarkers_matrix(
    biomarkers_batch: np.ndarray | Sequence[Sequence[float]] | Sequence[BiomarkerResult],
) -> np.ndarray:
    """Convert biomarker batch sequence to (N, 4) float array."""
    if isinstance(biomarkers_batch, Sequence) and len(biomarkers_batch) > 0:
        first = biomarkers_batch[0]
        if isinstance(first, BiomarkerResult):
            rows = []
            for item in biomarkers_batch:
                if not isinstance(item, BiomarkerResult):
                    raise ValueError(f"Inconsistent batch items: expected BiomarkerResult, got {type(item).__name__}")
                rows.append([item.A, item.B_fuzzy, item.C, item.D_verified])
            arr = np.array(rows, dtype=np.float64)
        else:
            arr = np.asarray(biomarkers_batch, dtype=np.float64)
    else:
        arr = np.asarray(biomarkers_batch, dtype=np.float64)

    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"biomarkers_batch must have shape (N, 4), got {arr.shape}")

    if not np.all(np.isfinite(arr)):
        raise ValueError("biomarkers_batch contains non-finite NaN or Inf values")

    return arr


def evaluate_crisp_abcd_baseline(
    biomarkers_batch: np.ndarray | Sequence[Sequence[float]] | Sequence[BiomarkerResult],
    calibration: FISCalibration,
) -> np.ndarray:
    """Evaluate Baseline 2 (Crisp ABCD + Fixed Thresholds) using step indicators at P50.

    Each biomarker x_j is evaluated against the median cutoff P50:
        I_j = 1 if x_j >= P50_j else 0
    Risk score is the mean fraction of triggered indicators:
        risk_crisp = (1/4) * sum(I_j) in [0.0, 1.0]

    Args:
        biomarkers_batch: Batch of ABCD biomarker vectors (N, 4).
        calibration: Calibrated FIS percentiles container supplying P50 cutoffs.

    Returns:
        1D float NumPy array of shape (N,) in [0.0, 1.0].

    Raises:
        ValueError: If input dimensions or biomarker scores are invalid.
    """
    if not isinstance(calibration, FISCalibration):
        raise ValueError(f"calibration must be an instance of FISCalibration, got {type(calibration).__name__}")

    arr = _extract_biomarkers_matrix(biomarkers_batch)

    # Cutoffs at median P50 for [A, B, C, D]
    p50_a = calibration.cutoffs["A"][1]
    p50_b = (
        calibration.cutoffs["B_fuzzy"][1]
        if "B_fuzzy" in calibration.cutoffs
        else calibration.cutoffs["B"][1]
    )
    p50_c = calibration.cutoffs["C"][1]
    p50_d = (
        calibration.cutoffs["D_verified"][1]
        if "D_verified" in calibration.cutoffs
        else calibration.cutoffs["D"][1]
    )

    p50_cutoffs = np.array([p50_a, p50_b, p50_c, p50_d], dtype=np.float64)

    # Compute step indicator matrix (N, 4)
    indicators = (arr >= p50_cutoffs).astype(np.float64)

    # Fraction of triggered indicators in [0.0, 1.0]
    scores = np.mean(indicators, axis=1)
    return scores


def run_ablation_study(
    y_true: np.ndarray | Sequence[int],
    p_cnn_raw: np.ndarray | Sequence[float],
    biomarkers_batch: np.ndarray | Sequence[Sequence[float]] | Sequence[BiomarkerResult],
    calibration: FISCalibration,
    config: DermaConfig | None = None,
) -> AblationReport:
    """Execute 3-tier comparative ablation study.

    Tiers evaluated:
    1. Pure MobileNetV2 Black-Box (unverified CNN confidence p_cnn_raw).
    2. Crisp ABCD + Fixed Thresholds (step indicators at median P50).
    3. Proposed DermaFuzzy-XAI (saliency-gated Mamdani FIS).

    Args:
        y_true: Ground truth binary labels in {0, 1}.
        p_cnn_raw: Unverified deep CNN confidence probabilities in [0.0, 1.0].
        biomarkers_batch: Batch of ABCD biomarker vectors (N, 4).
        calibration: Calibrated FIS percentiles container.
        config: Optional DermaConfig instance. Defaults to load_config().

    Returns:
        Immutable AblationReport dataclass with comparative metrics and export formatters.
    """
    if config is None:
        config = load_config()

    yt = _validate_1d_array(y_true, "y_true")
    p_cnn = _validate_1d_array(p_cnn_raw, "p_cnn_raw").astype(np.float64)
    mat = _extract_biomarkers_matrix(biomarkers_batch)

    n_samples = len(yt)
    if len(p_cnn) != n_samples or len(mat) != n_samples:
        raise ValueError(
            f"Length mismatch: y_true={n_samples}, p_cnn_raw={len(p_cnn)}, biomarkers_batch={len(mat)}"
        )

    # Tier 1: Pure MobileNetV2 Black-Box
    metrics_tier1 = compute_clinical_metrics(yt, p_cnn, threshold=0.50)

    # Tier 2: Crisp ABCD + Fixed Thresholds
    p_crisp = evaluate_crisp_abcd_baseline(mat, calibration)
    metrics_tier2 = compute_clinical_metrics(yt, p_crisp, threshold=0.50)

    # Tier 3: Proposed DermaFuzzy-XAI
    p_fuzzy = np.zeros(n_samples, dtype=np.float64)
    for i in range(n_samples):
        b_vec = mat[i]
        fis_pred = evaluate_mamdani_fis(b_vec, calibration, config=config)
        p_fuzzy[i] = fis_pred.risk_score / 100.0

    metrics_tier3 = compute_clinical_metrics(yt, p_fuzzy, threshold=0.50)

    tiers = {
        "1_pure_mobilenetv2_blackbox": metrics_tier1,
        "2_crisp_abcd_fixed_thresholds": metrics_tier2,
        "3_proposed_dermafuzzy_xai": metrics_tier3,
    }

    return AblationReport(tiers=tiers, sample_count=n_samples)
