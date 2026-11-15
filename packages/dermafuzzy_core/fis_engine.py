"""Mamdani Fuzzy Inference System & Clinical Explanation Engine (Room 4).

This module implements pure, deterministic, and auditable Mamdani Fuzzy Inference System (FIS)
logic for malignant melanoma risk stratification, governed by:
- Constitution Principle I: Zero top-level heavy framework imports (TensorFlow, Keras, PyTorch).
- Constitution Principle II: Strict [0.0, 100.0] risk bounds, Partition of Unity.
- Constitution Principle III: Zero pickle files; calibration strictly from training split via JSON.
- Constitution Principle IV: Config-driven parameters from root derma_config.yaml.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Mapping, Sequence

import numpy as np
import skfuzzy as fuzz

from dermafuzzy_core.config import CANONICAL_FEATURE_NAMES, DermaConfig, load_config

if TYPE_CHECKING:
    from dermafuzzy_core.biomarkers import BiomarkerResult
    from dermafuzzy_core.preprocessing import PreprocessingResult

# Algorithmic & Defuzzification Constants
MAMDANI_UNIVERSE_POINTS: Final[int] = 1001
DEGENERATE_SEPARATION_DELTA: Final[float] = 1e-4
CENTROID_POTENTIAL_ALPHA: Final[float] = 0.70
PURE_LOW_CENTROID_BOUND: Final[float] = 13.166666666666664
PURE_HIGH_CENTROID_BOUND: Final[float] = 85.90909090909089
DEFAULT_FALLBACK_CENTROID: Final[float] = 50.0
EPSILON_SUM_GUARD: Final[float] = 1e-4
TIE_BREAK_EPSILON: Final[float] = 1e-6
MIN_CALIBRATION_SAMPLES: Final[int] = 4


# ------------------------------------------------------------------------------
# Domain Dataclasses
# ------------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class FuzzyMembershipSet:
    """Linguistic fuzzy membership degrees for a single biomarker satisfying Partition of Unity."""

    low: float
    medium: float
    high: float

    def __post_init__(self) -> None:
        for val, name in ((self.low, "low"), (self.medium, "medium"), (self.high, "high")):
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(f"Fuzzy membership '{name}' must be a finite float, got {val!r}")
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"Fuzzy membership '{name}' must be in [0.0, 1.0], got {val}")

        tot = self.low + self.medium + self.high
        if not math.isclose(tot, 1.0, abs_tol=1e-5):
            raise ValueError(f"Partition of Unity violated: sum={tot} != 1.0")

    def as_tuple(self) -> tuple[float, float, float]:
        """Return (low, medium, high) as a tuple."""
        return (self.low, self.medium, self.high)

    def as_dict(self) -> dict[str, float]:
        """Return memberships as a dictionary."""
        return {"low": self.low, "medium": self.medium, "high": self.high}


@dataclass(frozen=True, slots=True)
class ClinicalRule:
    """Auditable clinical rule metadata."""

    rule_id: str
    name: str
    consequent: str
    description: str


CLINICAL_RULES: tuple[ClinicalRule, ...] = (
    ClinicalRule(
        rule_id="R1",
        name="Classic ABCD Malignant",
        consequent="High",
        description="High Asymmetry, Border Fuzziness, and Color Variation indicate severe structural lesion atypia.",
    ),
    ClinicalRule(
        rule_id="R2",
        name="Deep-Verified Malignant",
        consequent="High",
        description="High Border Fuzziness and Color Variation confirmed by Saliency-Verified Deep CNN Attention.",
    ),
    ClinicalRule(
        rule_id="R3",
        name="Early Structural Malignant",
        consequent="High",
        description="High Asymmetry or Border Fuzziness confirmed by Saliency-Verified Deep CNN Attention.",
    ),
    ClinicalRule(
        rule_id="R4",
        name="Borderline Dysplastic Nevus",
        consequent="Medium",
        description="Moderate clinical biomarkers warranting dermoscopic follow-up and clinical surveillance.",
    ),
    ClinicalRule(
        rule_id="R5",
        name="Benign Nevus & Artifact Protection",
        consequent="Low",
        description="Low ABCD structural/color variance with suppressed or low deep risk confirming benign lesion.",
    ),
)


@dataclass(frozen=True, slots=True)
class FISCalibration:
    """Immutable calibrated biomarker percentiles container."""

    cutoffs: Mapping[str, tuple[float, float, float]]
    n_samples: int
    generated_at: str
    config_hash: str

    def __init__(
        self,
        cutoffs: Mapping[str, tuple[float, float, float]],
        n_samples: int,
        generated_at: str,
        config_hash: str,
    ) -> None:
        if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < MIN_CALIBRATION_SAMPLES:
            raise ValueError(f"n_samples must be an integer >= {MIN_CALIBRATION_SAMPLES}, got {n_samples!r}")

        expected_keys = set(CANONICAL_FEATURE_NAMES)
        actual_keys = set(cutoffs.keys())
        if actual_keys != expected_keys:
            raise ValueError(f"cutoffs keys must exactly match {expected_keys}, got {actual_keys}")

        validated_cutoffs: dict[str, tuple[float, float, float]] = {}
        for feat in CANONICAL_FEATURE_NAMES:
            c = cutoffs[feat]
            if len(c) != 3:
                raise ValueError(f"Cutoffs for '{feat}' must be a 3-element tuple, got {c}")
            p25, p50, p75 = c
            for val, name in ((p25, "p25"), (p50, "p50"), (p75, "p75")):
                if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
                    raise ValueError(f"Cutoff {name} for '{feat}' must be a finite float, got {val!r}")
            if not (0.0 < p25 < p50 < p75 < 1.0):
                raise ValueError(
                    f"Strict monotonicity violated for '{feat}': must satisfy 0 < p25 < p50 < p75 < 1, "
                    f"got ({p25}, {p50}, {p75})"
                )
            if (p50 - p25) < DEGENERATE_SEPARATION_DELTA - 1e-9 or (p75 - p50) < DEGENERATE_SEPARATION_DELTA - 1e-9:
                raise ValueError(
                    f"Degenerate separation guard violated for '{feat}': step must be >= {DEGENERATE_SEPARATION_DELTA}, "
                    f"got steps ({p50 - p25:.6e}, {p75 - p50:.6e})"
                )
            validated_cutoffs[feat] = (float(p25), float(p50), float(p75))

        object.__setattr__(self, "cutoffs", MappingProxyType(validated_cutoffs))
        object.__setattr__(self, "n_samples", int(n_samples))
        object.__setattr__(self, "generated_at", str(generated_at))
        object.__setattr__(self, "config_hash", str(config_hash))

    def get_cutoffs(self, feature_name: str) -> tuple[float, float, float]:
        """Retrieve the (P25, P50, P75) cutoffs for a biomarker."""
        if feature_name not in self.cutoffs:
            raise KeyError(f"Unknown feature '{feature_name}'. Valid: {list(self.cutoffs.keys())}")
        return self.cutoffs[feature_name]

    def to_dict(self) -> dict[str, Any]:
        """Convert calibration to dictionary format for JSON serialization."""
        return {
            "version": "1.0.0",
            "generated_at": self.generated_at,
            "n_samples": self.n_samples,
            "feature_names": list(CANONICAL_FEATURE_NAMES),
            "cutoffs": {k: list(v) for k, v in self.cutoffs.items()},
            "config_hash": self.config_hash,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize calibration to a formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FISCalibration:
        """Construct FISCalibration from a validated dictionary."""
        if not isinstance(data, dict):
            raise ValueError(f"Expected dictionary for calibration, got {type(data).__name__}")
        required_keys = {"generated_at", "n_samples", "cutoffs", "config_hash"}
        missing = required_keys - set(data.keys())
        if missing:
            raise ValueError(f"Calibration dictionary missing required keys: {missing}")

        version = data.get("version", "1.0.0")
        feature_names = data.get("feature_names", list(CANONICAL_FEATURE_NAMES))

        raw_cutoffs = data["cutoffs"]
        if not isinstance(raw_cutoffs, dict):
            raise ValueError("Cutoffs must be a dictionary")

        cutoffs: dict[str, tuple[float, float, float]] = {}
        for feat in CANONICAL_FEATURE_NAMES:
            if feat not in raw_cutoffs:
                raise ValueError(f"Missing cutoffs for feature '{feat}'")
            c = raw_cutoffs[feat]
            if not isinstance(c, (list, tuple)) or len(c) != 3:
                raise ValueError(f"Cutoff values for '{feat}' must be 3 numbers, got {c}")
            cutoffs[feat] = (float(c[0]), float(c[1]), float(c[2]))

        return cls(
            cutoffs=cutoffs,
            n_samples=int(data["n_samples"]),
            generated_at=str(data["generated_at"]),
            config_hash=str(data["config_hash"]),
        )

    @classmethod
    def from_json(cls, json_str: str) -> FISCalibration:
        """Parse FISCalibration from a JSON string."""
        try:
            data = json.loads(json_str)
        except Exception as e:
            raise ValueError(f"Failed to parse calibration JSON: {e}") from e
        return cls.from_dict(data)


@dataclass(frozen=True, slots=True)
class FISPrediction:
    """Explainable Mamdani FIS diagnostic prediction decision artifact."""

    risk_score: float
    risk_category: str
    dominant_rule_id: str
    dominant_rule_text: str
    rule_firings: Mapping[str, float]
    memberships: Mapping[str, Mapping[str, float]]

    def __init__(
        self,
        risk_score: float,
        risk_category: str,
        dominant_rule_id: str,
        dominant_rule_text: str,
        rule_firings: Mapping[str, float],
        memberships: Mapping[str, Mapping[str, float]],
    ) -> None:
        if isinstance(risk_score, bool) or not isinstance(risk_score, (int, float)) or not math.isfinite(risk_score):
            raise ValueError(f"risk_score must be a finite float, got {risk_score!r}")
        bounded_score = float(np.clip(risk_score, 0.0, 100.0))

        if risk_category not in {"Low", "Medium", "High"}:
            raise ValueError(f"risk_category must be 'Low', 'Medium', or 'High', got {risk_category!r}")

        object.__setattr__(self, "risk_score", bounded_score)
        object.__setattr__(self, "risk_category", str(risk_category))
        object.__setattr__(self, "dominant_rule_id", str(dominant_rule_id))
        object.__setattr__(self, "dominant_rule_text", str(dominant_rule_text))
        object.__setattr__(self, "rule_firings", MappingProxyType(dict(rule_firings)))

        frozen_memberships: dict[str, Mapping[str, float]] = {}
        for feat, m in memberships.items():
            frozen_memberships[feat] = MappingProxyType(dict(m))
        object.__setattr__(self, "memberships", MappingProxyType(frozen_memberships))

    def as_dict(self) -> dict[str, Any]:
        """Convert prediction to serializable dictionary."""
        return {
            "risk_score": self.risk_score,
            "risk_category": self.risk_category,
            "dominant_rule_id": self.dominant_rule_id,
            "dominant_rule_text": self.dominant_rule_text,
            "rule_firings": dict(self.rule_firings),
            "memberships": {k: dict(v) for k, v in self.memberships.items()},
        }


# ------------------------------------------------------------------------------
# Calibration Engine (User Story 1)
# ------------------------------------------------------------------------------

def _validate_score(val: Any, name: str) -> float:
    """Validate that a score is a finite float bounded in [0.0, 1.0]."""
    if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
        raise ValueError(f"Biomarker '{name}' must be a finite float, got {val!r}")
    if not (0.0 <= val <= 1.0):
        raise ValueError(f"Biomarker '{name}' must be in [0.0, 1.0], got {val}")
    return float(val)


def _validate_train_matrix(train_matrix: Any) -> np.ndarray:
    """Validate 2D training matrix."""
    if isinstance(train_matrix, (bool, int, float, str)) or train_matrix is None:
        raise ValueError(f"train_matrix must be a 2D array or sequence, got {type(train_matrix).__name__}")

    try:
        arr = np.asarray(train_matrix, dtype=np.float64)
    except Exception as e:
        raise ValueError(f"Cannot convert train_matrix to float array: {e}") from e

    if arr.ndim != 2:
        raise ValueError(f"train_matrix must be a 2D array, got shape {arr.shape}")
    if arr.shape[0] < MIN_CALIBRATION_SAMPLES:
        raise ValueError(f"train_matrix must have at least {MIN_CALIBRATION_SAMPLES} rows, got {arr.shape[0]}")
    if arr.shape[1] != 4:
        raise ValueError(f"train_matrix must have exactly 4 columns for ABCD, got {arr.shape[1]}")

    if not np.all(np.isfinite(arr)):
        raise ValueError("train_matrix contains non-finite values (NaN or Inf)")
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        raise ValueError("train_matrix contains values outside [0.0, 1.0]")

    return arr


def calibrate_fis_percentiles(
    train_matrix: np.ndarray | Sequence[Sequence[float]],
    config: DermaConfig | None = None,
) -> FISCalibration:
    """Calibrate empirical biomarker percentiles from a training dataset split.

    Args:
        train_matrix: 2D array-like of shape (N, 4) with N >= 4.
                      Columns correspond to [A, B_fuzzy, C, D_verified].
        config: Optional DermaConfig instance. Defaults to load_config().

    Returns:
        FISCalibration instance with strictly monotonic, guarded cutoffs.

    Raises:
        ValueError: If train_matrix is not 2D, N < 4, contains non-finite values,
                    or values outside [0.0, 1.0].
    """
    arr = _validate_train_matrix(train_matrix)
    cfg = config if config is not None else load_config()

    percentiles = list(cfg.mamdani_fis.percentiles)
    if percentiles != [25, 50, 75]:
        percentiles = [25, 50, 75]

    delta = DEGENERATE_SEPARATION_DELTA
    cutoffs: dict[str, tuple[float, float, float]] = {}

    for j, feat in enumerate(CANONICAL_FEATURE_NAMES):
        col = arr[:, j]
        raw_p = np.percentile(col, percentiles)
        p25, p50, p75 = float(raw_p[0]), float(raw_p[1]), float(raw_p[2])

        # Degenerate Separation Guard
        # Clamp within [delta, 1.0 - delta]
        p25 = max(delta, p25)
        p50 = max(p25 + delta, p50)
        p75 = max(p50 + delta, p75)

        if p75 > 1.0 - delta:
            p75 = 1.0 - delta
            p50 = min(p50, p75 - delta)
            p25 = min(p25, p50 - delta)

        cutoffs[feat] = (p25, p50, p75)

    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    raw_hash_str = json.dumps(
        {"n": int(arr.shape[0]), "cutoffs": {k: list(v) for k, v in cutoffs.items()}},
        sort_keys=True,
    )
    config_hash = hashlib.sha256(raw_hash_str.encode("utf-8")).hexdigest()

    return FISCalibration(
        cutoffs=cutoffs,
        n_samples=int(arr.shape[0]),
        generated_at=now_iso,
        config_hash=config_hash,
    )


def save_calibration_json(
    calibration: FISCalibration,
    filepath: str | Path = "artifacts/fis_calibration.json",
) -> Path:
    """Serialize calibration parameters to human-readable JSON."""
    if not isinstance(calibration, FISCalibration):
        raise ValueError(f"Expected FISCalibration instance, got {type(calibration).__name__}")

    path = Path(filepath).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    json_text = calibration.to_json(indent=2)
    path.write_text(json_text, encoding="utf-8")
    return path


def load_calibration_json(
    filepath: str | Path = "artifacts/fis_calibration.json",
) -> FISCalibration:
    """Load calibration parameters from human-readable JSON with schema validation."""
    path = Path(filepath).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Calibration file not found at {path}")

    text = path.read_text(encoding="utf-8")
    return FISCalibration.from_json(text)


# ------------------------------------------------------------------------------
# Mamdani Fuzzification & Rule Engine (User Story 2)
# ------------------------------------------------------------------------------

def compute_memberships(
    score: float,
    cutoffs: tuple[float, float, float],
) -> FuzzyMembershipSet:
    """Compute linguistic memberships satisfying Partition of Unity.

    Piecewise-linear continuous formulations:
    - Low (left shoulder trapezoid): 1.0 for x <= p25, linear drop to 0.0 at p50.
    - High (right shoulder trapezoid): 0.0 for x <= p50, linear rise to 1.0 at p75.
    - Medium (triangular center): 1.0 - Low - High, guaranteeing sum == 1.0.
    """
    x = _validate_score(score, "score")
    p25, p50, p75 = cutoffs

    # Low: left-shoulder trapezoid
    if x <= p25:
        low = 1.0
    elif x < p50:
        low = (p50 - x) / (p50 - p25)
    else:
        low = 0.0

    # High: right-shoulder trapezoid
    if x <= p50:
        high = 0.0
    elif x < p75:
        high = (x - p50) / (p75 - p50)
    else:
        high = 1.0

    # Medium: exact Partition of Unity closure
    medium = max(0.0, 1.0 - low - high)

    return FuzzyMembershipSet(low=float(low), medium=float(medium), high=float(high))


def _s_potential(m: FuzzyMembershipSet) -> float:
    """Evaluate smooth monotonic risk index for a single biomarker."""
    return 0.0 * m.low + 50.0 * m.medium + 100.0 * m.high


def _fire_clinical_rules(
    A_m: FuzzyMembershipSet,
    B_m: FuzzyMembershipSet,
    C_m: FuzzyMembershipSet,
    D_m: FuzzyMembershipSet,
) -> dict[str, float]:
    """Evaluate 5 clinical fuzzy inference rules using Zadeh t-norm (min) and s-norm (max)."""
    # Rule 1: IF A is High AND B is High AND C is High THEN Risk is High
    w1 = min(A_m.high, B_m.high, C_m.high)

    # Rule 2: IF B is High AND C is High AND D is High THEN Risk is High
    w2 = min(B_m.high, C_m.high, D_m.high)

    # Rule 3: IF (A is High OR B is High) AND D is High THEN Risk is High
    w3 = min(max(A_m.high, B_m.high), D_m.high)

    # Rule 4: IF B is Medium AND C is Medium AND D is Medium (or moderate features) THEN Risk is Medium
    mean_medium = float(np.mean([A_m.medium, B_m.medium, C_m.medium, D_m.medium]))
    w4 = max(min(B_m.medium, C_m.medium, D_m.medium), mean_medium)

    # Rule 5: IF A is Low AND B is Low AND C is Low AND (D is Low OR D is Medium) THEN Risk is Low
    w5 = min(A_m.low, B_m.low, C_m.low, max(D_m.low, D_m.medium))

    return {
        "R1": float(w1),
        "R2": float(w2),
        "R3": float(w3),
        "R4": float(w4),
        "R5": float(w5),
    }


def _defuzzify_mamdani_centroid(
    firings: dict[str, float],
    m_dict: dict[str, FuzzyMembershipSet],
) -> float:
    """Defuzzify via Mamdani centroid and smooth partition-of-unity risk potential blend."""
    w_high = max(firings["R1"], firings["R2"], firings["R3"])
    w_med = firings["R4"]
    w_low = firings["R5"]

    z = np.linspace(0.0, 100.0, MAMDANI_UNIVERSE_POINTS)
    mf_low = fuzz.trapmf(z, [0.0, 0.0, 15.0, 35.0])
    mf_med = fuzz.trimf(z, [25.0, 50.0, 75.0])
    mf_high = fuzz.trapmf(z, [65.0, 80.0, 100.0, 100.0])

    clip_low = np.fmin(w_low, mf_low)
    clip_med = np.fmin(w_med, mf_med)
    clip_high = np.fmin(w_high, mf_high)
    agg = np.fmax(clip_low, np.fmax(clip_med, clip_high))

    if float(np.sum(agg)) > EPSILON_SUM_GUARD:
        raw_centroid = float(fuzz.defuzz(z, agg, "centroid"))
    else:
        raw_centroid = DEFAULT_FALLBACK_CENTROID

    # Smooth Monotonic Partition-of-Unity Risk Potential (alpha=0.70)
    s_mean = float(np.mean([_s_potential(m) for m in m_dict.values()]))
    s_mapped = PURE_LOW_CENTROID_BOUND + (s_mean / 100.0) * (PURE_HIGH_CENTROID_BOUND - PURE_LOW_CENTROID_BOUND)

    final_risk = (1.0 - CENTROID_POTENTIAL_ALPHA) * raw_centroid + CENTROID_POTENTIAL_ALPHA * s_mapped
    return float(np.clip(final_risk, 0.0, 100.0))


def _determine_dominant_rule(firings: dict[str, float]) -> tuple[str, str]:
    """Resolve dominant clinical rule using fixed tie-break priority order."""
    priority_order = ("R1", "R2", "R3", "R4", "R5")
    dominant_rule_id = "R5"
    max_firing = -1.0

    for r_id in priority_order:
        f_val = firings[r_id]
        if f_val > max_firing + TIE_BREAK_EPSILON:
            max_firing = f_val
            dominant_rule_id = r_id

    rule_lookup = {r.rule_id: r for r in CLINICAL_RULES}
    dom_rule = rule_lookup[dominant_rule_id]
    dominant_rule_text = f"{dom_rule.name}: {dom_rule.description}"
    return dominant_rule_id, dominant_rule_text


def evaluate_mamdani_fis(
    biomarkers: BiomarkerResult | tuple[float, float, float, float] | Sequence[float],
    calibration: FISCalibration,
    config: DermaConfig | None = None,
) -> FISPrediction:
    """Evaluate Mamdani fuzzy inference system with centroid defuzzification.

    Args:
        biomarkers: BiomarkerResult or (A, B_fuzzy, C, D_verified) sequence in [0.0, 1.0]^4.
        calibration: Calibrated percentiles container.
        config: Optional DermaConfig instance. Defaults to load_config().

    Returns:
        Immutable FISPrediction with risk_score, risk_category, dominant_rule, and firings.
    """
    if not isinstance(calibration, FISCalibration):
        raise ValueError(f"calibration must be an instance of FISCalibration, got {type(calibration).__name__}")

    cfg = config if config is not None else load_config()

    # Extract biomarker scores
    if hasattr(biomarkers, "as_vector"):
        vec = biomarkers.as_vector()
    elif (
        isinstance(biomarkers, (list, tuple, np.ndarray))
        and len(biomarkers) == 4
        and getattr(biomarkers, "ndim", 1) == 1
    ):
        vec = tuple(biomarkers)
    else:
        raise ValueError(f"biomarkers must be BiomarkerResult or 4-element sequence, got {biomarkers!r}")

    scores = [_validate_score(v, CANONICAL_FEATURE_NAMES[i]) for i, v in enumerate(vec)]

    # Compute memberships per feature
    m_dict: dict[str, FuzzyMembershipSet] = {
        feat: compute_memberships(s, calibration.get_cutoffs(feat))
        for feat, s in zip(CANONICAL_FEATURE_NAMES, scores)
    }

    firings = _fire_clinical_rules(
        m_dict["A"], m_dict["B_fuzzy"], m_dict["C"], m_dict["D_verified"]
    )
    final_risk = _defuzzify_mamdani_centroid(firings, m_dict)

    # Determine risk category
    low_max = float(cfg.mamdani_fis.risk_thresholds["low_max"])
    medium_max = float(cfg.mamdani_fis.risk_thresholds["medium_max"])

    if final_risk <= low_max:
        risk_category = "Low"
    elif final_risk < medium_max:
        risk_category = "Medium"
    else:
        risk_category = "High"

    dominant_rule_id, dominant_rule_text = _determine_dominant_rule(firings)

    memberships_export: dict[str, dict[str, float]] = {
        feat: m.as_dict() for feat, m in m_dict.items()
    }

    return FISPrediction(
        risk_score=final_risk,
        risk_category=risk_category,
        dominant_rule_id=dominant_rule_id,
        dominant_rule_text=dominant_rule_text,
        rule_firings=firings,
        memberships=memberships_export,
    )


def diagnose_image(
    img_rgb: np.ndarray,
    p_cnn: float,
    heatmap: np.ndarray,
    calibration: FISCalibration,
    config: DermaConfig | None = None,
) -> tuple[PreprocessingResult, BiomarkerResult, FISPrediction]:
    """Execute end-to-end diagnostic pipeline: Room 2 -> Room 3 -> Room 4.

    Args:
        img_rgb: Raw dermoscopic image (uint8, shape (H, W, 3)).
        p_cnn: Raw deep CNN confidence probability in [0.0, 1.0].
        heatmap: 2D Grad-CAM attention heatmap.
        calibration: Calibrated FIS percentiles container.
        config: Optional DermaConfig instance. Defaults to load_config().

    Returns:
        Tuple of (PreprocessingResult, BiomarkerResult, FISPrediction).

    Raises:
        ValueError: If input image, p_cnn, heatmap, or calibration are invalid.
    """
    from dermafuzzy_core.biomarkers import extract_abcd_biomarkers
    from dermafuzzy_core.preprocessing import preprocess_image

    cfg = config if config is not None else load_config()

    prep_result = preprocess_image(img_rgb, cfg)
    bio_result = extract_abcd_biomarkers(
        clean_rgb=prep_result.clean_rgb,
        mask=prep_result.lesion_mask,
        boundary_ring=prep_result.boundary_ring,
        config=cfg,
        p_cnn=p_cnn,
        heatmap=heatmap,
    )
    fis_pred = evaluate_mamdani_fis(
        biomarkers=bio_result,
        calibration=calibration,
        config=cfg,
    )

    return prep_result, bio_result, fis_pred
