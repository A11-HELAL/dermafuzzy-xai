"""Unified Configuration Engine for DermaFuzzy-XAI.

Provides immutable domain dataclasses, YAML loader, and strict mathematical
and domain validation rules. Fully complies with Constitution Principles:
- Principle I: Pure Core & Zero Training-Serving Skew (pure Python standard library + PyYAML).
- Principle II: Strict Mathematical & Domain Bounds (odd kernels, float sums, [0.0, 1.0] range).
- Principle III: Zero Data Leakage & No-Pickle Security Policy (safe YAML loading, zero pickle).
- Principle IV: Config-Driven & Test-First Gate (all hyperparameters loaded from root YAML).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

CANONICAL_FEATURE_NAMES: tuple[str, ...] = ("A", "B_fuzzy", "C", "D_verified")
EPSILON_MAX_BOUND: float = 1e-2


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """Dataset acquisition, pre-filtering, and train/val/test split configuration."""

    name: str
    image_size: tuple[int, int]
    target_classes: Mapping[str, str]
    samples_per_class: int
    random_state: int
    split_ratios: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    """Image enhancement, morphological parameters, and denominator stabilizers."""

    blackhat_kernel_size: int
    inpaint_radius: int
    clahe_clip_limit: float
    clahe_tile_grid: tuple[int, int]
    gaussian_blur_kernel: int
    boundary_ring_kernel: int
    mask_area_bounds: tuple[float, float]
    eps: float


@dataclass(frozen=True, slots=True)
class BiomarkersConfig:
    """Explainable ABCD clinical biomarker weights, names, and value ranges."""

    b_fuzzy_weights: Mapping[str, float]
    feature_names: tuple[str, ...]
    value_range: tuple[float, float]


@dataclass(frozen=True, slots=True)
class MamdaniFISConfig:
    """Mamdani Fuzzy Inference System percentile calibration and defuzzification."""

    percentiles: tuple[float, ...]
    defuzzification_method: str
    risk_thresholds: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class DermaConfig:
    """Root configuration aggregating all sub-configurations."""

    dataset: DatasetConfig
    preprocessing: PreprocessingConfig
    biomarkers: BiomarkersConfig
    mamdani_fis: MamdaniFISConfig


def _validate_int(
    name: str,
    value: Any,
    min_val: int | None = None,
    odd_only: bool = False,
) -> int:
    """Validate that value is a genuine integer (rejecting bool) satisfying domain constraints."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Field '{name}' must be an integer, got {type(value).__name__} ({value})")
    if min_val is not None and value < min_val:
        raise ValueError(f"Field '{name}' must be >= {min_val}, got {value}")
    if odd_only:
        if value <= 0 or value % 2 == 0:
            raise ValueError(f"Kernel '{name}' must be a positive odd integer, got {value}")
    return value


def _validate_finite_float(
    name: str,
    value: Any,
    min_exclusive: float | None = None,
    max_exclusive: float | None = None,
    min_inclusive: float | None = None,
    max_inclusive: float | None = None,
) -> float:
    """Validate that value is a finite float or int (rejecting bool, NaN, and Inf)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Field '{name}' must be a float or int, got {type(value).__name__} ({value})")
    fval = float(value)
    if not math.isfinite(fval):
        raise ValueError(f"Field '{name}' must be a finite float, got {value}")
    if min_exclusive is not None and fval <= min_exclusive:
        raise ValueError(f"Field '{name}' must be > {min_exclusive}, got {fval}")
    if max_exclusive is not None and fval >= max_exclusive:
        raise ValueError(f"Field '{name}' must be < {max_exclusive}, got {fval}")
    if min_inclusive is not None and fval < min_inclusive:
        raise ValueError(f"Field '{name}' must be >= {min_inclusive}, got {fval}")
    if max_inclusive is not None and fval > max_inclusive:
        raise ValueError(f"Field '{name}' must be <= {max_inclusive}, got {fval}")
    return fval


def validate_config(raw_dict: dict[str, Any]) -> DermaConfig:
    """Validate a raw configuration dictionary and construct typed domain dataclasses.

    Parameters
    ----------
    raw_dict : dict[str, Any]
        Dictionary representation of configuration sections.

    Returns
    -------
    DermaConfig
        Validated, deeply-immutable DermaConfig instance.

    Raises
    ------
    ValueError
        If required sections are missing or if any domain validation rule is violated.
    """
    if not isinstance(raw_dict, dict):
        raise ValueError(f"Configuration must be a dictionary, got {type(raw_dict).__name__}")

    required_sections = {"dataset", "preprocessing", "biomarkers", "mamdani_fis"}
    missing_sections = required_sections - raw_dict.keys()
    if missing_sections:
        raise ValueError(f"Missing required configuration sections: {sorted(missing_sections)}")

    # --------------------------------------------------------------------------
    # 1. Dataset Validation
    # --------------------------------------------------------------------------
    ds = raw_dict["dataset"]
    if not isinstance(ds, dict):
        raise ValueError("Section 'dataset' must be a dictionary")

    ds_name = str(ds.get("name", "")).strip()
    if not ds_name:
        raise ValueError("dataset.name cannot be empty")

    raw_image_size = ds.get("image_size")
    if not (isinstance(raw_image_size, (list, tuple)) and len(raw_image_size) == 2):
        raise ValueError("dataset.image_size must be a 2-element list or tuple of integers [width, height]")
    w = _validate_int("dataset.image_size[0]", raw_image_size[0], min_val=1)
    h = _validate_int("dataset.image_size[1]", raw_image_size[1], min_val=1)
    image_size = (w, h)

    target_classes = ds.get("target_classes")
    if not isinstance(target_classes, dict) or set(target_classes.keys()) != {"malignant", "benign"}:
        raise ValueError("dataset.target_classes must contain exactly {'malignant', 'benign'} keys")
    val_m = target_classes["malignant"]
    val_b = target_classes["benign"]
    if not isinstance(val_m, str) or not isinstance(val_b, str):
        raise ValueError("dataset.target_classes values must be strings")
    clean_val_m = val_m.strip()
    clean_val_b = val_b.strip()
    if not clean_val_m or not clean_val_b:
        raise ValueError("dataset.target_classes values cannot be empty strings")
    if clean_val_m == clean_val_b:
        raise ValueError(
            f"dataset.target_classes values must be distinct, got malignant='{clean_val_m}' and benign='{clean_val_b}'"
        )

    samples_per_class = _validate_int("dataset.samples_per_class", ds.get("samples_per_class"), min_val=1)
    random_state = _validate_int("dataset.random_state", ds.get("random_state", 42))

    split_ratios = ds.get("split_ratios")
    if not isinstance(split_ratios, dict) or set(split_ratios.keys()) != {"train", "val", "test"}:
        raise ValueError("dataset.split_ratios must contain exactly {'train', 'val', 'test'} keys")

    validated_splits: dict[str, float] = {}
    for split_key in ("train", "val", "test"):
        val = _validate_finite_float(
            f"dataset.split_ratios['{split_key}']",
            split_ratios[split_key],
            min_exclusive=0.0,
            max_exclusive=1.0,
        )
        validated_splits[split_key] = val

    sum_splits = sum(validated_splits.values())
    if not math.isclose(sum_splits, 1.0, abs_tol=1e-6):
        raise ValueError(f"dataset.split_ratios must sum to 1.0 (+/- 1e-6), got {sum_splits:.6f}")

    dataset_config = DatasetConfig(
        name=ds_name,
        image_size=image_size,
        target_classes=MappingProxyType({"malignant": clean_val_m, "benign": clean_val_b}),
        samples_per_class=samples_per_class,
        random_state=random_state,
        split_ratios=MappingProxyType(validated_splits),
    )

    # --------------------------------------------------------------------------
    # 2. Preprocessing Validation
    # --------------------------------------------------------------------------
    prep = raw_dict["preprocessing"]
    if not isinstance(prep, dict):
        raise ValueError("Section 'preprocessing' must be a dictionary")

    blackhat_k = _validate_int("blackhat_kernel_size", prep.get("blackhat_kernel_size"), odd_only=True)
    inpaint_r = _validate_int("preprocessing.inpaint_radius", prep.get("inpaint_radius"), min_val=1)
    clahe_clip = _validate_finite_float(
        "preprocessing.clahe_clip_limit", prep.get("clahe_clip_limit", 2.0), min_exclusive=0.0
    )

    raw_tile_grid = prep.get("clahe_tile_grid")
    if not (isinstance(raw_tile_grid, (list, tuple)) and len(raw_tile_grid) == 2):
        raise ValueError("preprocessing.clahe_tile_grid must be a 2-element tuple/list of integers [w, h]")
    gw = _validate_int("preprocessing.clahe_tile_grid[0]", raw_tile_grid[0], min_val=1)
    gh = _validate_int("preprocessing.clahe_tile_grid[1]", raw_tile_grid[1], min_val=1)
    clahe_tile_grid = (gw, gh)

    gauss_k = _validate_int("gaussian_blur_kernel", prep.get("gaussian_blur_kernel"), odd_only=True)
    boundary_k = _validate_int("boundary_ring_kernel", prep.get("boundary_ring_kernel"), odd_only=True)

    raw_area_bounds = prep.get("mask_area_bounds")
    if not (isinstance(raw_area_bounds, (list, tuple)) and len(raw_area_bounds) == 2):
        raise ValueError("preprocessing.mask_area_bounds must be a 2-element tuple/list [min_area, max_area]")
    min_area = _validate_finite_float(
        "preprocessing.mask_area_bounds[0]", raw_area_bounds[0], min_exclusive=0.0, max_exclusive=1.0
    )
    max_area = _validate_finite_float(
        "preprocessing.mask_area_bounds[1]", raw_area_bounds[1], min_exclusive=0.0, max_exclusive=1.0
    )
    if not (min_area < max_area):
        raise ValueError(
            f"preprocessing.mask_area_bounds must satisfy 0.0 < min < max < 1.0, got [{min_area}, {max_area}]"
        )
    mask_area_bounds = (min_area, max_area)

    eps = _validate_finite_float(
        "preprocessing.eps",
        prep.get("eps", 1e-6),
        min_exclusive=0.0,
        max_inclusive=EPSILON_MAX_BOUND,
    )

    preprocessing_config = PreprocessingConfig(
        blackhat_kernel_size=blackhat_k,
        inpaint_radius=inpaint_r,
        clahe_clip_limit=clahe_clip,
        clahe_tile_grid=clahe_tile_grid,
        gaussian_blur_kernel=gauss_k,
        boundary_ring_kernel=boundary_k,
        mask_area_bounds=mask_area_bounds,
        eps=eps,
    )

    # --------------------------------------------------------------------------
    # 3. Biomarkers Validation
    # --------------------------------------------------------------------------
    bio = raw_dict["biomarkers"]
    if not isinstance(bio, dict):
        raise ValueError("Section 'biomarkers' must be a dictionary")

    raw_weights = bio.get("b_fuzzy_weights")
    if not isinstance(raw_weights, dict) or set(raw_weights.keys()) != {"w_geo", "w_grad"}:
        raise ValueError("biomarkers.b_fuzzy_weights must contain exactly {'w_geo', 'w_grad'} keys")
    w_geo = _validate_finite_float("biomarkers.b_fuzzy_weights['w_geo']", raw_weights["w_geo"], min_exclusive=0.0, max_exclusive=1.0)
    w_grad = _validate_finite_float("biomarkers.b_fuzzy_weights['w_grad']", raw_weights["w_grad"], min_exclusive=0.0, max_exclusive=1.0)
    sum_weights = w_geo + w_grad
    if not math.isclose(sum_weights, 1.0, abs_tol=1e-6):
        raise ValueError(f"biomarkers.b_fuzzy_weights must sum to 1.0 (+/- 1e-6), got {sum_weights:.6f}")

    raw_features = bio.get("feature_names")
    if not (isinstance(raw_features, (list, tuple)) and tuple(raw_features) == CANONICAL_FEATURE_NAMES):
        raise ValueError(
            f"biomarkers.feature_names must strictly match {CANONICAL_FEATURE_NAMES} in exact canonical order, got {raw_features}"
        )
    feature_names = CANONICAL_FEATURE_NAMES

    raw_val_range = bio.get("value_range")
    if not (isinstance(raw_val_range, (list, tuple)) and len(raw_val_range) == 2):
        raise ValueError("biomarkers.value_range must be [0.0, 1.0]")
    vr_min = _validate_finite_float("biomarkers.value_range[0]", raw_val_range[0])
    vr_max = _validate_finite_float("biomarkers.value_range[1]", raw_val_range[1])
    if vr_min != 0.0 or vr_max != 1.0:
        raise ValueError(f"biomarkers.value_range must strictly be [0.0, 1.0] per Constitution, got [{vr_min}, {vr_max}]")
    value_range = (0.0, 1.0)

    biomarkers_config = BiomarkersConfig(
        b_fuzzy_weights=MappingProxyType({"w_geo": w_geo, "w_grad": w_grad}),
        feature_names=feature_names,
        value_range=value_range,
    )

    # --------------------------------------------------------------------------
    # 4. Mamdani FIS Validation
    # --------------------------------------------------------------------------
    fis = raw_dict["mamdani_fis"]
    if not isinstance(fis, dict):
        raise ValueError("Section 'mamdani_fis' must be a dictionary")

    raw_percentiles = fis.get("percentiles")
    if not (isinstance(raw_percentiles, (list, tuple)) and len(raw_percentiles) == 3):
        raise ValueError("mamdani_fis.percentiles must contain exactly 3 calibration percentiles [P25, P50, P75]")

    p0 = _validate_finite_float("mamdani_fis.percentiles[0]", raw_percentiles[0], min_exclusive=0.0, max_exclusive=100.0)
    p1 = _validate_finite_float("mamdani_fis.percentiles[1]", raw_percentiles[1], min_exclusive=0.0, max_exclusive=100.0)
    p2 = _validate_finite_float("mamdani_fis.percentiles[2]", raw_percentiles[2], min_exclusive=0.0, max_exclusive=100.0)

    if not (p0 < p1 < p2):
        raise ValueError(
            f"mamdani_fis.percentiles must be strictly monotonically increasing, got [{p0}, {p1}, {p2}]"
        )
    percentiles = (p0, p1, p2)

    defuzz = str(fis.get("defuzzification_method", "centroid")).strip()
    if defuzz not in {"centroid", "bisector", "mom"}:
        raise ValueError(f"mamdani_fis.defuzzification_method '{defuzz}' not supported; choose from centroid, bisector, mom")

    raw_risk = fis.get("risk_thresholds")
    if not isinstance(raw_risk, dict) or set(raw_risk.keys()) != {"low_max", "medium_max"}:
        raise ValueError("mamdani_fis.risk_thresholds must contain exactly {'low_max', 'medium_max'} keys")
    low_max = _validate_finite_float(
        "mamdani_fis.risk_thresholds['low_max']", raw_risk["low_max"], min_exclusive=0.0, max_exclusive=100.0
    )
    med_max = _validate_finite_float(
        "mamdani_fis.risk_thresholds['medium_max']", raw_risk["medium_max"], min_exclusive=0.0, max_exclusive=100.0
    )
    if not (low_max < med_max):
        raise ValueError(
            f"mamdani_fis.risk_thresholds must satisfy 0 < low_max < medium_max < 100, got [{low_max}, {med_max}]"
        )
    risk_thresholds = {"low_max": low_max, "medium_max": med_max}

    mamdani_config = MamdaniFISConfig(
        percentiles=percentiles,
        defuzzification_method=defuzz,
        risk_thresholds=MappingProxyType(risk_thresholds),
    )

    return DermaConfig(
        dataset=dataset_config,
        preprocessing=preprocessing_config,
        biomarkers=biomarkers_config,
        mamdani_fis=mamdani_config,
    )


def load_config(path: str | Path | None = None) -> DermaConfig:
    """Load, parse, and validate the DermaFuzzy-XAI configuration from disk.

    Parameters
    ----------
    path : str | Path | None, optional
        Path to the YAML configuration file. If None, defaults to 'derma_config.yaml'
        at the current working directory or repository root.

    Returns
    -------
    DermaConfig
        An immutable, strongly-typed domain configuration instance.

    Raises
    ------
    FileNotFoundError
        If the specified configuration file does not exist.
    ValueError
        If the file is empty, malformed, or violates domain validation rules.
    """
    if path is None:
        candidate_paths = [
            Path.cwd() / "derma_config.yaml",
            Path(__file__).resolve().parents[2] / "derma_config.yaml",
        ]
        resolved_path: Path | None = None
        for candidate in candidate_paths:
            if candidate.is_file():
                resolved_path = candidate
                break
        if resolved_path is None:
            raise FileNotFoundError("Default configuration file 'derma_config.yaml' not found in workspace root.")
        path_obj = resolved_path
    else:
        path_obj = Path(path)
        if not path_obj.is_file():
            raise FileNotFoundError(f"Configuration file not found: {path_obj}")

    try:
        with open(path_obj, "r", encoding="utf-8") as f:
            raw_dict = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ValueError(f"Malformed YAML in configuration file '{path_obj}': {exc}") from exc

    if raw_dict is None:
        raise ValueError(f"Configuration file at '{path_obj}' is empty")

    return validate_config(raw_dict)
