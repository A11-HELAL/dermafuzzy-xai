"""Unit tests for dermafuzzy_core.config.

Covers:
- US1: Unified configuration loading, field fidelity, shallow & deep immutability (MappingProxyType).
- US2: Strict mathematical bounds, kernel parity, float finiteness (no bool/NaN/Inf),
       individual split bounds, exact keys, exact domain invariants (target_classes, feature_names, percentiles).
- US3: Architectural boundary verification (zero top-level heavy imports).
"""

from __future__ import annotations

import copy
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest
import yaml

from dermafuzzy_core.config import (
    CANONICAL_FEATURE_NAMES,
    BiomarkersConfig,
    DatasetConfig,
    DermaConfig,
    MamdaniFISConfig,
    PreprocessingConfig,
    load_config,
    validate_config,
)


@pytest.fixture
def valid_config_dict() -> dict[str, Any]:
    """Return a pristine dictionary representing valid configuration."""
    return {
        "dataset": {
            "name": "HAM10000",
            "image_size": [224, 224],
            "target_classes": {"malignant": "mel", "benign": "nv"},
            "samples_per_class": 1000,
            "random_state": 42,
            "split_ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
        },
        "preprocessing": {
            "blackhat_kernel_size": 17,
            "inpaint_radius": 3,
            "clahe_clip_limit": 2.0,
            "clahe_tile_grid": [8, 8],
            "gaussian_blur_kernel": 5,
            "boundary_ring_kernel": 5,
            "mask_area_bounds": [0.02, 0.90],
            "eps": 1.0e-6,
        },
        "biomarkers": {
            "b_fuzzy_weights": {"w_geo": 0.5, "w_grad": 0.5},
            "feature_names": ["A", "B_fuzzy", "C", "D_verified"],
            "value_range": [0.0, 1.0],
        },
        "mamdani_fis": {
            "percentiles": [25, 50, 75],
            "defuzzification_method": "centroid",
            "risk_thresholds": {"low_max": 35.0, "medium_max": 65.0},
        },
    }


# ------------------------------------------------------------------------------
# User Story 1: Unified Configuration Loading & Deep Immutability
# ------------------------------------------------------------------------------

def test_load_config_valid() -> None:
    """Verify that default derma_config.yaml loads with exact field values and types."""
    cfg = load_config()

    assert isinstance(cfg, DermaConfig)

    # Dataset assertions
    assert isinstance(cfg.dataset, DatasetConfig)
    assert cfg.dataset.name == "HAM10000"
    assert cfg.dataset.image_size == (224, 224)
    assert isinstance(cfg.dataset.target_classes, MappingProxyType)
    assert dict(cfg.dataset.target_classes) == {"malignant": "mel", "benign": "nv"}
    assert cfg.dataset.samples_per_class == 1000
    assert cfg.dataset.random_state == 42
    assert isinstance(cfg.dataset.split_ratios, MappingProxyType)
    assert dict(cfg.dataset.split_ratios) == {"train": 0.70, "val": 0.15, "test": 0.15}

    # Preprocessing assertions
    assert isinstance(cfg.preprocessing, PreprocessingConfig)
    assert cfg.preprocessing.blackhat_kernel_size == 17
    assert cfg.preprocessing.inpaint_radius == 3
    assert cfg.preprocessing.clahe_clip_limit == 2.0
    assert cfg.preprocessing.clahe_tile_grid == (8, 8)
    assert cfg.preprocessing.gaussian_blur_kernel == 5
    assert cfg.preprocessing.boundary_ring_kernel == 5
    assert cfg.preprocessing.mask_area_bounds == (0.02, 0.90)
    assert cfg.preprocessing.eps == 1.0e-6

    # Biomarkers assertions
    assert isinstance(cfg.biomarkers, BiomarkersConfig)
    assert isinstance(cfg.biomarkers.b_fuzzy_weights, MappingProxyType)
    assert dict(cfg.biomarkers.b_fuzzy_weights) == {"w_geo": 0.5, "w_grad": 0.5}
    assert cfg.biomarkers.feature_names == CANONICAL_FEATURE_NAMES
    assert cfg.biomarkers.value_range == (0.0, 1.0)

    # Mamdani FIS assertions
    assert isinstance(cfg.mamdani_fis, MamdaniFISConfig)
    assert cfg.mamdani_fis.percentiles == (25.0, 50.0, 75.0)
    assert cfg.mamdani_fis.defuzzification_method == "centroid"
    assert isinstance(cfg.mamdani_fis.risk_thresholds, MappingProxyType)
    assert dict(cfg.mamdani_fis.risk_thresholds) == {"low_max": 35.0, "medium_max": 65.0}


def test_load_config_missing_file() -> None:
    """Verify FileNotFoundError is raised when config file does not exist."""
    with pytest.raises(FileNotFoundError, match="not found"):
        load_config("nonexistent_path_to_config.yaml")


def test_load_config_custom_file(tmp_path: Path, valid_config_dict: dict[str, Any]) -> None:
    """Verify load_config loads cleanly from a custom path."""
    config_file = tmp_path / "custom_config.yaml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(valid_config_dict, f)

    cfg = load_config(config_file)
    assert cfg.dataset.name == "HAM10000"


def test_load_config_malformed_yaml(tmp_path: Path) -> None:
    """Verify malformed YAML raises ValueError with clear diagnostic message."""
    bad_yaml_file = tmp_path / "corrupt.yaml"
    bad_yaml_file.write_text("dataset:\n  name: [unclosed list\npreprocessing: : invalid", encoding="utf-8")

    with pytest.raises(ValueError, match="Malformed YAML"):
        load_config(bad_yaml_file)


def test_config_shallow_immutability() -> None:
    """Verify dataclass fields cannot be reassigned at runtime (FrozenInstanceError)."""
    cfg = load_config()

    with pytest.raises(FrozenInstanceError):
        cfg.dataset.name = "ModifiedName"  # type: ignore[misc]

    with pytest.raises(FrozenInstanceError):
        cfg.preprocessing.blackhat_kernel_size = 21  # type: ignore[misc]

    with pytest.raises(FrozenInstanceError):
        cfg.biomarkers.value_range = (0.0, 2.0)  # type: ignore[misc]


def test_config_deep_immutability_mapping_proxy() -> None:
    """Verify nested mappings are wrapped in MappingProxyType and mutate attempts raise TypeError."""
    cfg = load_config()

    with pytest.raises(TypeError):
        cfg.dataset.target_classes["malignant"] = "mel_new"  # type: ignore[index]

    with pytest.raises(TypeError):
        cfg.dataset.target_classes["new_class"] = "foo"  # type: ignore[index]

    with pytest.raises(TypeError):
        cfg.dataset.split_ratios["train"] = 0.50  # type: ignore[index]

    with pytest.raises(TypeError):
        cfg.biomarkers.b_fuzzy_weights["w_geo"] = 0.80  # type: ignore[index]

    with pytest.raises(TypeError):
        cfg.mamdani_fis.risk_thresholds["low_max"] = 40.0  # type: ignore[index]


# ------------------------------------------------------------------------------
# User Story 2: Strict Mathematical, Type, & Domain Bounds
# ------------------------------------------------------------------------------

def test_rejects_missing_sections() -> None:
    """Verify validation fails if top-level sections are missing."""
    with pytest.raises(ValueError, match="Missing required configuration sections"):
        validate_config({"dataset": {}})


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("dataset", "samples_per_class"), True),
        (("dataset", "random_state"), True),
        (("dataset", "image_size"), [True, 224]),
        (("dataset", "image_size"), [224, False]),
        (("preprocessing", "blackhat_kernel_size"), True),
        (("preprocessing", "inpaint_radius"), True),
        (("preprocessing", "clahe_tile_grid"), [True, 8]),
        (("preprocessing", "gaussian_blur_kernel"), True),
        (("preprocessing", "boundary_ring_kernel"), True),
    ],
)
def test_rejects_bool_in_integer_fields(
    valid_config_dict: dict[str, Any], path: tuple[str, ...], value: Any
) -> None:
    """Verify boolean values are rejected across all integer fields."""
    bad_dict = copy.deepcopy(valid_config_dict)
    curr = bad_dict
    for key in path[:-1]:
        curr = curr[key]
    curr[path[-1]] = value

    with pytest.raises(ValueError, match="must be an integer"):
        validate_config(bad_dict)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("preprocessing", "clahe_clip_limit"), True),
        (("preprocessing", "eps"), True),
        (("preprocessing", "eps"), 0.05),  # eps > 1e-2 bound
        (("preprocessing", "eps"), -1e-6),  # eps <= 0
        (("preprocessing", "clahe_clip_limit"), float("nan")),
        (("preprocessing", "clahe_clip_limit"), float("inf")),
        (("preprocessing", "eps"), float("nan")),
        (("preprocessing", "eps"), float("inf")),
        (("dataset", "split_ratios", "train"), True),
        (("dataset", "split_ratios", "train"), float("nan")),
        (("dataset", "split_ratios", "train"), float("inf")),
        (("biomarkers", "b_fuzzy_weights", "w_geo"), True),
        (("biomarkers", "b_fuzzy_weights", "w_geo"), float("nan")),
        (("mamdani_fis", "risk_thresholds", "low_max"), float("nan")),
        (("mamdani_fis", "risk_thresholds", "low_max"), float("inf")),
    ],
)
def test_rejects_bool_nan_inf_in_float_fields(
    valid_config_dict: dict[str, Any], path: tuple[str, ...], value: Any
) -> None:
    """Verify bool, NaN, and Inf values are strictly rejected across all float fields."""
    bad_dict = copy.deepcopy(valid_config_dict)
    curr = bad_dict
    for key in path[:-1]:
        curr = curr[key]
    curr[path[-1]] = value

    with pytest.raises(ValueError):
        validate_config(bad_dict)


@pytest.mark.parametrize("kernel_field", ["blackhat_kernel_size", "gaussian_blur_kernel", "boundary_ring_kernel"])
def test_rejects_even_kernel_sizes(valid_config_dict: dict[str, Any], kernel_field: str) -> None:
    """Verify even kernel diameters are strictly rejected with ValueError."""
    bad_dict = copy.deepcopy(valid_config_dict)
    bad_dict["preprocessing"][kernel_field] = 16

    with pytest.raises(ValueError, match="must be a positive odd integer"):
        validate_config(bad_dict)


@pytest.mark.parametrize("kernel_field", ["blackhat_kernel_size", "gaussian_blur_kernel", "boundary_ring_kernel"])
def test_rejects_non_positive_kernel_sizes(valid_config_dict: dict[str, Any], kernel_field: str) -> None:
    """Verify zero or negative kernels are strictly rejected with ValueError."""
    bad_dict = copy.deepcopy(valid_config_dict)
    bad_dict["preprocessing"][kernel_field] = -3

    with pytest.raises(ValueError, match="must be a positive odd integer"):
        validate_config(bad_dict)


def test_rejects_invalid_split_ratios(valid_config_dict: dict[str, Any]) -> None:
    """Verify dataset split ratios that do not sum to 1.0 are rejected."""
    bad_dict = copy.deepcopy(valid_config_dict)
    bad_dict["dataset"]["split_ratios"] = {"train": 0.80, "val": 0.15, "test": 0.15}  # sum = 1.10

    with pytest.raises(ValueError, match="split_ratios must sum to 1.0"):
        validate_config(bad_dict)


def test_rejects_negative_split_ratios_summing_to_one(valid_config_dict: dict[str, Any]) -> None:
    """Verify negative split ratios summing to 1.0 are rejected (e.g. 1.2, -0.1, -0.1)."""
    bad_dict = copy.deepcopy(valid_config_dict)
    bad_dict["dataset"]["split_ratios"] = {"train": 1.20, "val": -0.10, "test": -0.10}

    with pytest.raises(ValueError, match="must be < 1.0"):
        validate_config(bad_dict)

    bad_dict["dataset"]["split_ratios"] = {"train": 0.90, "val": 0.20, "test": -0.10}
    with pytest.raises(ValueError, match="must be > 0.0"):
        validate_config(bad_dict)


def test_rejects_split_ratios_keys_mismatch(valid_config_dict: dict[str, Any]) -> None:
    """Verify extra or missing keys in split_ratios are rejected."""
    bad_dict = copy.deepcopy(valid_config_dict)
    bad_dict["dataset"]["split_ratios"] = {"train": 0.70, "val": 0.15, "test": 0.15, "holdout": 0.0}

    with pytest.raises(ValueError, match="must contain exactly"):
        validate_config(bad_dict)


def test_rejects_invalid_b_fuzzy_weights(valid_config_dict: dict[str, Any]) -> None:
    """Verify b_fuzzy_weights that do not sum to 1.0 are rejected."""
    bad_dict = copy.deepcopy(valid_config_dict)
    bad_dict["biomarkers"]["b_fuzzy_weights"] = {"w_geo": 0.6, "w_grad": 0.6}  # sum = 1.20

    with pytest.raises(ValueError, match="b_fuzzy_weights must sum to 1.0"):
        validate_config(bad_dict)


def test_rejects_b_fuzzy_weights_keys_mismatch(valid_config_dict: dict[str, Any]) -> None:
    """Verify extra or missing keys in b_fuzzy_weights are rejected."""
    bad_dict = copy.deepcopy(valid_config_dict)
    bad_dict["biomarkers"]["b_fuzzy_weights"] = {"w_geo": 0.5, "w_grad": 0.5, "extra": 0.0}

    with pytest.raises(ValueError, match="must contain exactly"):
        validate_config(bad_dict)


def test_rejects_b_fuzzy_weights_boundary_values(valid_config_dict: dict[str, Any]) -> None:
    """Verify 0.0 or 1.0 weights are rejected since weights must be 0.0 < w < 1.0."""
    bad_dict = copy.deepcopy(valid_config_dict)
    bad_dict["biomarkers"]["b_fuzzy_weights"] = {"w_geo": 0.0, "w_grad": 1.0}

    with pytest.raises(ValueError, match="must be > 0.0"):
        validate_config(bad_dict)


def test_rejects_target_classes_invalid_keys(valid_config_dict: dict[str, Any]) -> None:
    """Verify target_classes must have exact {'malignant', 'benign'} keys."""
    bad_dict = copy.deepcopy(valid_config_dict)
    bad_dict["dataset"]["target_classes"] = {"malignant": "mel", "normal": "nv"}

    with pytest.raises(ValueError, match="must contain exactly"):
        validate_config(bad_dict)


def test_rejects_target_classes_empty_or_identical(valid_config_dict: dict[str, Any]) -> None:
    """Verify target_classes cannot have empty strings or identical values."""
    bad_dict = copy.deepcopy(valid_config_dict)
    bad_dict["dataset"]["target_classes"] = {"malignant": "mel", "benign": ""}

    with pytest.raises(ValueError, match="cannot be empty"):
        validate_config(bad_dict)

    bad_dict["dataset"]["target_classes"] = {"malignant": "mel", "benign": "mel"}
    with pytest.raises(ValueError, match="must be distinct"):
        validate_config(bad_dict)


def test_rejects_feature_names_order_or_values(valid_config_dict: dict[str, Any]) -> None:
    """Verify feature_names must strictly match CANONICAL_FEATURE_NAMES in exact order."""
    bad_dict = copy.deepcopy(valid_config_dict)
    bad_dict["biomarkers"]["feature_names"] = ["B_fuzzy", "A", "C", "D_verified"]  # wrong order

    with pytest.raises(ValueError, match="must strictly match"):
        validate_config(bad_dict)


@pytest.mark.parametrize(
    "bad_percentiles",
    [
        [50, 25, 75],       # Not monotonic
        [25, 25, 75],       # Duplicate
        [0, 50, 75],        # Boundary <= 0
        [25, 50, 100],      # Boundary >= 100
        [25, 50],           # len != 3
        [20, 40, 60, 80],   # len != 3
        [25, float("nan"), 75],  # NaN
    ],
)
def test_rejects_disordered_or_invalid_percentiles(
    valid_config_dict: dict[str, Any], bad_percentiles: list[float]
) -> None:
    """Verify FIS calibration percentiles must be exactly 3 strictly increasing values in (0, 100)."""
    bad_dict = copy.deepcopy(valid_config_dict)
    bad_dict["mamdani_fis"]["percentiles"] = bad_percentiles

    with pytest.raises(ValueError):
        validate_config(bad_dict)


def test_rejects_invalid_mask_area_bounds(valid_config_dict: dict[str, Any]) -> None:
    """Verify inverted or out-of-bounds mask area boundaries are rejected."""
    bad_dict = copy.deepcopy(valid_config_dict)
    bad_dict["preprocessing"]["mask_area_bounds"] = [0.90, 0.02]  # min > max

    with pytest.raises(ValueError, match="mask_area_bounds"):
        validate_config(bad_dict)


def test_rejects_invalid_value_range(valid_config_dict: dict[str, Any]) -> None:
    """Verify biomarker value range must strictly be [0.0, 1.0] per Constitution."""
    bad_dict = copy.deepcopy(valid_config_dict)
    bad_dict["biomarkers"]["value_range"] = [0.0, 2.0]

    with pytest.raises(ValueError, match="value_range must strictly be"):
        validate_config(bad_dict)


# ------------------------------------------------------------------------------
# User Story 3: Walking Skeleton Architecture & Boundary Verification
# ------------------------------------------------------------------------------

def test_zero_heavy_imports() -> None:
    """Verify importing dermafuzzy_core does NOT load heavy ML/DL frameworks per Constitution Principle I."""
    code = (
        "import sys\n"
        "import dermafuzzy_core\n"
        "import dermafuzzy_core.config\n"
        "prohibited = ['tensorflow', 'keras', 'torch', 'torchvision', 'scipy']\n"
        "loaded = [fw for fw in prohibited if fw in sys.modules]\n"
        "if loaded:\n"
        "    sys.exit(f'Loaded prohibited frameworks: {loaded}')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Constitution violation in dermafuzzy_core: {result.stderr or result.stdout}"
