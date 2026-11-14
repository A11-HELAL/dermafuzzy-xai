# API Contract: `dermafuzzy_core.config`

**Feature**: [001-walking-skeleton-config](../spec.md) | **Date**: 2026-11-15

## Module: `packages/dermafuzzy_core/config.py`

This module provides the pure Python configuration loading, validation, and domain modeling interface for DermaFuzzy-XAI.

### 1. Function `load_config`

```python
def load_config(path: str | Path | None = None) -> DermaConfig:
    """Load, parse, and validate the DermaFuzzy-XAI configuration from disk.

    Parameters
    ----------
    path : str | Path | None, optional
        Path to the YAML configuration file. If None, defaults to 'derma_config.yaml'
        located at the repository root or current working directory.

    Returns
    -------
    DermaConfig
        An immutable, strongly-typed domain configuration instance.

    Raises
    ------
    FileNotFoundError
        If the specified configuration file does not exist.
    ValueError
        If the configuration file is empty, malformed (yaml.YAMLError), or violates any domain
        validation rule (e.g. invalid kernel sizes, split ratio sums, weights, percentiles).
    """
```

### 2. Function `validate_config`

```python
def validate_config(raw_dict: dict[str, Any]) -> DermaConfig:
    """Validate a raw configuration dictionary and construct typed domain dataclasses.

    Parameters
    ----------
    raw_dict : dict[str, Any]
        Dictionary representation of configuration sections.

    Returns
    -------
    DermaConfig
        Validated, deeply-immutable DermaConfig instance with nested MappingProxyType maps.

    Raises
    ------
    ValueError
        - If required top-level sections ('dataset', 'preprocessing', 'biomarkers', 'mamdani_fis') are missing.
        - If boolean values are passed into integer or float fields.
        - If non-finite float values (NaN, Inf) are encountered.
        - If 'dataset.target_classes' keys != {'malignant', 'benign'} or values are not distinct non-empty strings.
        - If 'split_ratios' keys != {'train', 'val', 'test'}, or individual ratios are not in (0.0, 1.0), or do not sum to 1.0 +/- 1e-6.
        - If 'b_fuzzy_weights' keys != {'w_geo', 'w_grad'}, or weights are not in (0.0, 1.0), or do not sum to 1.0 +/- 1e-6.
        - If 'biomarkers.feature_names' does not strictly match ('A', 'B_fuzzy', 'C', 'D_verified') in exact order.
        - If any of 'blackhat_kernel_size', 'gaussian_blur_kernel', 'boundary_ring_kernel'
          are not positive odd integers (k > 0 and k % 2 == 1).
        - If 'preprocessing.eps' is not in (0.0, 1e-2].
        - If 'mamdani_fis.percentiles' does not contain exactly 3 strictly increasing values in (0, 100).
        - If 'mamdani_fis.risk_thresholds' keys != {'low_max', 'medium_max'} or 0 < low_max < med_max < 100 is violated.
        - If 'mask_area_bounds' are not ordered (0.0 < min_bound < max_bound < 1.0).
        - If 'value_range' does not equal [0.0, 1.0].
    """
```

### 3. Domain Dataclasses

- `DatasetConfig` (with `target_classes: Mapping[str, str]`, `split_ratios: Mapping[str, float]`)
- `PreprocessingConfig`
- `BiomarkersConfig` (with `b_fuzzy_weights: Mapping[str, float]`)
- `MamdaniFISConfig` (with `risk_thresholds: Mapping[str, float]`)
- `DermaConfig`

All dataclasses are decorated with `@dataclass(frozen=True, slots=True)` and wrap dictionary fields in `types.MappingProxyType` to guarantee both shallow and deep runtime immutability (mutations raise `FrozenInstanceError` or `TypeError`).
