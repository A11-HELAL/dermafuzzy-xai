"""DermaFuzzy-XAI Core Algorithmic Library.

This package contains 100% pure, deterministic Python functions for:
- Configuration parsing, typing, and validation (`dermafuzzy_core.config`).
- OpenCV preprocessing, hair removal, and perilesional boundary rings (`dermafuzzy_core.preprocessing`).
- Explainable ABCD clinical biomarker calculations (`dermafuzzy_core.biomarkers`).
- Saliency-verified Grad-CAM attention gating (`dermafuzzy_core.gradcam`).

Governed by Constitution Principle I: Zero top-level heavy framework imports (TensorFlow, Keras, PyTorch).
"""

from dermafuzzy_core.biomarkers import (
    BiomarkerResult,
    compute_asymmetry,
    compute_border_fuzziness,
    compute_color_variation,
    extract_abcd_biomarkers,
)
from dermafuzzy_core.config import (
    BiomarkersConfig,
    DatasetConfig,
    DermaConfig,
    MamdaniFISConfig,
    PreprocessingConfig,
    load_config,
    validate_config,
)
from dermafuzzy_core.gradcam import (
    compute_attention_overlap,
    compute_verified_confidence,
)
from dermafuzzy_core.preprocessing import (
    PreprocessingResult,
    apply_lab_clahe,
    extract_lesion_mask_and_ring,
    preprocess_image,
    remove_hair,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # Config
    "DermaConfig",
    "DatasetConfig",
    "PreprocessingConfig",
    "BiomarkersConfig",
    "MamdaniFISConfig",
    "load_config",
    "validate_config",
    # Preprocessing
    "PreprocessingResult",
    "preprocess_image",
    "remove_hair",
    "apply_lab_clahe",
    "extract_lesion_mask_and_ring",
    # Biomarkers
    "BiomarkerResult",
    "compute_asymmetry",
    "compute_border_fuzziness",
    "compute_color_variation",
    "extract_abcd_biomarkers",
    # Grad-CAM
    "compute_attention_overlap",
    "compute_verified_confidence",
]
