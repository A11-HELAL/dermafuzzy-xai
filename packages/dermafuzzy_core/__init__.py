"""DermaFuzzy-XAI Core Algorithmic Library.

This package contains 100% pure, deterministic Python functions for:
- Configuration parsing, typing, and validation (`dermafuzzy_core.config`).
- OpenCV preprocessing, hair removal, and perilesional boundary rings (`dermafuzzy_core.preprocessing`).
- Explainable ABCD clinical biomarker calculations (`dermafuzzy_core.biomarkers`).
- Saliency-verified Grad-CAM attention gating (`dermafuzzy_core.gradcam`).
- Data-calibrated Mamdani Fuzzy Inference System & Explanation (`dermafuzzy_core.fis_engine`).
- Clinical biostatistical evaluation & 3-tier ablation engine (`dermafuzzy_core.evaluation`).
- Publication-grade 4-panel clinical XAI audit visualizer (`dermafuzzy_core.visualizer`).

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
from dermafuzzy_core.evaluation import (
    AblationReport,
    ClinicalMetrics,
    compute_clinical_metrics,
    evaluate_crisp_abcd_baseline,
    run_ablation_study,
)
from dermafuzzy_core.fis_engine import (
    CLINICAL_RULES,
    ClinicalRule,
    FISCalibration,
    FISPrediction,
    FuzzyMembershipSet,
    calibrate_fis_percentiles,
    compute_memberships,
    diagnose_image,
    evaluate_mamdani_fis,
    load_calibration_json,
    save_calibration_json,
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
from dermafuzzy_core.visualizer import (
    FourPanelReportConfig,
    render_four_panel_report,
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
    # FIS Engine
    "FISCalibration",
    "FuzzyMembershipSet",
    "ClinicalRule",
    "FISPrediction",
    "CLINICAL_RULES",
    "calibrate_fis_percentiles",
    "save_calibration_json",
    "load_calibration_json",
    "compute_memberships",
    "evaluate_mamdani_fis",
    "diagnose_image",
    # Evaluation & Ablation
    "ClinicalMetrics",
    "AblationReport",
    "compute_clinical_metrics",
    "evaluate_crisp_abcd_baseline",
    "run_ablation_study",
    # Visualizer
    "FourPanelReportConfig",
    "render_four_panel_report",
]
