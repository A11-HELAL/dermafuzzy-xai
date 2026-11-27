# DermaFuzzy-XAI 5-Fold Lesion-Grouped Cross-Validation Report

- **Outer Splits**: `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` grouped strictly by `lesion_id`
- **Evaluated Tiers**: `tier1_cnn` (>=0.50), `tier2_crisp` (>=0.50), `tier3_standard` (>=50.0), `tier3_triage` (>=35.0)

## Cohort: `full_unseen` (N=3339: 1113 mel, 2226 nv)

### Mean ± SD Across 5 Folds

| Tier | Accuracy | Sensitivity | Specificity | Precision | F1-Score | ROC-AUC | PR-AUC |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `tier1_cnn` | 82.2% ± 1.5% | 81.2% ± 7.5% | 82.5% ± 4.6% | 70.3% ± 3.1% | 75.1% ± 2.8% | 0.9125 ± 0.0078 | 0.8487 ± 0.0187 |
| `tier2_crisp` | 69.1% ± 1.0% | 88.0% ± 4.0% | 59.7% ± 3.2% | 52.2% ± 1.9% | 65.5% ± 1.2% | 0.8100 ± 0.0163 | 0.6203 ± 0.0184 |
| `tier3_standard` | 82.0% ± 1.6% | 78.3% ± 4.6% | 83.9% ± 3.2% | 71.0% ± 2.8% | 74.3% ± 1.8% | 0.8937 ± 0.0123 | 0.8106 ± 0.0247 |
| `tier3_triage` | 73.3% ± 2.4% | 94.3% ± 3.0% | 62.7% ± 5.1% | 55.9% ± 2.6% | 70.2% ± 2.0% | 0.8937 ± 0.0123 | 0.8106 ± 0.0247 |

### Pooled Out-of-Fold (OOF) Evaluation & Paired Tests

| Tier | Accuracy | Sensitivity | Specificity | Precision | F1-Score | ROC-AUC | PR-AUC |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `tier1_cnn` | 82.2% | 81.5% | 82.6% | 70.0% | 75.3% | 0.9085 | 0.8439 |
| `tier2_crisp` | 69.1% | 88.0% | 59.7% | 52.2% | 65.5% | 0.8089 | 0.6185 |
| `tier3_standard` | 82.0% | 78.3% | 83.9% | 70.8% | 74.3% | 0.8931 | 0.8085 |
| `tier3_triage` | 73.3% | 94.4% | 62.7% | 55.9% | 70.2% | 0.8931 | 0.8085 |

| Comparison | ΔROC-AUC | DeLong z | DeLong p-value | McNemar (b/c) | McNemar p-value |
|---|:---:|:---:|:---:|:---:|:---:|
| `tier3_standard_vs_tier2_crisp` | +0.0842 | 17.356 | 1.7777e-67 | 586/156 | 2.2392e-59 |
| `tier3_standard_vs_tier1_cnn` | -0.0153 | -5.212 | 1.8643e-07 | 122/129 | 7.0498e-01 |
| `tier3_triage_vs_tier2_crisp` | +0.0842 | 17.356 | 1.7777e-67 | 359/220 | 8.3141e-09 |
| `tier3_triage_vs_tier1_cnn` | -0.0153 | -5.212 | 1.8643e-07 | 149/447 | 1.5592e-35 |

## Cohort: `balanced_1to1` (N=2226: 1113 mel, 1113 nv)

### Mean ± SD Across 5 Folds

| Tier | Accuracy | Sensitivity | Specificity | Precision | F1-Score | ROC-AUC | PR-AUC |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `tier1_cnn` | 82.1% ± 2.6% | 81.2% ± 7.5% | 83.1% ± 3.7% | 82.9% ± 2.0% | 81.9% ± 3.5% | 0.9109 ± 0.0098 | 0.9100 ± 0.0070 |
| `tier2_crisp` | 73.3% ± 0.8% | 88.0% ± 4.0% | 58.5% ± 4.1% | 68.0% ± 1.3% | 76.7% ± 1.0% | 0.8109 ± 0.0152 | 0.7618 ± 0.0161 |
| `tier3_standard` | 81.0% ± 1.4% | 78.3% ± 4.6% | 83.8% ± 3.1% | 83.0% ± 2.0% | 80.5% ± 2.0% | 0.8921 ± 0.0146 | 0.8884 ± 0.0151 |
| `tier3_triage` | 78.3% ± 2.0% | 94.3% ± 3.0% | 62.3% ± 4.6% | 71.5% ± 2.1% | 81.3% ± 1.5% | 0.8921 ± 0.0146 | 0.8884 ± 0.0151 |

### Pooled Out-of-Fold (OOF) Evaluation & Paired Tests

| Tier | Accuracy | Sensitivity | Specificity | Precision | F1-Score | ROC-AUC | PR-AUC |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `tier1_cnn` | 82.3% | 81.5% | 83.0% | 82.8% | 82.1% | 0.9068 | 0.9071 |
| `tier2_crisp` | 73.3% | 88.0% | 58.6% | 68.0% | 76.7% | 0.8095 | 0.7604 |
| `tier3_standard` | 81.0% | 78.3% | 83.8% | 82.9% | 80.5% | 0.8915 | 0.8880 |
| `tier3_triage` | 78.3% | 94.4% | 62.3% | 71.4% | 81.3% | 0.8915 | 0.8880 |

| Comparison | ΔROC-AUC | DeLong z | DeLong p-value | McNemar (b/c) | McNemar p-value |
|---|:---:|:---:|:---:|:---:|:---:|
| `tier3_standard_vs_tier2_crisp` | +0.0820 | 14.213 | 7.5739e-46 | 314/141 | 3.2603e-16 |
| `tier3_standard_vs_tier1_cnn` | -0.0154 | -4.498 | 6.8529e-06 | 74/101 | 4.9055e-02 |
| `tier3_triage_vs_tier2_crisp` | +0.0820 | 14.213 | 7.5739e-46 | 235/122 | 2.2774e-09 |
| `tier3_triage_vs_tier1_cnn` | -0.0154 | -4.498 | 6.8529e-06 | 147/234 | 9.6894e-06 |

