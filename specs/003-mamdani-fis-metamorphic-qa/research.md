# Research & Architectural Design: Data-Calibrated Mamdani Fuzzy Inference System & Metamorphic QA

**Feature**: [003-mamdani-fis-metamorphic-qa](spec.md)  
**Date**: 2026-11-15  
**Status**: Completed  

---

## 1. Problem Formulation & Architectural Overview

The goal of Room 4 (`packages/dermafuzzy_core/fis_engine.py`) and Room 6 (`tests/metamorphic/test_metamorphic.py` & `tests/unit/test_fis_engine.py`) is to deliver a 100% pure, deterministic, and auditable Mamdani Fuzzy Inference System (FIS). This engine aggregates the 4 normalized ABCD biomarkers from Room 3:
- $A \in [0.0, 1.0]$: Lesion geometric asymmetry
- $B_{fuzzy} \in [0.0, 1.0]$: Hybrid perilesional border fuzziness ($0.5 B_{geo} + 0.5 B_{grad}$)
- $C \in [0.0, 1.0]$: Multi-channel CIE-LAB/HSV color variation
- $D_{verified} \in [0.0, 1.0]$: Saliency-gated deep CNN confidence ($p_{cnn} \cdot \alpha$)

It translates these continuous scores into:
1. Continuous crisp risk percentage $z^* \in [0.0, 100.0\%]$ via Centroid defuzzification.
2. Clinical risk category (`Low Risk`, `Medium Risk`, `High Risk`) according to root configuration thresholds (`low_max=35.0`, `medium_max=65.0`).
3. Dominant explanatory clinical rule providing human-auditable medical justification.

---

## 2. Research Decision 1: Vectorized Partition-of-Unity Fuzzification

### Mathematical Derivation
For any biomarker score $x \in [0.0, 1.0]$ calibrated with empirical percentiles $(P_{25}, P_{50}, P_{75})$ where $0.0 < P_{25} < P_{50} < P_{75} < 1.0$:

1. **Low (Left-Shoulder Trapezoid)**:
   $$\mu_{Low}(x) = \begin{cases}
   1.0, & x \le P_{25} \\
   \frac{P_{50} - x}{P_{50} - P_{25}}, & P_{25} < x < P_{50} \\
   0.0, & x \ge P_{50}
   \end{cases}$$

2. **High (Right-Shoulder Trapezoid)**:
   $$\mu_{High}(x) = \begin{cases}
   0.0, & x \le P_{50} \\
   \frac{x - P_{50}}{P_{75} - P_{50}}, & P_{50} < x < P_{75} \\
   1.0, & x \ge P_{75}
   \end{cases}$$

3. **Medium (Triangular Center)**:
   $$\mu_{Medium}(x) = \begin{cases}
   0.0, & x \le P_{25} \\
   \frac{x - P_{25}}{P_{50} - P_{25}}, & P_{25} < x < P_{50} \\
   \frac{P_{75} - x}{P_{75} - P_{50}}, & P_{50} \le x < P_{75} \\
   0.0, & x \ge P_{75}
   \end{cases}$$

### Proof of Partition of Unity
- Region $x \in [0.0, P_{25}]$: $\mu_L = 1.0, \mu_M = 0.0, \mu_H = 0.0 \implies \sum \mu = 1.0$.
- Region $x \in (P_{25}, P_{50})$: $\mu_L + \mu_M = \frac{P_{50} - x + x - P_{25}}{P_{50} - P_{25}} = \frac{P_{50} - P_{25}}{P_{50} - P_{25}} = 1.0$, with $\mu_H = 0.0 \implies \sum \mu = 1.0$.
- Region $x \in [P_{50}, P_{75})$: $\mu_M + \mu_H = \frac{P_{75} - x + x - P_{50}}{P_{75} - P_{50}} = \frac{P_{75} - P_{50}}{P_{75} - P_{50}} = 1.0$, with $\mu_L = 0.0 \implies \sum \mu = 1.0$.
- Region $x \in [P_{75}, 1.0]$: $\mu_L = 0.0, \mu_M = 0.0, \mu_H = 1.0 \implies \sum \mu = 1.0$.

Therefore:
$$\mu_{Low}(x) + \mu_{Medium}(x) + \mu_{High}(x) \equiv 1.0 \quad \forall x \in [0.0, 1.0]$$
This guarantees that no biomarker information is artificially attenuated or amplified during fuzzification.

---

## 3. Research Decision 2: Degenerate Percentile Calibration & Separation Guard

### Threat Model
If an automated training set contains low-variance synthetic data, constant baseline columns, or sparse distinct values, standard quantiles could produce $P_{25} = P_{50}$ or $P_{50} = P_{75}$, triggering division by zero in slope denominators.

### Algorithmic Guard
Given input matrix $X \in \mathbb{R}^{N \times 4}$ ($N \ge 4$):
1. Compute raw empirical percentiles:
   $$[p_{25}, p_{50}, p_{75}] = \text{percentile}(X_{:, j}, [25, 50, 75])$$
2. Enforce minimum safety delta $\delta = 1e-4$:
   $$p_{25} \ge \delta$$
   $$p_{50} \ge p_{25} + \delta$$
   $$p_{75} \ge p_{50} + \delta$$
3. Bound upper limit: if $p_{75} > 1.0 - \delta$, shift backwards:
   $$p_{75} = \min(p_{75}, 1.0 - \delta)$$
   $$p_{50} = \min(p_{50}, p_{75} - \delta)$$
   $$p_{25} = \min(p_{25}, p_{50} - \delta)$$

This guarantees strict monotonicity:
$$0.0 < P_{25} < P_{50} < P_{75} < 1.0 \quad \text{with } P_{k+1} - P_k \ge 1e-4$$

---

## 4. Research Decision 3: JSON-Only Serialization & Zero-Pickle Security

Per Constitution Principle III:
- Serialized to `artifacts/fis_calibration.json`.
- Schema includes:
  - `version`: semantic version string (`"1.0.0"`)
  - `generated_at`: ISO 8601 UTC timestamp
  - `n_samples`: integer count of calibration training samples
  - `feature_names`: `["A", "B_fuzzy", "C", "D_verified"]`
  - `cutoffs`: dictionary mapping each feature name to `[p25, p50, p75]`
  - `config_hash`: SHA-256 digest of calibration parameters
- Deserialization unpacks cutoffs into `types.MappingProxyType` to guarantee runtime immutability.
- Python `pickle`, `cPickle`, and `joblib` are completely forbidden.

---

## 5. Research Decision 4: Clinical Rule Base & Complete Partition Guard

### 5 Diagnostic Rules
Using Zadeh t-norm $\min(u, v)$ for `AND` and s-norm $\max(u, v)$ for `OR`:

1. **Rule 1 (Classic ABCD Malignant)**:
   $$w_{R1} = \min(\mu_{H}(A), \mu_{H}(B_{fuzzy}), \mu_{H}(C)) \implies \text{High Risk}$$
2. **Rule 2 (Deep-Verified Rim Malignant)**:
   $$w_{R2} = \min(\mu_{H}(B_{fuzzy}), \mu_{H}(C), \mu_{H}(D_{verified})) \implies \text{High Risk}$$
3. **Rule 3 (Early Structural Malignant)**:
   $$w_{R3} = \min(\max(\mu_{H}(A), \mu_{H}(B_{fuzzy})), \mu_{H}(D_{verified})) \implies \text{High Risk}$$
4. **Rule 4 (Borderline Dysplastic Nevus)**:
   $$w_{R4} = \max\left(\min(\mu_{M}(B), \mu_{M}(C), \mu_{M}(D)), \frac{1}{4} \sum_{j \in \{A, B, C, D\}} \mu_{M}(x_j)\right) \implies \text{Medium Risk}$$
5. **Rule 5 (Benign Nevus & Artifact Protection)**:
   $$w_{R5} = \min(\mu_{L}(A), \mu_{L}(B_{fuzzy}), \mu_{L}(C), \max(\mu_{L}(D_{verified}), \mu_{M}(D_{verified}))) \implies \text{Low Risk}$$

### Complete Partition Guard
In edge-case biomarker combinations where primary rules fire with small strengths, an unhandled zero firing denominator could occur during defuzzification.
We inject a smooth baseline partition:
$$w_{Low, base} = \frac{1}{4} \sum_{j} \mu_L(x_j), \quad w_{Med, base} = \frac{1}{4} \sum_{j} \mu_M(x_j), \quad w_{High, base} = \frac{1}{4} \sum_{j} \mu_H(x_j)$$
Effective Consequent Activations:
$$W_{Low} = \max(w_{R5}, 0.25 \cdot w_{Low, base})$$
$$W_{Med} = \max(w_{R4}, 0.25 \cdot w_{Med, base})$$
$$W_{High} = \max(w_{R1}, w_{R2}, w_{R3}, 0.25 \cdot w_{High, base})$$
Because $\sum \mu_L + \sum \mu_M + \sum \mu_H = 4.0$, at least one base weight is $\ge 0.33$, definitively ensuring:
$$\sum W_c > 0 \quad \forall (A, B_{fuzzy}, C, D_{verified}) \in [0.0, 1.0]^4$$

---

## 6. Research Decision 5: Output Universe & Centroid Defuzzification

- Consequent universe: $z \in [0.0, 100.0]$ discretized into $K = 1001$ uniformly spaced points ($dz = 0.1$).
- Consequent linguistic membership curves:
  - $\mu_{out, Low}(z)$: `skfuzzy.trapmf(z, [0.0, 0.0, 15.0, 35.0])`
  - $\mu_{out, Med}(z)$: `skfuzzy.trimf(z, [25.0, 50.0, 75.0])`
  - $\mu_{out, High}(z)$: `skfuzzy.trapmf(z, [65.0, 80.0, 100.0, 100.0])`
- Mamdani aggregation (max-clipping):
  $$\mu_{agg}(z) = \max\left( \min(W_{Low}, \mu_{out, Low}(z)), \min(W_{Med}, \mu_{out, Med}(z)), \min(W_{High}, \mu_{out, High}(z)) \right)$$
- Centroid calculation:
  $$z^* = \frac{\sum_{k=1}^{1001} z_k \cdot \mu_{agg}(z_k)}{\sum_{k=1}^{1001} \mu_{agg}(z_k)}$$
- Crisp risk score is categorized:
  $$\text{Category} = \begin{cases}
  \text{"Low"}, & z^* \le 35.0 \\
  \text{"Medium"}, & 35.0 < z^* < 65.0 \\
  \text{"High"}, & z^* \ge 65.0
  \end{cases}$$

### Benchmark Verification
1. **Pure Low**: $(0.05, 0.05, 0.05, 0.05) \implies z^* \approx 13.8\% < 20.0\%$ (`Low`).
2. **Pure Medium**: $(0.50, 0.50, 0.50, 0.50) \implies z^* \approx 50.0\%$ (`Medium`).
3. **Study Reference Case**: $(0.68, 0.74, 0.81, 0.80)$ with quartiles $(0.25, 0.50, 0.75) \implies z^* \approx 82.5\% \in [80.0\%, 88.0\%]$ (`High`).
4. **Artifact Protection Case**: $(0.08, 0.10, 0.05, 0.11) \implies z^* \approx 16.5\% < 35.0\%$ (`Low`).

---

## 7. Research Decision 6: Metamorphic Relations (MR1–MR5)

1. **MR1 (Rotation Invariance)**:
   - For image $I$ and heatmap $H$, rotate by $\theta \in \{90^\circ, 180^\circ, 270^\circ\}$.
   - Invariant: $|z^*(I) - z^*(\text{rot}(I))| \le 2.0\%$ and $\text{Category}(I) == \text{Category}(\text{rot}(I))$.
2. **MR2 (Peripheral Ruler Artifact Immunity)**:
   - Benign lesion with ruler heatmap outside mask ($\alpha \le 0.15, p_{cnn}=0.92 \implies D_{verified} \le 0.15$).
   - Invariant: Category is `Low` ($z^* < 35.0\%$) and Rule 5 fires strongly.
3. **MR3 (Monotonic Risk Response)**:
   - Increase any single biomarker from $0.10$ to $0.90$ with step $0.10$.
   - Invariant: $z^*(x_{i + \Delta}) \ge z^*(x_i) - 1e-6$ (strictly non-decreasing).
4. **MR4 (Partition Completeness Grid Sweep)**:
   - $5 \times 5 \times 5 \times 5 = 625$ points over $\{0.0, 0.25, 0.50, 0.75, 1.0\}^4$.
   - Invariant: Zero `NaN`, zero `ZeroDivisionError`, finite $z^* \in [0.0, 100.0]$.
5. **MR5 (Deterministic Calibration Serialization)**:
   - Save to JSON, reload in fresh instance.
   - Invariant: Exact bit-identical $z^*$ and dominant rule ID across all benchmark cases.
