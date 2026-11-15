# Feature Specification: Data-Calibrated Mamdani Fuzzy Decision Support System & Metamorphic QA Gate

**Feature Branch**: `003-mamdani-fis-metamorphic-qa`  
**Created**: 2026-11-15  
**Status**: Draft  
**Input**: User description: "Build Spec 003: Data-Calibrated Mamdani Fuzzy Decision Support System & Metamorphic QA Gate (`003-mamdani-fis-metamorphic-qa`)."

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Data-Driven Training Calibration & Secure JSON Persistence (Room 4) (Priority: P1) 🎯 MVP

As a clinical data engineer and regulatory auditor, I want to calibrate biomarker distribution percentiles ($P_{25}, P_{50}, P_{75}$) exclusively from training splits and persist them into a human-readable, schema-validated JSON artifact, so that fuzzy inference thresholds are mathematically grounded, data leakage from validation/test splits is strictly prevented, and the system is protected against arbitrary code execution (no `pickle`).

**Why this priority**:
Without valid calibration parameters, fuzzy membership functions cannot be constructed. Regulatory medical device software mandates human-auditable, tamper-evident parameters with zero possibility of remote code execution vulnerabilities.

**Independent Test**:
Can be fully tested by feeding synthetic or empirical training distributions of ABCD biomarker vectors $(N \times 4)$, computing empirical percentiles, verifying strict monotonicity $0.0 < P_{25} < P_{50} < P_{75} < 1.0$ with a non-zero separation guard ($\delta \ge 1e-4$), serializing to JSON, reloading into an immutable data container, and confirming bit-identical numeric values.

**Acceptance Scenarios**:
1. **Given** a 2D array of training biomarker vectors with shape $(N, 4)$ ($N \ge 4$) bounded in $[0.0, 1.0]$, **When** `calibrate_fis_percentiles` is executed, **Then** percentiles $P_{25}, P_{50}, P_{75}$ are computed per biomarker with strict monotonic ordering ($P_{25} + \delta \le P_{50}$ and $P_{50} + \delta \le P_{75}$).
2. **Given** degenerate or low-variance training data where two percentiles coincide, **When** calibration runs, **Then** a safety separation delta ($\delta = 1e-4$) is enforced to guarantee non-zero width for triangular and trapezoidal membership ramps.
3. **Given** a calibrated parameter object, **When** saved to `artifacts/fis_calibration.json` and reloaded, **Then** all values match exactly, all mappings are wrapped in immutable proxies (`MappingProxyType`), and zero Python `pickle` files are used.
4. **Given** non-2D arrays, empty samples, non-finite values (`NaN`, `Inf`), or boolean inputs, **When** calibration is attempted, **Then** the system rejects them immediately with descriptive `ValueError`.

---

### User Story 2 - Pure Mamdani Fuzzy Inference Engine & Clinical Explainability (Room 4) (Priority: P2)

As a dermatologist or clinical oncologist, I want each patient's ABCD biomarker scores ($A, B_{fuzzy}, C, D_{verified}$) to be evaluated by transparent, interpretable Mamdani fuzzy rules with partition-of-unity membership functions and centroid defuzzification, so that I receive an actionable risk percentage ($[0.0, 100.0\%]$), a clinical risk tier (`Low`, `Medium`, `High`), and the exact firing strengths of the dominant diagnostic rules.

**Why this priority**:
Black-box neural networks cannot provide verifiable clinical justifications. The Mamdani Fuzzy Inference System (FIS) translates quantitative dermatoscopic features into symbolic, human-understandable clinical rules while preserving full continuous sensitivity across ambiguous lesion boundaries.

**Independent Test**:
Can be tested independently by supplying standardized benchmark biomarker vectors:
- Pure low-risk inputs yield Centroid $< 25.0\%$ (`Low Risk`).
- Pure medium-risk inputs yield Centroid in $[40.0\%, 60.0\%]$ (`Medium Risk`).
- Pure high-risk inputs yield Centroid $> 75.0\%$ (`High Risk`).
- Study worked example $(A=0.68, B_{fuzzy}=0.74, C=0.81, D_{verified}=0.80)$ yields `High Risk` in the expected range $[80.0\%, 88.0\%]$ with Rule 1/Rule 2 dominance.

**Acceptance Scenarios**:
1. **Given** any scalar biomarker score $x \in [0.0, 1.0]$ and calibrated cutoffs $(P_{25}, P_{50}, P_{75})$, **When** fuzzification is computed via left-shoulder trapezoid (`Low`), triangle (`Medium`), and right-shoulder trapezoid (`High`), **Then** Partition of Unity is strictly satisfied: $\mu_{Low}(x) + \mu_{Medium}(x) + \mu_{High}(x) = 1.0 \pm 1e-6$.
2. **Given** biomarker inputs, **When** the 5 clinical rules are evaluated:
   - Rule 1: `IF A is High AND B_fuzzy is High AND C is High THEN Risk is High`
   - Rule 2: `IF B_fuzzy is High AND C is High AND D_verified is High THEN Risk is High`
   - Rule 3: `IF (A is High OR B_fuzzy is High) AND D_verified is High THEN Risk is High`
   - Rule 4: `IF B_fuzzy is Medium AND C is Medium AND D_verified is Medium THEN Risk is Medium`
   - Rule 5: `IF A is Low AND B_fuzzy is Low AND C is Low AND (D_verified is Low OR D_verified is Medium) THEN Risk is Low`
   **Then** firing strengths are computed using standard Zadeh t-norm ($\min$) for `AND` and s-norm ($\max$) for `OR`.
3. **Given** any biomarker combination across the continuous domain $[0.0, 1.0]^4$, **When** defuzzification is executed, **Then** a Complete Rule Partition Guard guarantees total firing strength $\sum w_i > 0$, preventing undefined division by zero.
4. **Given** an aggregated output risk fuzzy set over universe $z \in [0, 100]$, **When** defuzzified via Centroid method, **Then** the crisp risk percentage is categorized according to configuration thresholds (`Low`: $\le 35.0\%$, `Medium`: $35.0\% - 65.0\%$, `High`: $> 65.0\%$) and returned in an immutable `FISPrediction` object.

---

### User Story 3 - End-to-End Metamorphic Diagnostic QA Gate (Room 6) (Priority: P3)

As a clinical quality assurance lead, I want to verify the entire integrated diagnostic pipeline (OpenCV preprocessing $\to$ ABCD biomarker extraction $\to$ Grad-CAM attention gating $\to$ Mamdani FIS) against 5 clinical Metamorphic Relations (MR1–MR5), so that we guarantee diagnostic stability under real-world clinical variations, artifact presence, and data serialization.

**Why this priority**:
Traditional unit testing checks static values, but medical diagnostic software requires metamorphic testing to ensure behavioral consistency across rotations, artifact occlusions, and monotonicity invariants that cannot be captured by ground-truth labels alone.

**Independent Test**:
Can be fully tested by executing `pytest tests/metamorphic/test_metamorphic.py`, evaluating end-to-end pipelines across MR1 through MR5 with deterministic synthetic dermoscopic phantoms.

**Acceptance Scenarios**:
1. **MR1 (Rotation Invariance)**: **Given** an input dermoscopic image and associated Grad-CAM heatmap, **When** the inputs are rotated by 90°, 180°, and 270°, **Then** the final FIS `risk_score` changes by no more than $\pm 2.0$ percentage points ($|\Delta \text{score}| \le 2.0\%$) and preserves the identical `risk_category`.
2. **MR2 (Peripheral Ruler Artifact Immunity)**: **Given** a benign circular lesion with high raw CNN confidence ($p_{cnn} = 0.92$) whose saliency heatmap highlights a peripheral ruler artifact outside the lesion ($\alpha \le 0.15$), **When** processed end-to-end, **Then** the verified confidence $D_{verified}$ drops to $\le 0.15$, activating Rule 5 to output `Low Risk` ($< 35.0\%$), whereas an on-lesion malignant phantom outputs `High Risk` ($> 65.0\%$).
3. **MR3 (Monotonic Biomarker Risk Response)**: **Given** baseline biomarker inputs, **When** any single biomarker ($A, B_{fuzzy}, C, D_{verified}$) increases monotonically from $0.10$ to $0.90$ with other biomarkers held constant, **Then** the resulting `risk_score` is monotonically non-decreasing ($\Delta \text{risk} \ge -1e-6$).
4. **MR4 (Partition Completeness Grid Sweep)**: **Given** a full $5 \times 5 \times 5 \times 5$ hypercube grid (625 combinations covering $[0.0, 0.25, 0.50, 0.75, 1.00]^4$), **When** passed through the FIS engine, **Then** 100% of combinations yield valid, finite numerical risk scores in $[0.0, 100.0]$ with zero `NaN`, zero `Inf`, and zero unhandled edge cases.
5. **MR5 (Deterministic Calibration Serialization)**: **Given** calibrated parameters, **When** saved to JSON and reloaded in a clean process, **Then** batch predictions produce bit-identical `risk_score`, `risk_category`, and `dominant_rule_id` across all test vectors.

---

### Edge Cases

- **Flat/Zero Variance Training Data**: If all training samples for a biomarker have the same value (e.g. constant $0.5$), empirical percentiles collapse to a single point. Guard: add minimum separation offsets $\pm \delta$ ($1e-4$) ensuring $P_{25} < P_{50} < P_{75}$.
- **Extreme Boundary Inputs**: Biomarker inputs at exact endpoints ($0.000$ or $1.000$). Left and right shoulder membership functions must saturate cleanly at $1.0$ without numerical instability.
- **Off-Lesion Saliency Collapse**: When Grad-CAM heatmap is entirely outside the lesion mask, $\alpha = 0.0 \implies D_{verified} = 0.0$. The rule base must smoothly transition to benign/indeterminate rules without crashing or firing rules with zero weights.
- **Equi-firing Clinical Rules**: Multiple rules fire with identical maximal strength. The system must report all co-dominant rules deterministically based on rule priority order (Rule 1 > Rule 2 > Rule 3 > Rule 4 > Rule 5) without raising ambiguous selection exceptions.
- **Zero Rule Firing Corner Cases**: In rare biomarker regions between rule antecedent definitions, primary rules could fire with near-zero strength. Complete Rule Partition Guard injects a proportional baseline activation to guarantee $\sum w_i > 0$.

---

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide `calibrate_fis_percentiles(train_biomarkers, config=None) -> FISCalibration` accepting a 2D array of shape $(N, 4)$ ($N \ge 4$) representing $[A, B_{fuzzy}, C, D_{verified}]$ scores in $[0.0, 1.0]$.
- **FR-002**: System MUST compute empirical percentiles ($P_{25}, P_{50}, P_{75}$) per biomarker using `config.mamdani_fis.percentiles`.
- **FR-003**: System MUST enforce strict monotonic separation $0.0 < P_{25} < P_{50} < P_{75} < 1.0$ with minimum delta $\delta = 1e-4$, adjusting percentiles if training data exhibits degeneracy.
- **FR-004**: System MUST serialize `FISCalibration` to and deserialize from human-readable JSON (`artifacts/fis_calibration.json`) with full schema validation and `MappingProxyType` deep immutability.
- **FR-005**: System MUST strictly reject Python `pickle`, `.pkl`, and `.pickle` serialization formats across all modules.
- **FR-006**: System MUST implement membership functions for `Low` (left-shoulder trapezoid), `Medium` (triangle), and `High` (right-shoulder trapezoid) satisfying Partition of Unity: $\mu_{Low}(x) + \mu_{Medium}(x) + \mu_{High}(x) = 1.0 \pm 1e-6$ for all $x \in [0.0, 1.0]$.
- **FR-007**: System MUST evaluate the 5 clinical Mamdani rules:
  - Rule 1 (Classic ABCD Malignant)
  - Rule 2 (Deep-Verified Rim Malignant)
  - Rule 3 (Early Structural/Border Malignant)
  - Rule 4 (Borderline Dysplastic / Follow-Up)
  - Rule 5 (Benign & Artifact Protection)
- **FR-008**: System MUST combine rule antecedents using standard Zadeh t-norm ($\min$) for `AND` and s-norm ($\max$) for `OR`.
- **FR-009**: System MUST guarantee non-zero total rule firing strength ($\sum w_i > 0$) for every possible input in $[0.0, 1.0]^4$ via a complete rule partition guard.
- **FR-010**: System MUST define output consequent sets over universe $z \in [0, 100]$ aligned with configuration thresholds `low_max=35.0` and `medium_max=65.0`.
- **FR-011**: System MUST compute crisp risk scores via Centroid defuzzification: $z^* = \frac{\int z \cdot \mu_{out}(z) dz}{\int \mu_{out}(z) dz} \in [0.0, 100.0]$.
- **FR-012**: System MUST return an immutable, frozen dataclass `FISPrediction` containing `risk_score: float`, `risk_category: str`, `dominant_rule_id: str`, `dominant_rule_text: str`, `rule_firings: Mapping[str, float]`, and `memberships: Mapping[str, Mapping[str, float]]`.
- **FR-013**: System MUST reject `bool`, `NaN`, `Inf`, non-2D arrays, and out-of-bounds inputs across all public functions with descriptive `ValueError`.
- **FR-014**: System MUST verify MR1 (Rotation Invariance $|\Delta \text{score}| \le 2.0\%$), MR2 (Peripheral Ruler Artifact Immunity), MR3 (Monotonic Risk Response), MR4 (Partition Completeness Grid Sweep of 625 points), and MR5 (Deterministic JSON Serialization) in `tests/metamorphic/test_metamorphic.py`.
- **FR-015**: System MUST maintain zero module-level imports of `tensorflow`, `keras`, or `torch` in `packages/dermafuzzy_core/fis_engine.py` per Constitution Principle I.

---

### Key Entities *(include if feature involves data)*

- **`FISCalibration`**: Immutable dataclass containing calibrated percentiles ($P_{25}, P_{50}, P_{75}$) for each of the 4 biomarkers ($A, B_{fuzzy}, C, D_{verified}$), metadata (sample count, timestamp, config hash), and dictionary/JSON conversion methods with `MappingProxyType` protection.
- **`FuzzyMembershipSet`**: Evaluated linguistic memberships (`low`, `medium`, `high`) for a single biomarker satisfying $\mu_{low} + \mu_{medium} + \mu_{high} = 1.0$.
- **`ClinicalRule`**: Representation of a clinical diagnostic rule, containing rule ID, antecedent expression, consequent linguistic term, and computed firing strength.
- **`FISPrediction`**: Final explainable diagnostic decision artifact containing crisp risk percentage ($0.0 - 100.0\%$), qualitative category (`Low`, `Medium`, `High`), dominant explanatory rule, and complete audit trail of all rule activations and membership values.

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% test pass rate across unit tests (`test_fis_engine.py`) and metamorphic test suite (`test_metamorphic.py`).
- **SC-002**: Pure membership partition: Partition of Unity $\mu_L(x) + \mu_M(x) + \mu_H(x) = 1.0$ holds within $|1.0 - \sum \mu| \le 1e-6$ across 10,000 random test samples in $[0.0, 1.0]$.
- **SC-003**: Metamorphic Rotation Invariance (MR1): Rotating test phantoms across 90°, 180°, and 270° changes final `risk_score` by no more than $2.0\%$ and maintains identical risk category.
- **SC-004**: Metamorphic Artifact Immunity (MR2): Benign lesions corrupted by strong off-lesion ruler activations ($\alpha \le 0.15$) are reliably classified as `Low Risk` ($\le 35.0\%$) via Rule 5 protection.
- **SC-005**: Metamorphic Grid Completeness (MR4): 625 continuous hypercube grid points across $[0.0, 1.0]^4$ execute with zero `NaN`, zero zero-division, and finite output in $[0.0, 100.0]$.
- **SC-006**: Sub-second execution: Full FIS evaluation of a single sample completes in $< 5\text{ms}$ on an Intel i5-7200U equivalent processor.

---

## Assumptions

- Training split biomarker vectors are pre-computed and normalized to $[0.0, 1.0]$ via Room 2 and Room 3 before percentile calibration.
- The 4 canonical biomarkers are evaluated in standard order: `A`, `B_fuzzy`, `C`, `D_verified`.
- Risk thresholds `low_max=35.0` and `medium_max=65.0` in `derma_config.yaml` represent consensus clinical risk stratification for melanocytic lesions.
- Metamorphic tests utilize deterministic synthetic phantoms created in `tests/unit/phantoms.py` to ensure reproducible, fast, self-contained CI execution without external image dataset dependencies.
