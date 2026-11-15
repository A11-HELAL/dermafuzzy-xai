# Feature Specification: Production OpenCV Preprocessing, Perilesional Boundary Ring & Saliency-Verified ABCD Biomarkers

**Feature Branch**: `002-opencv-abcd-biomarkers`

**Created**: 2026-11-15

**Status**: Draft

**Input**: User description: "Build Spec 002: Production OpenCV Preprocessing, Perilesional Boundary Ring & Saliency-Verified ABCD Biomarkers (002-opencv-abcd-biomarkers). Objective: Implement Room 2 (packages/dermafuzzy_core/preprocessing.py) and Room 3 (packages/dermafuzzy_core/biomarkers.py and packages/dermafuzzy_core/gradcam.py) as 100% pure, deterministic, config-driven scientific modules with read-only NumPy array guarantees and synthetic-phantom unit tests."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Deterministic Dermoscopic Preprocessing & Perilesional Ring Isolation (Priority: P1)

As a clinical computer vision researcher, I want raw dermoscopic images automatically cleaned of hair artifacts, normalized for illumination variations while preserving chromaticity, accurately segmented into lesion masks, and bounded with a perilesional boundary ring so that downstream biomarker equations operate on standardized, artifact-free representations with deep immutability guarantees.

**Why this priority**: Raw dermoscopic images contain disruptive hairs, gel bubbles, lighting gradients, and vignetting. Downstream ABCD and fuzzy diagnosis rely completely on an accurate, artifact-free segmentation mask $M$ and edge transition ring $R_{edge}$. Without deterministic preprocessing and zero-division fallbacks, biomarker extraction fails or produces invalid results.

**Independent Test**: Supply synthetic and real dermoscopic images (including dark hairs, uneven lighting, and degenerate empty/saturated frames), verify hair inpainting via Telea algorithm, verify CIE-LAB L-channel CLAHE preservation of chromatic channels, verify Largest Connected Component (LCC) segmentation, and verify that all output arrays are strictly write-protected (`arr.flags.writeable == False`).

**Acceptance Scenarios**:

1. **Given** an RGB dermoscopic image with dark hair occlusions, **When** `preprocess_image(img, config)` executes, **Then** dark hairs are detected into a binary mask and restored via Telea inpainting, illumination is normalized via CLAHE on the `L` channel while `a` and `b` channels remain bit-identical, and a binary lesion mask $M \in \{0, 1\}$ is extracted.
2. **Given** an extreme degenerate input (pure black, pure white, or uniform noise where segmented area ratio is outside $[0.02, 0.90]$), **When** preprocessing executes, **Then** a deterministic centered elliptical fallback mask is generated with `fallback_used=True`, ensuring downstream pipelines never receive an empty mask or divide by zero.
3. **Given** a successful `PreprocessingResult`, **When** any consumer attempts to mutate an array buffer (e.g., `result.clean_rgb[0, 0, 0] = 255`), **Then** Python raises `ValueError: assignment destination is read-only`.
4. **Given** the segmented mask $M$, **When** boundary ring isolation executes, **Then** morphological dilation and erosion with the configured 5x5 kernel generate a closed boundary band $R_{edge} = M_{outer} - M_{inner} \in \{0, 1\}$.

---

### User Story 2 - Explainable Hybrid ABCD Biomarker Extraction (Priority: P2)

As a dermatologist or diagnostic system, I want objective, continuous clinical biomarkers (Asymmetry $A$, Fuzzy Border Irregularity $B_{fuzzy}$, and Multi-Channel Color Variation $C$) computed from the preprocessed lesion and bounded strictly to $[0.0, 1.0]$ so that the diagnostic reasoning remains interpretable, mathematically stable, and invariant to image orientation.

**Why this priority**: The ABCD diagnostic criteria are the clinical gold standard for melanoma triage. Biomarkers must be robust against acquisition rotations and discretization noise (pixel staircasing) while capturing true biological irregularity and pigment heterogeneity.

**Independent Test**: Evaluate the biomarker suite on synthetic phantoms with known mathematical properties (ideal circle, asymmetric L-shape, jagged star, sharp vs. blurry perilesional rims, uniform vs. multi-pigment lesions), verifying expected score separations and orientation invariance across 0°, 30°, 45°, 90°, and 135° rotations.

**Acceptance Scenarios**:

1. **Given** an ideal circular synthetic lesion, **When** computing asymmetry $A$ and geometric border irregularity $B_{geo}$, **Then** $A \le 0.05$ and $B_{geo} \le 0.08$.
2. **Given** an asymmetric synthetic L-shape lesion, **When** computing asymmetry $A$, **Then** $A \ge 0.35$.
3. **Given** an asymmetric lesion rotated at 0°, 30°, 45°, 90°, and 135°, **When** computing asymmetry $A$, **Then** spatial moment alignment ensures rotation invariance with $|\Delta A| \le 0.03$.
4. **Given** a star-shaped synthetic contour, **When** computing geometric border irregularity $B_{geo}$ with polygon approximation smoothing, **Then** $B_{geo} \ge 0.50$.
5. **Given** a sharp step-edge lesion vs. a feathered Gaussian-blurred lesion rim, **When** computing perilesional rim fuzziness $B_{grad}$ within $R_{edge}$, **Then** the sharp rim yields $B_{grad} \le 0.25$ and the blurry rim yields $B_{grad} \ge 0.55$ (separation $\ge 0.30$).
6. **Given** a uniform color lesion vs. a quad-color synthetic lesion (brown, black, red, blue-gray), **When** computing multi-channel color variation $C$, **Then** the uniform lesion yields $C \le 0.03$ and the quad-color lesion yields $C \ge 0.45$.
7. **Given** extracted biomarkers, **When** calling `result.as_vector()`, **Then** a 4-element tuple of normalized floats in $[0.0, 1.0]$ is returned.

---

### User Story 3 - Saliency-Verified Deep Biomarker Attribution (Priority: P3)

As a safety auditor, I want deep learning classifier confidence spatially verified against the physical lesion mask using Grad-CAM attention overlap so that high CNN predictions driven by irrelevant peripheral artifacts (such as skin markers, hair remnants, or diagnostic rulers) are automatically penalized.

**Why this priority**: Black-box CNNs frequently rely on spurious correlations (e.g. surgical ink or measurement rulers). In accordance with Constitution Principle I, spatial verification ensures that deep biomarker confidence $D_{verified}$ reflects genuine lesion attribution.

**Independent Test**: Pass simulated CNN predictions ($p_{cnn}$) and Grad-CAM activation heatmaps representing on-lesion activations versus off-lesion background ruler artifacts (both low-resolution $7 \times 7$ and full-resolution $224 \times 224$), verifying automatic interpolation, overlap calculation $\alpha$, and confidence gating $D_{verified} = p_{cnn} \cdot \alpha$.

**Acceptance Scenarios**:

1. **Given** a high CNN probability $p_{cnn} = 0.90$ with a Grad-CAM heatmap concentrated entirely over the segmented lesion $M$, **When** computing verified confidence, **Then** spatial overlap $\alpha \ge 0.85$ and $D_{verified} \ge 0.75$.
2. **Given** a high CNN probability $p_{cnn} = 0.90$ with a Grad-CAM heatmap concentrated entirely off-lesion on a peripheral ruler artifact, **When** computing verified confidence, **Then** spatial overlap $\alpha \le 0.15$ and $D_{verified} \le 0.15$, penalizing the spurious prediction.
3. **Given** a low-resolution $7 \times 7$ activation heatmap (such as from a deep convolutional feature extractor), **When** passed into `verify_deep_confidence`, **Then** the heatmap is automatically upsampled via bilinear interpolation to match the mask spatial dimensions before overlap calculation.
4. **Given** an empty or completely inactive heatmap ($\sum H = 0$), **When** verification executes, **Then** $\alpha = 0.0$ and $D_{verified} = 0.0$ without numerical instability.

---

### Edge Cases

- **Empty / Non-3D / Invalid Input Images**: Arrays with shape not `(H, W, 3)`, non-uint8 types, or containing non-finite values are rejected at the input gate with descriptive `ValueError`.
- **Image Resolution Mismatch**: Input images differing from `config.dataset.image_size` are resized using `cv2.INTER_AREA` (for downsampling) or `cv2.INTER_LINEAR` (for upsampling) while maintaining channel integrity.
- **Extreme Vignetting / Border-Touching Lesions**: If Otsu thresholding produces a mask touching more than 50% of the image boundary perimeter, corner vignetting suppression is applied before LCC extraction.
- **Empty / Blank Segmentation (Zero-Division Risk)**: Pure black, pure white, or saturated inputs yielding an area ratio outside `[0.02, 0.90]` trigger the deterministic centered elliptical fallback mask, guaranteeing non-zero area for all subsequent biomarker denominators.
- **Single-Pixel / Degenerate Contours**: Contours with zero perimeter or area utilize denominator stabilizers ($\epsilon = 1e-6$) to ensure numerical safety.
- **Heatmaps with Negative Values**: Negative activations are clamped to zero via ReLU before summation and normalization.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Preprocessing module MUST reside in `packages/dermafuzzy_core/preprocessing.py` and biomarker verification modules MUST reside in `packages/dermafuzzy_core/biomarkers.py` and `packages/dermafuzzy_core/gradcam.py`.
- **FR-002**: Input images MUST be validated as 3D NumPy arrays of shape `(H, W, 3)` and dtype `uint8`. Invalid dimensions, empty inputs, or non-finite values MUST raise `ValueError`.
- **FR-003**: If input image dimensions differ from `config.dataset.image_size` `(W, H)`, the module MUST resize the image deterministically (`cv2.INTER_AREA` when downsampling, `cv2.INTER_LINEAR` when upsampling).
- **FR-004**: Hair removal MUST apply morphological Black-Hat filtering with an elliptical structuring element of diameter `config.preprocessing.blackhat_kernel_size` (17x17) on grayscale, threshold dark fibers into a binary mask, and restore texture via Telea inpainting (`cv2.INPAINT_TELEA`, radius `config.preprocessing.inpaint_radius`).
- **FR-005**: Contrast Limited Adaptive Histogram Equalization (CLAHE) MUST be applied strictly to the luminance ($L$) channel in CIE-LAB space with parameters `clipLimit=config.preprocessing.clahe_clip_limit` and `tileGridSize=config.preprocessing.clahe_tile_grid`. The chromatic channels `a` and `b` MUST remain strictly bit-identical.
- **FR-006**: Lesion segmentation MUST perform Gaussian smoothing (`gaussian_blur_kernel=5`) on the $L$ channel, apply inverse Otsu thresholding, suppress border-touching vignetting artifacts if border contact $> 50\%$, and retain only the Largest Connected Component (LCC).
- **FR-007**: If the segmented mask area ratio falls outside `config.preprocessing.mask_area_bounds` (`[0.02, 0.90]`), the module MUST generate a deterministic centered elliptical fallback mask and flag `fallback_used=True`.
- **FR-008**: The perilesional boundary ring $R_{edge}$ MUST be computed as the difference between morphological dilation and erosion using `config.preprocessing.boundary_ring_kernel` ($5 \times 5$): $R_{edge} = M_{outer} - M_{inner} \in \{0, 1\}$.
- **FR-009**: All output NumPy arrays in `PreprocessingResult` MUST be marked read-only via `arr.flags.writeable = False`.
- **FR-010**: Asymmetry score $A$ MUST be calculated by computing spatial central moments $\mu_{20}, \mu_{02}, \mu_{11}$, rotating the mask by $-\theta = -0.5 \arctan(2\mu_{11}/(\mu_{20} - \mu_{02}))$ aligned to array center $(W/2, H/2)$ via affine transformation with `cv2.INTER_NEAREST`, folding across horizontal and vertical axes via XOR, and normalizing:
  $$A = \text{np.clip}\left(\frac{\text{Area}(M_{left} \oplus M_{right}) + \text{Area}(M_{top} \oplus M_{bottom})}{2.0 \cdot \text{Area}(M) + \epsilon}, 0.0, 1.0\right)$$
- **FR-011**: Geometric border irregularity $B_{geo}$ MUST be calculated from the contour of $M$ smoothed via `cv2.approxPolyDP` with epsilon $0.005 \cdot P$, normalized as:
  $$B_{geo} = \text{np.clip}\left(1.0 - \frac{4\pi \text{Area}(M)}{P^2 + \epsilon}, 0.0, 1.0\right)$$
- **FR-012**: Perilesional rim gradient fuzziness $B_{grad}$ MUST be evaluated within $R_{edge} == 1$ across CIE-LAB $L$ and chromatic channels, quantifying normalized radial gradient spread and edge sharpness, clipped to $[0.0, 1.0]$.
- **FR-013**: Total border irregularity MUST combine geometric and gradient scores: $B_{fuzzy} = \text{np.clip}(w_{geo} B_{geo} + w_{grad} B_{grad}, 0.0, 1.0)$ using weights from `config.biomarkers.b_fuzzy_weights`.
- **FR-014**: Multi-channel color variation $C$ MUST evaluate pixel dispersion within $M == 1$ combining normalized standard deviations and inter-decile ranges across CIE-LAB ($L, a, b$) and HSV ($S, V$) channels, clipped to $[0.0, 1.0]$.
- **FR-015**: Attention-verified deep confidence $D_{verified}$ MUST validate $p_{cnn} \in [0.0, 1.0]$, apply ReLU to the Grad-CAM heatmap, resize the heatmap to match mask dimensions via bilinear interpolation, compute spatial overlap $\alpha = \text{np.clip}(\sum (H \odot M) / (\sum H + \epsilon), 0.0, 1.0)$, and compute $D_{verified} = \text{np.clip}(p_{cnn} \cdot \alpha, 0.0, 1.0)$.
- **FR-016**: `packages/dermafuzzy_core/` MUST NEVER import heavy deep learning frameworks (`tensorflow`, `keras`, `torch`) at the module level.

### Key Entities

- **PreprocessingResult**: Immutable dataclass encapsulating preprocessed imagery and segmentation:
  - `clean_rgb`: 3D uint8 array `(H, W, 3)`, read-only.
  - `hair_mask`: 2D uint8 array `(H, W)` in `{0, 255}`, read-only.
  - `lesion_mask`: 2D uint8 array `(H, W)` in `{0, 1}`, read-only.
  - `boundary_ring`: 2D uint8 array `(H, W)` in `{0, 1}`, read-only.
  - `area_ratio`: float proportion of lesion area to total image area.
  - `fallback_used`: boolean indicating whether the fallback ellipse mask was triggered.
- **BiomarkerResult**: Immutable dataclass encapsulating extracted explainable biomarkers:
  - `A`: float asymmetry score in $[0.0, 1.0]$.
  - `B_fuzzy`: float hybrid border irregularity score in $[0.0, 1.0]$.
  - `C`: float multi-channel color variation score in $[0.0, 1.0]$.
  - `D_verified`: float attention-verified deep confidence in $[0.0, 1.0]$.
  - `B_geo`: float geometric jaggedness score in $[0.0, 1.0]$.
  - `B_grad`: float perilesional rim fuzziness score in $[0.0, 1.0]$.
  - `alpha`: float Grad-CAM on-lesion attention overlap proportion in $[0.0, 1.0]$.
  - `.as_vector()`: returns `tuple[float, float, float, float]` corresponding to $(A, B_{fuzzy}, C, D_{verified})$.
  - `.as_dict()`: returns a dictionary mapping biomarker names to float scores.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Single-sample preprocessing, segmentation, boundary ring derivation, and full ABCD biomarker extraction executes in under 100 milliseconds on a standard dual-core CPU (Intel Core i5-7200U baseline).
- **SC-002**: 100% of NumPy arrays in `PreprocessingResult` have `writeable == False`; any mutation attempt raises `ValueError`.
- **SC-003**: Asymmetry metric $A$ achieves rotational invariance across 0°, 30°, 45°, 90°, and 135° rotations with maximum deviation $|\Delta A| \le 0.03$.
- **SC-004**: Synthetic phantom validation achieves required benchmark thresholds:
  - Ideal Circle: $A \le 0.05$ and $B_{geo} \le 0.08$.
  - Asymmetric L-Shape: $A \ge 0.35$.
  - Jagged Star: $B_{geo} \ge 0.50$.
  - Sharp vs Blurry Rim: $B_{grad}(\text{blurry}) - B_{grad}(\text{sharp}) \ge 0.30$.
  - Uniform vs Quad-Color: $C(\text{uniform}) \le 0.03$ and $C(\text{quad}) \ge 0.45$.
  - On-lesion Attention: $\alpha \ge 0.85$; Off-lesion Ruler Artifact: $\alpha \le 0.15$.
- **SC-005**: 100% test pass rate across `test_preprocessing.py`, `test_biomarkers.py`, and `test_gradcam.py`.
- **SC-006**: Zero heavy framework imports (`tensorflow`, `keras`, `torch`) in `dermafuzzy_core`.

## Assumptions

- OpenCV (`cv2`) and NumPy are available in the runtime environment.
- Input dermoscopic images follow standard 8-bit RGB color encoding.
- Grad-CAM heatmaps can be supplied as raw 2D NumPy float arrays from any upstream CNN architecture.
