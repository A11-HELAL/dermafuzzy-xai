# Research & Architecture Decisions: Production OpenCV Preprocessing & Saliency-Verified ABCD Biomarkers

**Feature**: [002-opencv-abcd-biomarkers](spec.md) | **Date**: 2026-11-15

## Overview
This document records technical research, algorithmic formulations, and mathematical decisions for Room 2 (`preprocessing.py`) and Room 3 (`biomarkers.py` and `gradcam.py`) in `packages/dermafuzzy_core/`.

---

## Decision 1: Grayscale Morphological Black-Hat Filter & Telea Inpainting

### Context
Dermoscopic images frequently contain hair occlusions that cross lesion borders. If unhandled, hair lines falsely inflate border jaggedness ($B_{geo}$), create spurious color variations ($C$), and misguide Grad-CAM attention.

### Decision
1. Convert RGB to grayscale: `gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)`.
2. Build an elliptical structuring element: `k = config.preprocessing.blackhat_kernel_size` (17x17), `kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))`.
3. Compute morphological Black-Hat: `blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)`.
4. Isolate hair fibers using dynamic thresholding: calculate high percentile threshold (e.g. 98th percentile or Otsu if high contrast) on non-zero blackhat values. Uniform, clean skin produces an empty mask, while dark strands are isolated.
5. Inpaint using Fast Marching Telea: `cv2.inpaint(img_rgb, hair_mask, r, cv2.INPAINT_TELEA)` with `r = config.preprocessing.inpaint_radius` (3px).

### Rationale
- Morphological Black-Hat isolates structures that are darker than their immediate background and narrower than the kernel diameter (ideal for thin hair strands).
- Telea inpainting smoothly propagates surrounding skin and lesion pigment into the hair channels without introducing blur artifacts.

### Alternatives Considered
- **Bilateral Filtering**: Fails to remove thick hairs and blurs legitimate micro-pigment globules.
- **Navier-Stokes Inpainting (`cv2.INPAINT_NS`)**: Slower on dual-core CPUs and produces softer, less defined edges compared to Telea.

---

## Decision 2: Chromatic-Preserving CIE-LAB CLAHE Normalization

### Context
Dermoscopy image datasets (such as HAM10000) exhibit significant lighting variation, vignette shading, and flash reflection. Direct RGB equalization alters chromatic ratios, corrupting biological color scoring.

### Decision
1. Convert RGB to `CIE-LAB`: `lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)`.
2. Apply Contrast Limited Adaptive Histogram Equalization (`cv2.createCLAHE`) strictly to the Luminance channel `lab[:, :, 0]`:
   - `clipLimit = config.preprocessing.clahe_clip_limit` (2.0)
   - `tileGridSize = config.preprocessing.clahe_tile_grid` (8, 8)
3. Maintain channels `lab[:, :, 1]` (`a`) and `lab[:, :, 2]` (`b`) bit-identical.
4. Convert back to RGB: `clean_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)`.

### Rationale
- Completely adheres to Constitution Principle II (strict mandate that CLAHE must never operate directly on RGB channels).
- Enhances local structural contrast without shifting hue or chromatic saturation.

### Alternatives Considered
- **RGB Histogram Equalization**: Severely distorts chromaticity, turning natural melanin pigments into unnatural artifacts.
- **Global Gamma Correction**: Cannot compensate for non-uniform spatial illumination (e.g., center-bright, edge-dim dermatoscope lighting).

---

## Decision 3: Robust Lesion Segmentation with Border-Polarity Guard and Deterministic Fallback

### Context
Lesions must be segmented into a binary mask $M \in \{0, 1\}$. Thresholding edge cases include:
1. Corner vignetting (dark ring around circular dermatoscope optics).
2. Pure black, pure white, or saturated frames yielding degenerate 0-pixel or full-frame masks.

### Decision
1. Smooth luminance with Gaussian blur: `cv2.GaussianBlur(L, (5, 5), 0)`.
2. Apply inverse Otsu thresholding: `cv2.threshold(blurred, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)`.
3. Border-Polarity Guard: Sample border frame pixels (outer 2% perimeter). If $>50\%$ of border pixels are foreground while center region is background (optical vignette inversion), clear the border frame before connected components analysis.
4. Extract Largest Connected Component (LCC) via `cv2.connectedComponentsWithStats`.
5. Fallback Gate: Calculate `area_ratio = M.sum() / (H * W)`. If `area_ratio` is outside `config.preprocessing.mask_area_bounds` (`[0.02, 0.90]`), generate a deterministic centered elliptical fallback mask of radii `(int(0.35 * W), int(0.35 * H))` and set `fallback_used = True`.
6. Compute perilesional boundary ring: `R_edge = cv2.dilate(M, k5) - cv2.erode(M, k5)` with a 5x5 elliptical structuring element.
7. Deep immutability: Set `arr.flags.writeable = False` on all returned NumPy arrays.

### Rationale
- Prevents downstream zero-division errors across all ABCD calculations.
- LCC eliminates scattered peripheral pigment specks while preserving the primary lesion.

---

## Decision 4: Centroid & Principal Inertia Axis Alignment for Rotation-Invariant Asymmetry $A$

### Context
Clinical asymmetry evaluation requires folding the lesion along its primary and secondary symmetry axes. An arbitrary lesion orientation in the camera frame must not alter the asymmetry score.

### Decision
1. Compute spatial central moments on binary mask $M$: $\mu_{20}, \mu_{02}, \mu_{11}$ using `cv2.moments(mask)`.
2. Compute centroid $(c_x, c_y) = (m_{10}/(m_{00} + \epsilon), m_{01}/(m_{00} + \epsilon))$.
3. Compute principal axis orientation: $\theta = 0.5 \arctan(2\mu_{11} / (\mu_{20} - \mu_{02}))$.
4. Construct affine matrix rotating by $\theta$ and translating $(c_x, c_y)$ to array center $((W-1)/2, (H-1)/2)$.
5. Warp mask with `cv2.warpAffine(mask, M_affine, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)`.
6. Fold across vertical and horizontal center axes using `np.fliplr` and `np.flipud`:
   $$\text{xor}_h = \frac{1}{2} \sum \left((M_{aligned} > 0) \oplus \text{fliplr}(M_{aligned} > 0)\right)$$
   $$\text{xor}_v = \frac{1}{2} \sum \left((M_{aligned} > 0) \oplus \text{flipud}(M_{aligned} > 0)\right)$$
   $$A = \text{np.clip}\left(\frac{\text{xor}_h + \text{xor}_v}{\text{Area}(M) + \epsilon}, 0.0, 1.0\right)$$

### Rationale
- Rotation to the principal axis aligns the lesion along its natural inertia axes.
- Dividing by 2 in single-sided XOR folds prevents double-counting mismatched pixel pairs.
- Yields guaranteed rotation invariance ($|\Delta A| \le 0.03$ across arbitrary rotations).

---

## Decision 5: Hybrid Fuzzy Border Irregularity ($B_{fuzzy} = w_{geo} B_{geo} + w_{grad} B_{grad}$)

### Context
Classic ABCD defines border irregularity via macroscopic indentations, while dermoscopy also recognizes fuzzy, faded pigment transitions into normal skin.

### Decision
1. **Geometric Jaggedness ($B_{geo}$)**:
   - Extract largest external contour using `cv2.findContours`.
   - Apply polygon approximation smoothing: `cv2.approxPolyDP(contour, 0.004 * raw_perimeter, True)` to remove grid discretization staircase noise.
   - Circularity deficit:
     $$B_{geo} = \text{np.clip}\left(1.0 - \frac{4\pi \text{Area}(M)}{P^2 + \epsilon}, 0.0, 1.0\right)$$
   - Ideal circle yields $B_{geo} \le 0.08$; jagged star yields $B_{geo} \ge 0.50$.
2. **Perilesional Rim Gradient Fuzziness ($B_{grad}$)**:
   - Convert `clean_rgb` to normalized float CIE-LAB.
   - Compute Sobel gradient magnitude on luminance $L$: $G = \sqrt{G_x^2 + G_y^2}$.
   - Within boundary ring $R_{edge} > 0$, measure crest concentration (ratio of top 10% gradient to mean gradient) and gradient spread. A sharp 1-pixel step edge has concentrated gradient peak ($B_{grad} \le 0.25$), whereas a Gaussian-feathered transition spreads gradient uniformly across the ring ($B_{grad} \ge 0.55$).
3. **Fuse**:
   $$B_{fuzzy} = \text{np.clip}(w_{geo} B_{geo} + w_{grad} B_{grad}, 0.0, 1.0)$$

---

## Decision 6: Multi-Channel Statistical Color Heterogeneity ($C$)

### Context
Melanomas exhibit multiple colors (light brown, dark brown, black, red, white, blue-gray). Benign nevi are predominantly uniform.

### Decision
1. Restrict evaluation strictly to lesion pixels ($M > 0$).
2. Compute color metrics across:
   - CIE-LAB channels ($L, a, b$): captures perceptual lightness and red-green/blue-yellow opposing colors.
   - HSV channels ($S, V$): captures color saturation and value depth.
3. Combine normalized standard deviation ($\sigma / \text{range}$) and inter-decile range ($(P_{90} - P_{10}) / \text{range}$) across channels.
4. Scale so solid uniform patches yield $C \le 0.03$, while multi-pigment melanoma phantoms yield $C \ge 0.45$.

---

## Decision 7: Grad-CAM Saliency Verification ($D_{verified} = p_{cnn} \cdot \alpha$)

### Context
Deep networks can produce high melanoma probabilities ($p_{cnn} \ge 0.90$) due to non-lesion artifacts (measurement rulers, hair tufts, optical markings).

### Decision
1. Validate $p_{cnn}$ is a finite float in $[0.0, 1.0]$.
2. Validate 2D heatmap is a finite NumPy array. Apply ReLU: $H = \max(H, 0.0)$.
3. If spatial shape differs from lesion mask $M$ (e.g. $7 \times 7 \to 224 \times 224$), upsample via bilinear interpolation: `cv2.resize(H, (W, H), interpolation=cv2.INTER_LINEAR)`.
4. If $\sum H \le \epsilon$, return $\alpha = 0.0, D_{verified} = 0.0$.
5. Compute spatial overlap ratio:
   $$\alpha = \text{np.clip}\left(\frac{\sum (H \odot M)}{\sum H + \epsilon}, 0.0, 1.0\right)$$
6. Compute verified deep confidence:
   $$D_{verified} = \text{np.clip}(p_{cnn} \cdot \alpha, 0.0, 1.0)$$

### Rationale
- Forces deep learning predictions to be grounded in the physical anatomical lesion mask.
- Prevents spurious background features from inflating diagnostic confidence.
