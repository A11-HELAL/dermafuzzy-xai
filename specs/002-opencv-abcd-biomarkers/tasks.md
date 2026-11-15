# Tasks: Production OpenCV Preprocessing, Perilesional Boundary Ring & Saliency-Verified ABCD Biomarkers

**Input**: Design documents from `/specs/002-opencv-abcd-biomarkers/`
**Prerequisites**: [plan.md](plan.md), [spec.md](spec.md), [research.md](research.md), [data-model.md](data-model.md), [contracts/](contracts/)
**Tests**: Complete unit and synthetic phantom test suites required by specification and Constitution Principle IV.
**Organization**: Tasks are grouped by user story to enable independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (`[US1]`, `[US2]`, `[US3]`)
- Include exact file paths in all descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Update packaging dependencies to support OpenCV and numerical computing.

- [X] T001 Update `pyproject.toml` dependencies to include `numpy>=1.24.0` and `opencv-python-headless>=4.8.0` alongside `pyyaml>=6.0`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Synthetic Dermoscopic Phantom generators and domain result dataclasses required by all user stories.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [X] T002 [P] Implement deterministic Synthetic Dermoscopic Phantom generators in `tests/unit/phantoms.py` providing:
  - `make_circle_phantom(size=(224, 224), radius=50)` -> ideal circle mask and RGB image
  - `make_l_shape_phantom(size=(224, 224))` -> asymmetric L-shape mask
  - `make_star_phantom(size=(224, 224), num_points=7)` -> jagged star contour mask
  - `make_hair_overlay(img_rgb, num_strands=5)` -> RGB image with thin dark hair fiber occlusions
  - `make_blurred_rim_phantom(size=(224, 224), sigma=6.0)` vs `make_sharp_step_phantom(size=(224, 224))` -> perilesional transition phantoms
  - `make_multi_pigment_phantom(size=(224, 224))` -> lesion with brown, black, red, and blue-gray quadrants vs uniform lesion
  - `make_gradcam_heatmap(size=(224, 224), on_lesion=True, low_res_shape=(7, 7))` -> on-lesion vs off-lesion ruler artifact heatmaps
- [X] T003 [P] Implement `PreprocessingResult` dataclass in `packages/dermafuzzy_core/preprocessing.py` using `@dataclass(frozen=True, slots=True)` with fields `clean_rgb: np.ndarray`, `hair_mask: np.ndarray`, `lesion_mask: np.ndarray`, `boundary_ring: np.ndarray`, `area_ratio: float`, and `fallback_used: bool`
- [X] T004 [P] Implement `BiomarkerResult` dataclass in `packages/dermafuzzy_core/biomarkers.py` using `@dataclass(frozen=True, slots=True)` with fields `A: float`, `B_fuzzy: float`, `C: float`, `D_verified: float`, `B_geo: float`, `B_grad: float`, `alpha: float`, and methods `.as_vector() -> tuple[float, float, float, float]` and `.as_dict() -> dict[str, float]`

**Checkpoint**: Foundation ready - phantom generators and domain dataclasses available for TDD.

---

## Phase 3: User Story 1 - Deterministic Dermoscopic Preprocessing & Perilesional Ring (Room 2) (Priority: P1) 🎯 MVP

**Goal**: Clean dark hair occlusions via Black-Hat + Telea, normalize lighting via CIE-LAB CLAHE on `L`, segment lesion via LCC with border vignetting suppression and fallback, extract 5x5 boundary ring, and lock arrays as read-only.

**Independent Test**: Run `pytest tests/unit/test_preprocessing.py` on synthetic hair images, blank/saturated frames, and normal phantoms; confirm hair removal, CLAHE `L` channel normalization with bit-identical `a`/`b` channels, `fallback_used=True` on area ratio violations outside `[0.02, 0.90]`, and `ValueError` on write attempts.

### Tests for User Story 1 ⚠️
> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [X] T005 [P] [US1] Write unit tests in `tests/unit/test_preprocessing.py` verifying hair removal recovery, CIE-LAB L-channel CLAHE with bit-identical chromatic preservation, LCC lesion segmentation, 5x5 boundary ring $R_{edge} = M_{outer} - M_{inner}$, centered elliptical fallback mask when area ratio is outside $[0.02, 0.90]$, and deep immutability (`arr.flags.writeable == False` raising `ValueError` on mutation)
- [X] T006 [P] [US1] Write negative input tests in `tests/unit/test_preprocessing.py` verifying rejection of non-3D tensors, wrong channel counts, non-uint8 dtypes, empty images, and non-finite values

### Implementation for User Story 1

- [X] T007 [US1] Implement input gate validation and deterministic resizing (`cv2.INTER_AREA`/`cv2.INTER_LINEAR` to `config.dataset.image_size`) in `packages/dermafuzzy_core/preprocessing.py`
- [X] T008 [US1] Implement `remove_hair(img_rgb, kernel_size, inpaint_radius)` using grayscale morphological Black-Hat with elliptical structuring element and `cv2.INPAINT_TELEA` in `packages/dermafuzzy_core/preprocessing.py`
- [X] T009 [US1] Implement `apply_lab_clahe(img_rgb, clip_limit, tile_grid)` applying `cv2.createCLAHE` strictly to the `L` channel in `CIE-LAB` space while preserving `a` and `b` bit-identical in `packages/dermafuzzy_core/preprocessing.py`
- [X] T010 [US1] Implement `extract_lesion_mask_and_ring(clean_rgb, config)` with Gaussian blur on `L`, inverse Otsu thresholding, border vignetting suppression (clearing border frame if border contact > 50%), LCC extraction, centered elliptical fallback on area ratio violations outside $[0.02, 0.90]$, and $5 \times 5$ boundary ring derivation in `packages/dermafuzzy_core/preprocessing.py`
- [X] T011 [US1] Implement `preprocess_image(img_rgb, config)` orchestrating Steps 1-4 and locking all returned NumPy arrays with `arr.flags.writeable = False` inside `PreprocessingResult` in `packages/dermafuzzy_core/preprocessing.py`

**Checkpoint**: At this point, User Story 1 (Room 2) is fully functional and testable independently (MVP ready).

---

## Phase 4: User Story 3 - Saliency-Verified Deep Confidence (Room 3: `gradcam.py`) (Priority: P3)

**Goal**: Validate CNN prediction $p_{cnn}$, apply ReLU and bilinear upsampling to 2D Grad-CAM heatmaps, calculate spatial attention overlap $\alpha$, and compute verified confidence $D_{verified} = p_{cnn} \cdot \alpha$.

**Independent Test**: Run `pytest tests/unit/test_gradcam.py` confirming on-lesion activation yields $\alpha \ge 0.85$, off-lesion ruler artifact yields $\alpha \le 0.15$, $7 \times 7$ maps upsample cleanly to $224 \times 224$, and zero-activation heatmaps yield $\alpha = 0.0, D_{verified} = 0.0$.

### Tests for User Story 3 ⚠️
> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [X] T012 [P] [US3] Write unit tests in `tests/unit/test_gradcam.py` verifying on-lesion ($\alpha \ge 0.85$) vs off-lesion ruler artifact ($\alpha \le 0.15$) attention overlap, automatic $7 \times 7 \to 224 \times 224$ bilinear upsampling, zero-activation handling ($\alpha = 0.0, D_{verified} = 0.0$), and strict input validation (`bool`, `NaN`, `Inf`, out-of-bounds `p_cnn` outside $[0.0, 1.0]$)

### Implementation for User Story 3

- [X] T013 [US3] Implement `compute_attention_overlap(heatmap, mask, eps)` and `compute_verified_confidence(p_cnn, alpha)` with ReLU clamping, bilinear interpolation, and confidence gating in `packages/dermafuzzy_core/gradcam.py`

**Checkpoint**: User Story 3 is complete and ready for integration into the full ABCD biomarker pipeline.

---

## Phase 5: User Story 2 - Explainable Hybrid ABCD Biomarker Extraction (Room 3: `biomarkers.py`) (Priority: P2)

**Goal**: Extract normalized Asymmetry $A$ (via moments and axis rotation), hybrid Fuzzy Border Irregularity $B_{fuzzy}$ ($w_{geo} B_{geo} + w_{grad} B_{grad}$), and multi-channel Color Variation $C$ (CIE-LAB and HSV dispersion).

**Independent Test**: Run `pytest tests/unit/test_biomarkers.py` verifying circle ($A \le 0.05, B_{geo} \le 0.08$), L-shape ($A \ge 0.35$), star ($B_{geo} \ge 0.50$), rotation invariance across 0°, 30°, 45°, 90°, 135° ($|\Delta A| \le 0.03$), rim fuzziness separation ($\ge 0.30$), and color variation on uniform ($C \le 0.03$) vs quad-color ($C \ge 0.45$).

### Tests for User Story 2 ⚠️
> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [X] T014 [P] [US2] Write unit tests in `tests/unit/test_biomarkers.py` verifying Asymmetry $A$ on circle and L-shape, rotation invariance across 0°, 30°, 45°, 90°, 135° ($|\Delta A| \le 0.03$), geometric border jaggedness $B_{geo}$ on circle and star, perilesional rim gradient fuzziness $B_{grad}$ separation ($\ge 0.30$) between sharp and blurry rims, and color variation $C$ on uniform vs multi-pigment lesions
- [X] T015 [P] [US2] Write negative validation tests in `tests/unit/test_biomarkers.py` verifying rejection of invalid shapes, non-finite inputs, and degenerate masks

### Implementation for User Story 2

- [X] T016 [US2] Implement `compute_asymmetry(mask, eps)` via spatial moments, centroid/axis affine transformation with `cv2.INTER_NEAREST`, vertical/horizontal XOR single-sided folds, and $[0.0, 1.0]$ normalization in `packages/dermafuzzy_core/biomarkers.py`
- [X] T017 [US2] Implement `compute_border_fuzziness(clean_rgb, mask, boundary_ring, config)` combining polygon-smoothed circularity deficit $B_{geo}$ (`cv2.approxPolyDP` with `0.004 * P`) and Sobel crest concentration $B_{grad}$ inside `boundary_ring > 0` in `packages/dermafuzzy_core/biomarkers.py`
- [X] T018 [US2] Implement `compute_color_variation(clean_rgb, mask, eps)` evaluating normalized standard deviation and inter-decile range across CIE-LAB ($L, a, b$) and HSV ($S, V$) inside `mask > 0` in `packages/dermafuzzy_core/biomarkers.py`
- [X] T019 [US2] Implement `extract_abcd_biomarkers(clean_rgb, mask, boundary_ring, config, p_cnn, heatmap)` orchestrating $A, B_{fuzzy}, C, D_{verified}$ and returning `BiomarkerResult` in `packages/dermafuzzy_core/biomarkers.py`

**Checkpoint**: User Stories 1, 2, and 3 are functional and fully covered by positive and negative unit tests.

---

## Phase 6: Public API Integration

**Purpose**: Re-export public API symbols cleanly in package root.

- [X] T020 Update `packages/dermafuzzy_core/__init__.py` to re-export `PreprocessingResult`, `preprocess_image`, `BiomarkerResult`, `compute_asymmetry`, `compute_border_fuzziness`, `compute_color_variation`, `extract_abcd_biomarkers`, `compute_attention_overlap`, and `compute_verified_confidence` alongside `__version__`

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Execute full verification suite and confirm compliance with constitution and quickstart guide.

- [X] T021 [P] Run complete unit test suite with `pytest -v tests/unit/` confirming 100% pass rate across `test_config.py`, `test_preprocessing.py`, `test_gradcam.py`, and `test_biomarkers.py`
- [X] T022 [P] Verify Constitution Principle I compliance across all modules in `packages/dermafuzzy_core/` asserting zero imports of `tensorflow`, `keras`, or `torch`
- [X] T023 Run quickstart verification scenarios from `specs/002-opencv-abcd-biomarkers/quickstart.md` to confirm end-to-end execution

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately.
- **Foundational (Phase 2)**: Depends on Setup (T001) - BLOCKS all user stories.
- **User Story 1 (Phase 3)**: Depends on Foundational (T002, T003). Delivers Room 2 MVP.
- **User Story 3 (Phase 4)**: Depends on Foundational (T002). Delivers Grad-CAM verification.
- **User Story 2 (Phase 5)**: Depends on US1 (T011) and US3 (T013) for full ABCD orchestration.
- **Public API (Phase 6)**: Depends on US1, US2, and US3 completion.
- **Polish (Phase 7)**: Depends on all prior phases being complete.

### User Story Dependencies

- **User Story 1 (P1)**: Foundational phantoms and dataclass ready → TDD tests (T005, T006) → implementation (T007-T011).
- **User Story 3 (P3)**: Heatmap phantoms ready → TDD tests (T012) → implementation (T013).
- **User Story 2 (P2)**: Integrates preprocessing masks from US1 and saliency from US3 → TDD tests (T014, T015) → implementation (T016-T019).

### Parallel Opportunities

- **Phase 2 Foundational**: `T002` (phantoms), `T003` (`PreprocessingResult`), and `T004` (`BiomarkerResult`) can all run in parallel.
- **Phase 3 & Phase 4**: Once Foundational is complete, US1 (Room 2) and US3 (Grad-CAM) can proceed in parallel.
- **Phase 7 Polish**: `T021` (pytest) and `T022` (pure core check) can run in parallel.

---

## Implementation Strategy

### MVP First (User Story 1 - Room 2 Preprocessing)
1. Complete Phase 1: Setup (`T001`).
2. Complete Phase 2: Foundational phantoms and dataclasses (`T002`, `T003`, `T004`).
3. Complete Phase 3: Room 2 Preprocessing tests (`T005`, `T006`) and implementation (`T007`-`T011`).
4. **VALIDATE MVP**: Confirm `preprocess_image()` successfully cleans hair, equalizes lighting, segments lesions, and locks output arrays.

### Incremental Delivery
1. Add User Story 3 (`T012`, `T013`) for Grad-CAM attention overlap and confidence verification.
2. Add User Story 2 (`T014`-`T019`) for complete ABCD biomarker calculations.
3. Update package re-exports (`T020`).
4. Run Phase 7 Polish (`T021`-`T023`) to verify full test suite and quickstart readiness.
