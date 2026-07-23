# Canonical Alignment Subsystem — Audit Report

**Date**: 2026-07-20
**Scope**: Forward canonical alignment (`orbital_shape_prior_st1/data_prep/canonical_align.py`), inverse native mapping (`orbital_shape_prior_st1/engine/native_mapping.py`), and their integration into the CNISP inference pipeline (`orbital_shape_prior_st1/scripts/032_cnisp_infer_corrector.py`).

---

## Table of Contents

1. [Purpose & Role in Pipeline](#1-purpose--role-in-pipeline)
2. [Forward Alignment Pipeline (canonical_align.py)](#2-forward-alignment-pipeline)
3. [Inverse Mapping Pipeline (native_mapping.py)](#3-inverse-mapping-pipeline)
4. [Integration in Inference (032_cnisp_infer_corrector.py)](#4-integration-in-inference)
5. [Label Scheme Handling](#5-label-scheme-handling)
6. [The OS→OD Flip](#6-the-osod-flip)
7. [The Deployment Index Shift Bug Fix](#7-the-deployment-index-shift-bug-fix)
8. [Metadata Contract](#8-metadata-contract)
9. [Downstream Consumers](#9-downstream-consumers)
10. [Design Choices & Invariants](#10-design-choices--invariants)
11. [Known Edge Cases & Safeguards](#11-known-edge-cases--safeguards)

---

## 1. Purpose & Role in Pipeline

The canonical alignment subsystem transforms arbitrary-orientation orbital CT segmentation masks into a **standardized coordinate frame** (RAS+, single-eye, fixed physical extent) suitable for the CNISP implicit MLP. At inference time, the inverse mapping reconstructs the model's output back into the original scanner coordinate system.

```
Forward (data_prep):
  Full-head NIfTI → per-eye 80mm canonical patch (RAS+, single-eye LCC, OS flipped)

Inverse (engine):
  64mm sub-patch prediction → 80mm disk patch → un-flip → un-reorient → full-head native
```

The forward path runs ONCE offline during data preparation. The inverse path runs at every inference (training validation, test-time optimization, corrector pipeline).

---

## 2. Forward Alignment Pipeline

**File**: `orbital_shape_prior_st1/data_prep/canonical_align.py` (802 lines)

### 2.1 Pipeline Steps (per case)

| Step | Operation | Key Function |
|------|-----------|--------------|
| 1 | Load full-head segmentation NIfTI | `align_single_case` |
| 2 | Detect label scheme (nnunet vs labelfusion vs offset) | `detect_label_scheme` |
| 3 | Separate OD/OS via globe connected component analysis | `separate_eyes` |
| 4 | Per eye: locate single-eye LCC in midplane-clipped bbox | `_eye_lcc_in_search_bbox` |
| 5 | Crop 80mm cubic patch centered on LCC centroid | `compute_crop_slices` |
| 6 | Zero out non-LCC voxels (single-eye cleanup) | in-line in `align_single_case` |
| 7 | Remap labels to canonical scheme {0:BG, 1:ON, 2:Globe, 3:Fat, 4:Recti} | `remap_to_canonical` |
| 8 | Reorient array to RAS+ | `reorient_to_ras` |
| 9 | Validate affine is diagonal | `validate_diagonal_affine` |
| 10 | If OS: flip along sagittal axis (axis 0) → pseudo-OD | `flip_os_to_od` |
| 11 | Save patch NIfTI + metadata JSON | `align_dataset` |

### 2.2 OD/OS Separation (`separate_eyes`)

1. Extract the Globe label from the mask using the detected scheme
2. Run 26-connected component analysis on the Globe binary mask
3. Filter CCs by minimum voxel count (default 50)
4. Keep the 2 largest CCs (or 1 if only one exists)
5. Determine L-R world axis from the affine (`nib.aff2axcodes`)
6. Sort by L-R world coordinate: more positive in the R direction = OD, less positive = OS

For single-globe cases (one CC only): assign OD/OS by the sign of its world L-R coordinate.

### 2.3 Single-Eye LCC Isolation (`_eye_lcc_in_search_bbox`)

This is the critical step that ensures each patch contains exactly one eye:

1. Build a search bounding box (default 1.5 × patch_size = 120mm) centered on this eye's globe centroid
2. Clip the bbox at the midplane between the two globe centroids (belt-and-suspenders safeguard)
3. Extract foreground mask within the clipped bbox (using the label_map for scheme-agnostic detection)
4. Find the Largest Connected Component (26-connectivity) of that foreground
5. The LCC centroid becomes the **crop centroid** for the 80mm patch
6. The LCC mask is used to **zero out** any non-LCC voxels in the final patch

**Anatomy assumption** (load-bearing): the four labelled structures inside one orbit (ON + Globe + Fat + Recti) form ONE 26-connected component, and the two orbits never connect to each other through any of those four labels.

### 2.4 Patch Coordinate Convention

- Patch size is in **mm** (not voxels): the downstream MLP works in physical coordinates
- An 80mm patch at 0.5mm spacing = 160 voxels; at 1.0mm spacing = 80 voxels
- Coordinate convention: `offset = spacing/2` (align_corners=False, following Amiranashvili et al.)
- The 80mm patch is a buffer; training re-crops a tighter 64mm sub-patch around the visible LCC centroid

### 2.5 The OS→OD Flip (`flip_os_to_od`)

```python
def flip_os_to_od(data, affine):
    flipped = np.flip(data, axis=0).copy()
    fa = affine.copy()
    fa[:, 0] *= -1
    fa[:3, 3] += affine[:3, 0] * (data.shape[0] - 1)
    return flipped, fa
```

- Flips along **axis 0** (sagittal axis in RAS) — mirrors left/right
- The affine is updated to maintain correct world coordinates: first column negated, origin shifted by `(shape-1) * column_0`
- Purpose: the MLP sees all eyes as if they were OD, reducing the hypothesis space by 2x

### 2.6 Dataset-Level Processing (`align_dataset`)

- Accepts three input sources: QA-filtered checklist CSV, atlas label directory, or an explicit manifest CSV
- Writes per-eye outputs: `labels/{casename}.nii.gz` (uint8 patch) + `metadata/{casename}.json`
- Reports per-case: voxel shape, spacing, structures found
- Errors are caught per-case to keep the batch going

---

## 3. Inverse Mapping Pipeline

**File**: `orbital_shape_prior_st1/engine/native_mapping.py` (697 lines)

### 3.1 Pipeline Steps (per eye prediction)

| Step | Operation | Key Function |
|------|-----------|--------------|
| 1 | LCC cleanup on 64mm prediction (defensive) | `lcc_cleanup_with_warning` |
| 2 | Place 64mm sub-patch into 80mm disk patch | `place_sub_patch_in_disk` |
| 3 | If was_flipped (OS): reverse flip (np.flip axis=0) | `reverse_flip` |
| 4 | Reverse RAS reorientation back to original | `reverse_reorient` |
| 5 | Place into full-head volume using crop_slices | `place_patch_in_volume` |

### 3.2 Key Functions

**`reverse_flip(data)`**: `np.flip(data, axis=0).copy()` — since flip_os_to_od mirrors axis 0, the inverse is the same operation.

**`reverse_reorient(data, original_ornt_codes)`**: Uses nibabel's `ornt_transform` from RAS back to the original orientation codes stored in metadata.

**`place_patch_in_volume(patch, original_shape, crop_slices)`**: Creates a zero volume of the original head shape and inserts the patch at the recorded crop_slices position. Handles boundary clipping.

**`place_sub_patch_in_disk(sub_patch, disk_shape, sub_crop_lo_vox_dense)`**: Places the 64mm sub-patch prediction into the 80mm disk patch frame. Includes out-of-bounds clipping for edge cases.

**`lcc_cleanup_with_warning(pred_class_map, casename)`**: Defensive check — the prediction should already be a single connected shape per eye. If multiple CCs exist, keeps only the largest and prints a WARNING. Non-empty strip = the prior emitted spurious disconnected components.

### 3.3 OD/OS Merge

After both eyes are individually mapped back to native space:

- Because LCC cleanup guarantees single-eye patches, OD and OS can NEVER overlap in the merged volume
- The merge is a simple "foreground wins" union: `merged[eye_vol > 0] = eye_vol[eye_vol > 0]`
- Label remapping (canonical → original scheme) is deferred until AFTER the merge

### 3.4 Affine Reconstruction (`reconstruct_canonical_patch_affine`)

Rebuilds the canonical patch affine from metadata by replaying the forward alignment steps:

1. Start with `original_affine`
2. Apply crop offset: `pa[:3, 3] += original_affine[:3, :3] @ crop_lo`
3. Apply RAS reorientation transform
4. If `was_flipped`: apply the OS→OD flip to the affine

Used by `_deployment_index_shift` to relate two different canonical crops (observed vs dense).

---

## 4. Integration in Inference

**File**: `orbital_shape_prior_st1/scripts/032_cnisp_infer_corrector.py`

The corrector inference script does NOT call `canonical_align` directly. It:

1. **Consumes pre-aligned patches** from disk (produced offline by `align_dataset`)
2. **Resolves alignment metadata** via `_meta_path_for_case` for the dense target and `_observed_meta_path_for` for per-step observed inputs
3. **Passes metadata to native_mapping** for the 5-step inverse after CNISP produces its dense decode
4. **Applies `_deployment_index_shift`** when the observed input has a different canonical crop origin than the dense target (deployment mode with nnUNet pred as the input)

### 4.1 Native Mask Production Flow

```
CNISP latent optimization
  → optimized latent code per (case, step, eye)
  → dense decode at iso-0.5mm on the target patch grid
  → argmax → integer class map in canonical scheme {0..4}
  → native_mapping inverse (LCC → sub-patch → disk → un-flip → un-reorient → full-head)
  → OD/OS merge
  → canonical-to-original label remap {1→1, 2→3, 3→4, 4→2} (canonical→nnunet)
  → NIfTI save with original affine
```

---

## 5. Label Scheme Handling

### 5.1 Input Schemes

| Source | Scheme | Labels |
|--------|--------|--------|
| nnUNet predictions (checklist) | `nnunet` | {1:ON, 2:Recti, 3:Globe, 4:Fat} |
| Atlas manual GT (label-fusion) | `labelfusion` | {1:ON, 3:Recti, 5:Globe, 7:Fat} |
| Atlas with -1000 offset | `labelfusion` (shifted) | {-999:ON, -997:Recti, -995:Globe, -993:Fat} |

### 5.2 Detection Logic (`detect_label_scheme`)

1. Extract unique non-zero labels
2. If minimum label < 0: compute offset, shift labels up, then classify
3. If {5, 7} ⊆ labels → `labelfusion`
4. If 2 ∈ labels → `nnunet`
5. Otherwise → `unknown` (raises error downstream)

### 5.3 Canonical Output (Fixed)

```python
CANONICAL_LABELS = {"BG": 0, "ON": 1, "Globe": 2, "Fat": 3, "Recti": 4}
```

All patches on disk use this scheme. The MLP trains and predicts in canonical labels.

### 5.4 Remap Back to Original

At native-space export, canonical labels are remapped back. The canonical→nnunet mapping:
- 1 (ON) → 1 (ON)
- 2 (Globe) → 3 (Globe)
- 3 (Fat) → 4 (Fat)
- 4 (Recti) → 2 (Recti)

---

## 6. The OS→OD Flip

### 6.1 What is Flipped

**OS (left eye)** is flipped along sagittal axis (array axis 0 in RAS) to appear as pseudo-OD.

### 6.2 Why

The implicit MLP (AutoDecoder) has a fixed-size latent space. Training all eyes as if they were OD reduces the spatial hypothesis space by half — the model never needs to learn "left-vs-right" symmetry explicitly; it only models one side's anatomy.

### 6.3 When (Forward)

In `align_single_case`, after RAS reorientation, if `eye_info["eye"] == "OS"`:
```python
was_flipped = (eye_info["eye"] == "OS")
if was_flipped:
    patch, pa = flip_os_to_od(patch, pa)
```

### 6.4 When (Inverse)

In `native_mapping`, step 3:
```python
if meta.get("was_flipped", False):
    data = reverse_flip(data)  # np.flip(data, axis=0)
```

### 6.5 Affine Consistency

The flip must update the affine so world coordinates remain correct:
- Forward: negate column 0, shift origin by `(shape[0]-1) * original_column_0`
- Inverse: the array flip alone suffices because `reverse_reorient` (step 4) and `place_patch_in_volume` (step 5) use the metadata crop_slices / original_ornt — NOT the patched affine

---

## 7. The Deployment Index Shift Bug Fix

### 7.1 The Bug (documented in `native_mapping_buggy.py`)

In **deployment mode** (where CNISP fits to an nnUNet prediction rather than a GT), the observed input patch and the dense target patch have **different world origins** (different globe centroid locations due to sparsification drift). The naive inverse places each eye at the dense-grid index equal to its observed-grid index, silently assuming the two crops share a world origin.

**Symptoms**:
- OD looked approximately correct (small error)
- OS was grossly misplaced — the axis-0 flip **mirrors** the crop-origin gap
- Error worsened with step_size (larger sparsification drift)

### 7.2 The Fix (`_deployment_index_shift`)

```python
def _deployment_index_shift(dense_meta, observed_meta):
    a_dense = reconstruct_canonical_patch_affine(dense_meta)
    a_obs = reconstruct_canonical_patch_affine(observed_meta)
    delta_world = a_obs[:3, 3] - a_dense[:3, 3]
    return np.linalg.inv(a_dense[:3, :3]) @ delta_world
```

Computes the world-origin difference between the two canonical patches, expressed in dense canonical voxels. This shift is added to `sub_crop_lo_vox_dense` before step (2) placement. Because the dense affine carries the OS flip, the sign is handled automatically for both eyes.

### 7.3 `native_mapping_buggy.py`

The buggy version is kept intentionally — it deliberately OMITS `_deployment_index_shift` to reproduce the pre-fix OS mirror error for ablation experiments. Used only for paired comparison / debugging.

---

## 8. Metadata Contract

### 8.1 `AlignmentMetadata` Fields (stored as JSON per casename)

| Field | Type | Purpose |
|-------|------|---------|
| `source` | str | "checklist" or "atlas" |
| `source_id` | str | Subject/atlas identifier |
| `eye` | str | "OD" or "OS" |
| `casename` | str | `{source_id}_{eye}` (unique key) |
| `original_nifti_path` | str | Path to source NIfTI |
| `original_affine` | 4×4 list | Original scanner affine |
| `original_shape` | [3] list | Original volume dimensions |
| `input_label_scheme` | str | "nnunet" or "labelfusion" |
| `globe_centroid_world` | [3] list | Globe CC centroid (world mm) |
| `crop_centroid_world` | [3] list | LCC centroid used for crop (world mm) |
| `patch_size_mm` | float | Side length of cubic patch (80.0) |
| `search_size_mm` | float | LCC search bbox size (120.0) |
| `crop_center_voxel` | [3] list | Crop center in original voxel coords |
| `crop_slices` | [[lo,hi]×3] | Exact crop indices for inverse placement |
| `original_ornt` | [3] list | Original orientation codes (e.g. ["R","A","S"]) |
| `target_ornt` | str | Always "RAS" |
| `was_flipped` | bool | True for OS eyes |
| `patch_spacing` | [3] list | Voxel spacing in mm |
| `patch_voxel_shape` | [3] list | Patch dimensions in voxels |
| `globe_volume_mm3` | float | Globe volume (QC) |
| `on_volume_mm3` | float | Optic nerve volume (QC) |
| `num_structures_found` | int | How many of 4 structures are present |
| `lcc_voxel_count` | int | Size of the kept LCC |
| `lcc_total_fg_in_bbox` | int | All foreground in the search bbox |
| `lcc_in_patch_count` | int | Foreground voxels in final patch |
| `lcc_fg_in_patch_before` | int | Foreground before LCC cleanup |

### 8.2 Which Fields Drive the Inverse

- `was_flipped` → step 3 (un-flip)
- `original_ornt` → step 4 (un-reorient)
- `crop_slices` + `original_shape` → step 5 (place in full-head)
- `original_affine` + `crop_slices` + `was_flipped` + `patch_voxel_shape` → `reconstruct_canonical_patch_affine` (deployment shift)

---

## 9. Downstream Consumers

| Consumer | What it Reads | Invariant It Relies On |
|----------|---------------|------------------------|
| `engine/dataset.py` | Canonical patches + metadata | Diagonal RAS+ affine; single-eye via LCC; canonical labels {0..4}; fixed 80mm patch_size_mm |
| `engine/infer.py` | Metadata for native inversion | `crop_slices`, `was_flipped`, `original_ornt`, `original_shape`, `original_affine` |
| `engine/native_mapping.py` | Metadata for full inverse | All fields above + `patch_voxel_shape` for affine reconstruction |
| `scripts/032_cnisp_infer_corrector.py` | Pre-aligned patches + per-step metadata | Two metadata variants (dense target vs observed input) for deployment shift |
| `scripts/031_valid_test.py` | Metadata for geometry validation | Verifies round-trip: native mask shape/affine match original |
| `nnunet/build_dataset835_*_patches.py` | `infer_patch_size_mm` | All metadata agree on patch_size_mm (consistency check) |
| `simulation/evaluation/plausibility.py` | Native-space output masks | Merged OD+OS in one file with correct spatial positioning |

---

## 10. Design Choices & Invariants

### 10.1 Load-Bearing Invariants

1. **Single-eye LCC guarantee**: Every canonical patch contains exactly one eye's foreground as a single 26-connected component. Violated → downstream merge overwrites at midline, topology metrics report false violations.

2. **80mm physical patch size**: The MLP's `latent_coords = image_size / 2` formula hard-codes this. Using a different patch_size_mm silently shifts predictions by `(training - new) / 2` mm. The `infer_patch_size_mm` helper exists specifically to prevent this drift.

3. **Diagonal RAS+ affine**: After reorientation, `spacing = np.diagonal(affine[:3,:3])`. The dataset's coordinate-grid constructor and the native_mapping's affine reconstruction both rely on this. Non-diagonal affines (oblique acquisitions) trigger a warning but the pipeline continues.

4. **OS flipped = pseudo-OD**: The MLP never sees a "left eye" geometry. The flip is axis 0 in RAS (sagittal). The inverse must flip the same axis.

5. **Canonical label order {0:BG, 1:ON, 2:Globe, 3:Fat, 4:Recti}**: Hard-coded in training, inference, augmentation. The corrector uses `{1:ON, 2:Recti, 3:Globe, 4:Fat}` (nnunet scheme) — the canonical→nnunet remap happens at native export.

### 10.2 Design Rationale

| Choice | Rationale |
|--------|-----------|
| Patch size in mm (not voxels) | MLP works in physical coordinates; varying voxel count across resolutions is correct |
| 80mm patch = buffer, 64mm sub-patch = training | The visible-LCC centroid drifts from the dense-LCC centroid under sparsification; the buffer absorbs this drift |
| LCC cleanup at BOTH forward and inverse | Forward: ensures training data is single-eye. Inverse: defensive check on prediction quality |
| Midplane clip + LCC (belt-and-suspenders) | Midplane clip is cheap but imperfect (can fail if fat bridges orbits); LCC handles the residual case |
| `search_size_mm = 1.5 × patch_size_mm` | Ensures the LCC centroid is found even if it's shifted from the globe centroid; any crop placed at that centroid still fits inside the bbox |
| Metadata stores the ORIGINAL affine/ornt/shape | The inverse needs the scanner frame, not the canonical frame |

---

## 11. Known Edge Cases & Safeguards

| Edge Case | Handling |
|-----------|----------|
| **Single-globe case** (only 1 CC found) | Assigns OD/OS by world-coordinate sign; proceeds with one eye only |
| **No globe CC found** | Prints `SKIP`, returns empty list (case excluded from dataset) |
| **Non-diagonal affine post-RAS** | Prints WARNING, continues; spacing extracted via column norms instead of `np.diagonal` |
| **LCC search returns no foreground** | Falls back to globe centroid, skips LCC cleanup (prints WARN) |
| **Significant strip during LCC cleanup** (>5%) | Prints info message for QC review; expected when 80mm cube crosses midline |
| **Deployment obs/dense crop-origin mismatch** | `_deployment_index_shift` corrects it; without the fix, OS is grossly misplaced (the known bug) |
| **Oblique acquisition (axis 0 ≈ through-plane)** | The OS flip mirrors axis 0; if this is the through-plane axis, the flip is in the "wrong" direction. Flagged by `dataset.py`'s near-empty-overlap guard at bank ingest |
| **Patch extends beyond volume boundary** | `compute_crop_slices` clamps to volume shape; `place_patch_in_volume` clips at boundaries |
| **Two eyes with bridging fat** | Midplane clip prevents bbox from crossing; LCC still selects the correct component even if one voxel bridges |
| **Label scheme detection failure** | `detect_label_scheme` returns `"unknown"`, `align_single_case` raises `ValueError` → batch continues via try/except |
| **MRI input** | Must be pre-converted to CT label convention by `relabel_mri.py` BEFORE reaching canonical_align |

---

*Report generated by codebase audit. Date: 2026-07-20.*
