"""
Canonical alignment for orbital segmentation patches.

Pipeline per case:
    1. Load full-head segmentation NIfTI
    2. Separate OD/OS via whole-foreground connected component analysis
    3. For each eye:
        a. Compute globe centroid in world coordinates
        b. Crop a cubic patch (default 64mm) centered on globe centroid
        c. Reorient array to RAS+
        d. Validate affine is diagonal (required by downstream MLP)
        e. If OS: flip along sagittal axis → pseudo-OD
        f. Remap labels to canonical order: {0:BG, 1:ON, 2:Globe, 3:Fat, 4:Recti}
        g. Save aligned patch as NIfTI + metadata JSON

Why patch size is in mm, not voxels:
    The downstream implicit MLP works in physical coordinates (mm). It
    generates a coordinate grid from (spacing, patch_shape) at runtime.
    A 64mm patch with 0.5mm spacing has 128 voxels; with 1.0mm spacing
    it has 64 voxels. Both are valid — the MLP sees the same physical
    coordinate range [0, 64]mm. The coordinate convention follows
    Amiranashvili et al.: offset = spacing/2 (align_corners=False),
    so voxel centers are at spacing/2, 3*spacing/2, 5*spacing/2, ...

Data sources (Stage 1, CT only):
    1. QA-kept nnUNet predictions: review_checklist CSV, labels {1,2,3,4}
    2. CTONS_atlas_TBI manual GT: label-fusion scheme, labels {1,3,5,7}
    Both use the same CT label mapping (ON/Globe/Fat/Recti).
"""

import json
import glob
import pandas as pd
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from scipy import ndimage


# ── Label conventions ─────────────────────────────────────────────
# Input maps: names MUST match CANONICAL_LABELS keys exactly
NNUNET_MAP_CT = {1: "ON", 2: "Recti", 3: "Globe", 4: "Fat"}
LABELFUSION_MAP_CT = {1: "ON", 3: "Recti", 5: "Globe", 7: "Fat"}

# Canonical output: fixed across all downstream code
CANONICAL_LABELS = {"BG": 0, "ON": 1, "Globe": 2, "Fat": 3, "Recti": 4}
CANONICAL_LABEL_NAMES = {v: k for k, v in CANONICAL_LABELS.items()}
NUM_CLASSES = len(CANONICAL_LABELS)


@dataclass
class AlignmentMetadata:
    """Everything needed to invert the transform or analyze alignment quality."""
    source: str                 # "checklist" or "atlas"
    source_id: str              # subject ID or atlas filename
    eye: str                    # "OD" or "OS"
    casename: str               # unique key: "{source_id}_{eye}"

    original_nifti_path: str
    original_affine: list       # 4×4 nested list
    original_shape: list

    input_label_scheme: str     # "nnunet" or "labelfusion"

    globe_centroid_world: list  # [x, y, z] in RAS mm
    patch_size_mm: float
    crop_center_voxel: list
    crop_slices: list

    original_ornt: list
    target_ornt: str            # always "RAS"
    was_flipped: bool

    patch_spacing: list         # [sx, sy, sz] mm — varies per scan
    patch_voxel_shape: list     # [nx, ny, nz] — varies per scan

    globe_volume_mm3: float
    on_volume_mm3: float
    num_structures_found: int

    centroid_source: str          # "dense" or "sparse"  
    sparsify_centroid_params: dict  # {"axis": 2, "step": 4, "offset": 0} or null


# ── Label detection ───────────────────────────────────────────────
# Note: the label map is for CT scans, MRI scans has different map
def detect_label_scheme(data: np.ndarray) -> Tuple[str, Dict[int, str]]:
    labels = set(np.unique(data)) - {0}
    if not labels:
        return "empty", {}

    # Detect -1000 offset (atlas_labels convention: -1000 = BG, -999=ON, etc.)
    min_label = min(labels)
    if min_label < 0:
        offset = min_label  # e.g., -1000
        labels = {l - offset for l in labels} - {0}
        # Rebuild map with original (negative) keys
        if labels & {5, 7}:
            return "labelfusion", {
                offset + 1: "ON", offset + 3: "Recti",
                offset + 5: "Globe", offset + 7: "Fat",
            }
        if 2 in labels:
            return "nnunet", {
                offset + 1: "ON", offset + 2: "Recti",
                offset + 3: "Globe", offset + 4: "Fat",
            }
        return "empty", {}

    if 2 in labels:
        return "nnunet", NNUNET_MAP_CT
    if labels & {5, 7}:
        return "labelfusion", LABELFUSION_MAP_CT
    return "labelfusion", LABELFUSION_MAP_CT


def remap_to_canonical(data: np.ndarray, input_map: Dict[int, str]) -> np.ndarray:
    out = np.zeros_like(data)
    for input_label, structure_name in input_map.items():
        canonical_label = CANONICAL_LABELS[structure_name]
        out[data == input_label] = canonical_label
    return out


# ── OD/OS separation ─────────────────────────────────────────────

def separate_eyes(data, affine, globe_label, min_vox=50):
    globe_mask = (data == globe_label).astype(np.uint8)  # patient's globe should exist
    struct_26 = ndimage.generate_binary_structure(3, 3)
    labeled, n_cc = ndimage.label(globe_mask, structure=struct_26) # get two connected components

    eyes = []
    for cc_id in range(1, n_cc + 1):
        cc = (labeled == cc_id)
        nvox = int(cc.sum())
        if nvox < min_vox:
            continue
        c_vox = np.array(ndimage.center_of_mass(cc))
        c_world = (affine @ np.append(c_vox, 1.0))[:3]
        eyes.append({"centroid_voxel": c_vox, "centroid_world": c_world, "nvox": nvox})

    # Sort by L-R world coordinate, accounting for affine orientation
    axcodes = nib.aff2axcodes(affine)
    lr_axis = next((i for i, c in enumerate(axcodes) if c in ('R', 'L')), 0)
    sign = 1 if axcodes[lr_axis] == 'R' else -1

    if len(eyes) < 2:
        eyes[0]["eye"] = "OD" if eyes[0]["centroid_world"][lr_axis] * sign > 0 else "OS"
        return eyes

    # Filter to 2 largest CCs
    eyes = sorted(eyes, key=lambda e: e["nvox"], reverse=True)[:2]
    # Assign OD/OS by laterality
    eyes.sort(key=lambda e: e["centroid_world"][lr_axis] * sign, reverse=True)

    eyes[0]["eye"] = "OD"
    eyes[1]["eye"] = "OS"
    return eyes


# ── Crop + reorient + validate ────────────────────────────────────

def compute_crop_slices(centroid_voxel, volume_shape, patch_size_mm, voxel_sizes):
    half_vox = np.round((patch_size_mm / 2.0) / voxel_sizes).astype(int)
    center = np.round(centroid_voxel).astype(int)
    return [
        [int(max(0, center[ax] - half_vox[ax])),
         int(min(volume_shape[ax], center[ax] + half_vox[ax]))]
        for ax in range(3)
    ]


def reorient_to_ras(data, affine):
    orig_ornt = nib.orientations.io_orientation(affine)
    target_ornt = nib.orientations.axcodes2ornt("RAS")
    transform = nib.orientations.ornt_transform(orig_ornt, target_ornt)
    reoriented = nib.orientations.apply_orientation(data, transform)
    new_affine = affine @ nib.orientations.inv_ornt_aff(transform, data.shape)
    return reoriented, new_affine, list(nib.orientations.ornt2axcodes(orig_ornt))


def validate_diagonal_affine(affine, casename, tol=0.01):
    """
    After RAS reorientation, the 3×3 linear part should be diagonal with
    positive values. This mirrors Amiranashvili's is_matrix_scaling_and_transform.
    If not diagonal, spacing extraction via np.diagonal() would be wrong.
    """
    linear = affine[:3, :3]
    off_diag = np.abs(linear - np.diag(np.diagonal(linear))).max()
    if off_diag > tol:
        print(f"  WARN {casename}: off-diagonal={off_diag:.4f} after RAS reorient")
        return False
    if np.any(np.diagonal(linear)[:3] <= 0):
        print(f"  WARN {casename}: non-positive diagonal after RAS reorient")
        return False
    return True


def flip_os_to_od(data, affine):
    flipped = np.flip(data, axis=0).copy()
    fa = affine.copy()
    fa[:, 0] *= -1
    fa[:3, 3] += affine[:3, 0] * (data.shape[0] - 1)
    return flipped, fa


# ── Single-case ───────────────────────────────────────────────────

def align_single_case(seg_path, source_id, source="checklist", patch_size_mm=64.0, sparsify_for_centroid=True, sparsify_axis=2, sparsify_step=4, sparsify_offset=0):
    img = nib.load(seg_path)
    data = np.asarray(img.dataobj, dtype=np.int32)
    affine = img.affine.copy()

    scheme_name, label_map = detect_label_scheme(data)

    globe_lbl = next((l for l, n in label_map.items() if n == "Globe"), None)
    eyes = separate_eyes(data, affine, globe_lbl)

    voxel_sizes = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))
    results = []

    for eye_info in eyes:
        casename = f"{source_id}_{eye_info['eye']}"

        if sparsify_for_centroid:
            sparse_c = _sparse_globe_centroid(
                data, affine, globe_lbl,
                eye_info["centroid_voxel"],  # dense centroid as search anchor
                patch_size_mm, voxel_sizes,
                sparsify_axis, sparsify_step, sparsify_offset,
            )
            if sparse_c is not None:
                eye_info["centroid_voxel"] = sparse_c[0]
                eye_info["centroid_world"] = sparse_c[1]

        crop_sl = compute_crop_slices(
            eye_info["centroid_voxel"], data.shape, patch_size_mm, voxel_sizes
        )
        patch = data[
            crop_sl[0][0]:crop_sl[0][1],
            crop_sl[1][0]:crop_sl[1][1],
            crop_sl[2][0]:crop_sl[2][1],
        ].copy()

        crop_offset = np.array([s[0] for s in crop_sl], dtype=float)
        pa = affine.copy()
        pa[:3, 3] += affine[:3, :3] @ crop_offset

        patch = remap_to_canonical(patch, label_map)
        patch, pa, orig_ornt = reorient_to_ras(patch, pa)
        validate_diagonal_affine(pa, casename)

        was_flipped = (eye_info["eye"] == "OS")
        if was_flipped:
            patch, pa = flip_os_to_od(patch, pa)

        sp = np.abs(np.diagonal(pa)[:3])  # spacing from diagonal affine
        vv = float(np.prod(sp))

        used_sparse = (sparsify_for_centroid and sparse_c is not None)

        meta = AlignmentMetadata(
            source=source, source_id=str(source_id),
            eye=eye_info["eye"], casename=casename,
            original_nifti_path=str(seg_path),
            original_affine=affine.tolist(),
            original_shape=list(data.shape),
            input_label_scheme=scheme_name,
            globe_centroid_world=eye_info["centroid_world"].tolist(),
            patch_size_mm=patch_size_mm,
            crop_center_voxel=np.round(eye_info["centroid_voxel"]).astype(int).tolist(),
            crop_slices=crop_sl,
            original_ornt=orig_ornt, target_ornt="RAS",
            was_flipped=was_flipped,
            patch_spacing=sp.tolist(),
            patch_voxel_shape=list(patch.shape),
            globe_volume_mm3=float(np.sum(patch == CANONICAL_LABELS["Globe"])) * vv,
            on_volume_mm3=float(np.sum(patch == CANONICAL_LABELS["ON"])) * vv,
            centroid_source="sparse" if used_sparse else "dense",
            sparsify_centroid_params=(
                {"axis": sparsify_axis, "step": sparsify_step, "offset": sparsify_offset}
                if used_sparse else None
            ),
            num_structures_found=sum(
                1 for lbl in range(1, NUM_CLASSES) if np.any(patch == lbl)
            ),
        )
        results.append((patch, pa, meta))

    return results


# ── Dataset-level ─────────────────────────────────────────────────

def _collect_scan_list(checklist_csv=None, atlas_label_dir=None):
    """Build unified scan list from checklist + atlas."""
    scans = []

    if checklist_csv and Path(checklist_csv).exists():
        df = pd.read_csv(checklist_csv)
        if "keep" in df.columns:
            df = df[df["keep"] == True]  # noqa
        if "subject" in df.columns and "session" in df.columns:
            df = df.sort_values(["subject", "session"]).drop_duplicates(
                subset="subject", keep="first"
            )
        seg_col = next(
            (c for c in ["pred_path", "seg_path"] if c in df.columns), None
        )
        id_col = next(
            (c for c in ["subject", "study_id"] if c in df.columns), None
        )
        if seg_col and id_col:
            for _, row in df.iterrows():
                scans.append({
                    "seg_path": str(row[seg_col]),
                    "source_id": f"chk_{row[id_col]}",
                    "source": "checklist",
                })
        print(f"Checklist: {len(scans)} scans")

    if atlas_label_dir and Path(atlas_label_dir).exists():
        n0 = len(scans)
        for fp in sorted(glob.glob(str(Path(atlas_label_dir) / "*.nii.gz"))):
            fname = Path(fp).stem.replace(".nii", "")
            scans.append({
                "seg_path": fp,
                "source_id": f"atlas_{fname}",
                "source": "atlas",
            })
        print(f"Atlas: {len(scans) - n0} scans")

    print(f"Total: {len(scans)} scans")
    return scans


def align_dataset(checklist_csv=None, atlas_label_dir=None,
                  output_dir="aligned_patches", patch_size_mm=64.0):
    out_labels = Path(output_dir) / "labels"
    out_meta = Path(output_dir) / "metadata"
    out_labels.mkdir(parents=True, exist_ok=True)
    out_meta.mkdir(parents=True, exist_ok=True)

    scans = _collect_scan_list(checklist_csv, atlas_label_dir)
    all_meta = []

    for scan in scans:
        if not Path(scan["seg_path"]).exists():
            print(f"  SKIP {scan['source_id']}: not found")
            continue

        print(f"Processing {scan['source_id']}...")
        results = align_single_case(
            scan["seg_path"], scan["source_id"], scan["source"], patch_size_mm
        )
        for patch, pa, meta in results:
            nib.save(
                nib.Nifti1Image(patch.astype(np.uint8), pa),
                str(out_labels / f"{meta.casename}.nii.gz"),
            )
            with open(out_meta / f"{meta.casename}.json", "w") as f:
                json.dump(asdict(meta), f, indent=2)
            all_meta.append(meta)
            print(f"  {meta.casename}: voxels={meta.patch_voxel_shape} "
                  f"spacing={[f'{s:.2f}' for s in meta.patch_spacing]}mm "
                  f"structs={meta.num_structures_found}/4")

    print(f"\nTotal patches: {len(all_meta)}")

def _sparse_globe_centroid(data, affine, globe_lbl, dense_centroid_vox,
                           patch_size_mm, voxel_sizes, axis, step, offset):
    """
    Compute globe centroid from sparsified data within one eye's neighborhood.
    
    Strategy: use dense centroid to define a search region (the patch_size_mm 
    bounding box around this eye), then zero out non-observed slices within 
    that region and compute center_of_mass from surviving globe voxels.
    
    The zeroing-out approach is mathematically equivalent to extracting sparse 
    slices and mapping back: non-observed slices contribute 0 to both 
    numerator and denominator of center_of_mass.
    """
    # Generous bounding box around this eye (same size as the patch crop)
    half_vox = np.round((patch_size_mm / 2.0) / voxel_sizes).astype(int)
    center = np.round(dense_centroid_vox).astype(int)
    sl = tuple(
        slice(max(0, center[ax] - half_vox[ax]),
              min(data.shape[ax], center[ax] + half_vox[ax]))
        for ax in range(3)
    )
    
    # Globe mask restricted to this eye's region
    globe_in_eye = np.zeros_like(data, dtype=bool)
    globe_in_eye[sl] = (data[sl] == globe_lbl)
    
    # Zero out non-observed slices (simulate sparsification)
    sparse_globe = np.zeros_like(globe_in_eye)
    slc = [slice(None)] * 3
    for sid in range(offset, data.shape[axis], step):
        slc_copy = list(slc)
        slc_copy[axis] = sid
        sparse_globe[tuple(slc_copy)] = globe_in_eye[tuple(slc_copy)]
    
    if sparse_globe.sum() == 0:
        return None
    
    c_vox = np.array(ndimage.center_of_mass(sparse_globe))
    c_world = (affine @ np.append(c_vox, 1.0))[:3]
    return c_vox, c_world