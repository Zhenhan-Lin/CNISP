"""
Native-space mapping: reverse canonical alignment to place predictions
back into the original full-head volume coordinate system.

Inverse pipeline per eye patch:
    canonical pred → un-flip (if OS) → un-reorient (RAS → original) →
    remap labels → place at crop_slices in full volume

When both OD and OS exist for the same source, they are merged into
one volume (they occupy non-overlapping regions).

Usage from other modules:
    from engine.native_mapping import map_results_to_native
    native_paths = map_results_to_native(results, meta_dir, output_dir)
"""

import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np


# ── Inverse transform primitives ─────────────────────────────────

def reverse_flip(data: np.ndarray) -> np.ndarray:
    """Reverse OS→OD flip (axis 0 in RAS = sagittal)."""
    return np.flip(data, axis=0).copy()


def reverse_reorient(data: np.ndarray, original_ornt_codes: list) -> np.ndarray:
    """Reverse RAS reorientation back to original orientation."""
    ras_ornt = nib.orientations.axcodes2ornt("RAS")
    orig_ornt = nib.orientations.axcodes2ornt(tuple(original_ornt_codes))
    transform = nib.orientations.ornt_transform(ras_ornt, orig_ornt)
    return nib.orientations.apply_orientation(data, transform)


def place_patch_in_volume(
    patch: np.ndarray,
    original_shape: list,
    crop_slices: list,
) -> np.ndarray:
    """Place a cropped patch back into a full-size zero volume."""
    full = np.zeros(original_shape, dtype=patch.dtype)
    i0, i1 = crop_slices[0]
    j0, j1 = crop_slices[1]
    k0, k1 = crop_slices[2]
    pi = min(patch.shape[0], i1 - i0)
    pj = min(patch.shape[1], j1 - j0)
    pk = min(patch.shape[2], k1 - k0)
    full[i0:i0+pi, j0:j0+pj, k0:k0+pk] = patch[:pi, :pj, :pk]
    return full


def remap_canonical_to_original(data: np.ndarray, meta: dict) -> np.ndarray:
    """
    Remap canonical labels {0,1,2,3,4} back to original label scheme,
    so the _cnisp output matches the GT label values.
    """
    scheme = meta["input_label_scheme"]

    if scheme == "nnunet":
        # canonical {1:ON,2:Globe,3:Fat,4:Recti} → nnunet CT {1:ON,2:Recti,3:Globe,4:Fat}
        remap = {0: 0, 1: 1, 2: 3, 3: 4, 4: 2}
    elif scheme == "labelfusion":
        # canonical → labelfusion {1:ON, 3:Recti, 5:Globe, 7:Fat}
        remap = {0: 0, 1: 1, 2: 5, 3: 7, 4: 3}
        # Detect -1000 offset (atlas convention)
        orig_path = meta.get("original_nifti_path", "")
        if orig_path and Path(orig_path).exists():
            orig_data = np.asarray(nib.load(orig_path).dataobj, dtype=np.int32)
            min_label = int(np.min(orig_data))
            if min_label < 0:
                remap = {0: min_label, 1: min_label+1, 2: min_label+5,
                         3: min_label+7, 4: min_label+3}
    else:
        # Unknown scheme: identity remap. canonical_align labels detection
        # should always have set one of the known schemes; an "unknown"
        # value here means the original scheme could not be inferred,
        # so the output will not be label-compatible with the source GT.
        warnings.warn(
            f"native_mapping: unknown input_label_scheme={scheme!r} for "
            f"{meta.get('casename', '?')}; passing canonical labels through "
            f"unchanged (output will NOT match the original label scheme).",
            stacklevel=2,
        )
        remap = {i: i for i in range(5)}

    out = np.zeros_like(data)
    for canon, orig in remap.items():
        out[data == canon] = orig

    # Handle obs-vs-recon offset labels (11-14 = reconstructed versions of 1-4)
    # Apply the same remapping with offset preserved
    recon_offset = 10
    for canon, orig in remap.items():
        if canon == 0:
            continue
        # Map offset labels: e.g., canonical 11 → original_label + recon_offset
        offset_canon = canon + recon_offset
        offset_orig = orig + recon_offset if orig != 0 else 0
        out[data == offset_canon] = offset_orig

    return out


# ── Single-eye inverse mapping ───────────────────────────────────

def invert_alignment_single_eye(
    pred_patch: np.ndarray,
    meta: dict,
) -> np.ndarray:
    """
    Reverse canonical alignment for one eye patch.
    Returns the patch placed into a full-size volume.
    """
    patch = pred_patch.copy()

    # 1. Remap labels
    patch = remap_canonical_to_original(patch, meta)

    # 2. Un-flip
    if meta["was_flipped"]:
        patch = reverse_flip(patch)

    # 3. Un-reorient
    patch = reverse_reorient(patch, meta["original_ornt"])

    # 4. Place into full volume
    full = place_patch_in_volume(patch, meta["original_shape"], meta["crop_slices"])
    return full


# ── Merge eyes and save ──────────────────────────────────────────

def merge_and_save_native(
    eye_volumes: List[np.ndarray],
    reference_meta: dict,
    output_path: Path
):
    """
    Merge OD + OS full volumes and save as NIfTI.
    Non-zero voxels from each eye overwrite the merged volume
    (OD and OS occupy different spatial regions, so no conflict).
    """
    merged = np.zeros(reference_meta["original_shape"], dtype=np.int16)
    for vol in eye_volumes:
        mask = vol != 0
        # For offset schemes (e.g., -1000 based), detect actual background
        uniq = np.unique(vol)
        if len(uniq) > 1 and int(uniq[0]) < 0:
            mask = vol != int(uniq[0])
        merged[mask] = vol[mask]

    affine = np.array(reference_meta["original_affine"])
    nib.save(nib.Nifti1Image(merged, affine), str(output_path))
    return output_path


# ── High-level: map a batch of results to native space ────────────

def map_results_to_native(
    results: List[dict],
    meta_dir: Path,
    output_dir: Path,
    suffix: str = "_cnisp",
) -> List[Path]:
    """
    Map inference results back to native space.

    Args:
        results: list of dicts from infer_single_case, each with
            "casename", "pred_class_map", etc.
        meta_dir: directory containing alignment metadata JSONs
        output_dir: where to save _cnisp.nii.gz files

    Returns:
        list of output file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group results by source_id
    source_groups = defaultdict(list)
    for r in results:
        casename = r["casename"]
        meta_path = Path(meta_dir) / f"{casename}.json"
        if not meta_path.exists():
            print(f"  WARN: metadata not found for {casename}, skipping native mapping")
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        source_groups[meta["source_id"]].append((r, meta))

    output_paths = []
    for source_id, items in sorted(source_groups.items()):
        ref_meta = items[0][1]

        eye_volumes = []
        for result, meta in items:
            full_vol = invert_alignment_single_eye(result["pred_class_map"], meta)
            eye_volumes.append(full_vol)

        # Output filename: original stem + _cnisp
        orig_name = Path(ref_meta["original_nifti_path"]).name
        stem = orig_name.replace(".nii.gz", "").replace(".nii", "")
        out_path = output_dir / f"{stem}{suffix}.nii.gz"

        merge_and_save_native(eye_volumes, ref_meta, out_path)
        n_eyes = len(items)
        print(f"  {source_id}: {n_eyes} eye(s) → {out_path.name}")
        output_paths.append(out_path)

    return output_paths


# ── Isotropic patch export (with correct physical affine) ─────────

def map_iso_results_to_native(
    results: List[dict],
    meta_dir: Path,
    output_dir: Path,
    suffix: str = "_cnisp_iso",
) -> List[Path]:
    """
    Save isotropic predictions merged into full-volume at isotropic spacing.

    Creates a new full-head volume where all axes use the minimum (in-plane)
    spacing. OD and OS are merged into one file per source, matching
    native_space output convention.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by source_id
    source_groups = defaultdict(list)
    for r in results:
        casename = r["casename"]
        meta_path = Path(meta_dir) / f"{casename}.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        source_groups[meta["source_id"]].append((r, meta))

    output_paths = []
    for source_id, items in sorted(source_groups.items()):
        ref_meta = items[0][1]
        original_affine = np.array(ref_meta["original_affine"])
        original_shape = ref_meta["original_shape"]

        # Compute isotropic spacing and new volume shape
        orig_spacing = np.sqrt(np.sum(original_affine[:3, :3] ** 2, axis=0))
        iso_sp = float(orig_spacing.min())  # use in-plane spacing for all axes
        iso_shape = [int(round(original_shape[ax] * orig_spacing[ax] / iso_sp))
                     for ax in range(3)]

        # Build isotropic affine (same origin and orientation, different spacing)
        direction = original_affine[:3, :3] / orig_spacing  # unit direction vectors
        iso_affine = np.eye(4)
        iso_affine[:3, :3] = direction * iso_sp
        iso_affine[:3, 3] = original_affine[:3, 3]  # same origin

        merged = np.zeros(iso_shape, dtype=np.int32)

        for result, meta in items:
            pred_iso = result.get("pred_class_map_iso")
            if pred_iso is None:
                continue

            patch = pred_iso.copy()

            # Remap labels
            patch = remap_canonical_to_original(patch, meta)

            # Un-flip
            if meta["was_flipped"]:
                patch = reverse_flip(patch)

            # Un-reorient
            patch = reverse_reorient(patch, meta["original_ornt"])

            # Convert original crop_slices to isotropic voxel coordinates
            crop_slices_iso = []
            for ax in range(3):
                lo_phys = meta["crop_slices"][ax][0] * orig_spacing[ax]
                hi_phys = meta["crop_slices"][ax][1] * orig_spacing[ax]
                lo_iso = int(round(lo_phys / iso_sp))
                hi_iso = int(round(hi_phys / iso_sp))
                crop_slices_iso.append([lo_iso, hi_iso])

            # Place patch into isotropic volume
            for ax in range(3):
                lo, hi = crop_slices_iso[ax]
                p_size = patch.shape[ax]
                slot_size = hi - lo
                # Use the smaller of patch size and slot size
                use = min(p_size, slot_size, iso_shape[ax] - lo)
                crop_slices_iso[ax] = [lo, lo + use]

            i0, i1 = crop_slices_iso[0]
            j0, j1 = crop_slices_iso[1]
            k0, k1 = crop_slices_iso[2]
            pi = i1 - i0
            pj = j1 - j0
            pk = k1 - k0

            sub_patch = patch[:pi, :pj, :pk]
            mask = sub_patch != 0
            # For offset labels (e.g., -1000 based)
            uniq = np.unique(sub_patch)
            if len(uniq) > 1 and int(uniq[0]) < 0:
                mask = sub_patch != int(uniq[0])

            merged[i0:i1, j0:j1, k0:k1][mask] = sub_patch[mask]

        # Output filename
        orig_name = Path(ref_meta["original_nifti_path"]).name
        stem = orig_name.replace(".nii.gz", "").replace(".nii", "")
        out_path = output_dir / f"{stem}{suffix}.nii.gz"

        nib.save(nib.Nifti1Image(merged.astype(np.int16), iso_affine), str(out_path))
        n_labels = len(set(np.unique(merged))) - 1
        print(f"  {source_id}: {len(items)} eye(s) → {out_path.name} "
              f"(shape={iso_shape}, labels={n_labels})")
        output_paths.append(out_path)

    return output_paths