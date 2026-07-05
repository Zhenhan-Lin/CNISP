"""INTENTIONALLY BUGGY native-space mapping (pre-fix / pre-8540137 behaviour).

This module reproduces the CNISP native + iso mapping EXACTLY as it was BEFORE
the "deployment index shift" fix (commit 8540137, "god damn I cannot believe I
find another bug in my repo shit"). It exists ONLY for the rollback / ablation
experiment: regenerate the OLD (buggy) CNISP prelabels so the impact of the
native-mapping fix can be measured in isolation. DO NOT use it in production.

The bug
-------
The 64 mm prediction sub-patch is placed at ``sub_crop_lo_vox_dense`` with NO
re-framing to the OBSERVED input patch's crop origin. In deployment
(``nnunet_pred`` mode) the observed input patch and the dense target patch are
two DIFFERENT canonical crops of the same head that do NOT share a world origin.
The naive inverse therefore misplaces each eye by the crop-origin difference,
and for OS the axis-0 flip mirrors that error (OD looked ~fine while OS was
grossly off, worsening as step_size grows the sparse-vs-dense centroid gap).

The correct module (``engine.native_mapping``) fixes this via
``reconstruct_canonical_patch_affine`` / ``_deployment_index_shift`` applied
through the ``observed_meta`` / ``observed_meta_path_for`` arguments.

Drop-in compatibility
----------------------
The three functions below keep the CORRECT module's SIGNATURES -- they accept
``observed_meta`` / ``observed_meta_path_for`` -- so this module is a drop-in
swap for callers that pass them. They DELIBERATELY IGNORE those arguments; that
omission IS the reintroduced bug. Every UNCHANGED primitive (LCC cleanup, disk
placement, flip/reorient, label remap, merge) is imported verbatim from the
correct module so this file carries only the buggy delta.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np

# Unchanged primitives reused verbatim from the correct module (these did NOT
# change in the fix); only the placement of the sub-patch differs here.
from engine.native_mapping import (  # noqa: F401
    lcc_cleanup_with_warning,
    place_sub_patch_in_disk,
    place_patch_in_volume,
    reverse_flip,
    reverse_reorient,
    remap_canonical_to_original,
    merge_and_save_native,
    _extract_sub_crop_info,
)


# ── Single-eye inverse mapping (BUGGY: no observed-meta re-framing) ───

def invert_alignment_single_eye(
    pred_patch: np.ndarray,
    meta: dict,
    sub_crop_lo_vox_dense: Sequence[int],
    sub_crop_shape_vox_dense: Optional[Sequence[int]] = None,
    casename: Optional[str] = None,
    observed_meta: Optional[dict] = None,   # BUG: accepted for drop-in compat, IGNORED.
) -> np.ndarray:
    """Reverse canonical alignment for one eye -- WITHOUT the deployment shift.

    ``observed_meta`` is accepted so this is a drop-in replacement for the
    fixed function, but it is DELIBERATELY IGNORED: the sub-patch is placed at
    the raw ``sub_crop_lo_vox_dense`` (the pre-fix behaviour), which silently
    assumes the observed input patch and the dense target patch share a world
    origin. They do not in deployment -> the OS mirror / step-dependent
    misplacement bug is reproduced here on purpose.
    """
    if sub_crop_shape_vox_dense is not None:
        expected = tuple(int(v) for v in sub_crop_shape_vox_dense)
        if tuple(pred_patch.shape) != expected:
            raise ValueError(
                f"invert_alignment_single_eye: pred shape {pred_patch.shape} "
                f"!= sub_crop_shape_vox_dense {expected}; sub_crop sidecar "
                f"disagrees with the cached pred for "
                f"{casename or meta.get('casename', '?')}."
            )

    # (1) LCC clean-up on the prediction (no-op expected; WARNING if not).
    patch = lcc_cleanup_with_warning(
        pred_patch, casename=casename or meta.get("casename", "?"),
    ).astype(np.int16, copy=False)

    # (2) Place the 64 mm sub-patch inside the 80 mm canonical disk patch.
    #     BUG: no _deployment_index_shift(meta, observed_meta) correction.
    disk_shape = tuple(int(v) for v in meta["patch_voxel_shape"])
    disk_patch = place_sub_patch_in_disk(patch, disk_shape, sub_crop_lo_vox_dense)

    # (3) Un-flip   (4) Un-reorient
    if meta["was_flipped"]:
        disk_patch = reverse_flip(disk_patch)
    disk_patch = reverse_reorient(disk_patch, meta["original_ornt"])

    # (5) Place the disk patch into the full-head volume zeros.
    full = place_patch_in_volume(
        disk_patch, meta["original_shape"], meta["crop_slices"],
    )
    return full


# ── High-level: map a batch of results to native space (BUGGY) ───────

def map_results_to_native(
    results: List[dict],
    meta_dir: Path,
    output_dir: Path,
    suffix: str = "_cnisp",
    meta_path_for_casename: Optional[Callable[[str], Path]] = None,
    save_source_ids: Optional[set] = None,
    observed_meta_path_for: Optional[Callable[[str, int, int], Optional[Path]]] = None,  # IGNORED
) -> List[Path]:
    """Map results to native space -- pre-fix behaviour (no deployment shift).

    ``observed_meta_path_for`` is accepted for drop-in compatibility but is
    NEVER used, so every eye is inverted with the buggy
    ``invert_alignment_single_eye`` above.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _meta_path(casename: str) -> Path:
        if meta_path_for_casename is not None:
            return Path(meta_path_for_casename(casename))
        return Path(meta_dir) / f"{casename}.json"

    # Group results by source_id
    source_groups: Dict[str, List[Tuple[dict, dict]]] = defaultdict(list)
    for r in results:
        casename = r["casename"]
        meta_path = _meta_path(casename)
        if not meta_path.exists():
            print(f"  WARN: metadata not found for {casename} at {meta_path}, "
                  f"skipping native mapping")
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        source_groups[meta["source_id"]].append((r, meta))

    output_paths = []
    for source_id, items in sorted(source_groups.items()):
        if save_source_ids is not None and source_id not in save_source_ids:
            continue
        ref_meta = items[0][1]

        eye_volumes: List[np.ndarray] = []
        for result, meta in items:
            cn = result["casename"]
            sub_crop_lo, sub_crop_shape = _extract_sub_crop_info(result, cn)
            full_vol = invert_alignment_single_eye(
                result["pred_class_map"], meta,
                sub_crop_lo_vox_dense=sub_crop_lo,
                sub_crop_shape_vox_dense=sub_crop_shape,
                casename=cn,
            )
            eye_volumes.append(full_vol)

        orig_name = Path(ref_meta["original_nifti_path"]).name
        stem = orig_name.replace(".nii.gz", "").replace(".nii", "")
        out_path = output_dir / f"{stem}{suffix}.nii.gz"

        merge_and_save_native(eye_volumes, ref_meta, out_path)
        n_eyes = len(items)
        print(f"  {source_id}: {n_eyes} eye(s) -> {out_path.name}")
        output_paths.append(out_path)

    return output_paths


# ── Isotropic patch export (BUGGY: no deployment shift) ──────────────

def map_iso_results_to_native(
    results: List[dict],
    meta_dir: Path,
    output_dir: Path,
    suffix: str = "_cnisp_iso",
    iso_mm: Optional[float] = None,
    meta_path_for_casename: Optional[Callable[[str], Path]] = None,
    observed_meta_path_for: Optional[Callable[[str, int, int], Optional[Path]]] = None,  # IGNORED
) -> List[Path]:
    """Iso export -- pre-fix behaviour (no observed-meta re-framing).

    ``observed_meta_path_for`` is accepted for drop-in compatibility but is
    NEVER used: the iso sub-patch is laid into the iso disk at its raw
    ``sub_crop_lo`` position (the buggy placement), reproducing the OS mirror /
    step-dependent misplacement for the iso prelabels too.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _meta_path(casename: str) -> Path:
        if meta_path_for_casename is not None:
            return Path(meta_path_for_casename(casename))
        return Path(meta_dir) / f"{casename}.json"

    source_groups: Dict[str, List[Tuple[dict, dict]]] = defaultdict(list)
    for r in results:
        casename = r["casename"]
        meta_path = _meta_path(casename)
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

        # Iso spacing: FIXED iso_mm when provided (corrector: 0.5 -> head grid
        # defined by FOV + 0.5, independent of the source's native resolution),
        # else min over original (in-plane) spacing (legacy).
        orig_spacing = np.sqrt(np.sum(original_affine[:3, :3] ** 2, axis=0))
        iso_sp = float(iso_mm) if iso_mm is not None else float(orig_spacing.min())
        iso_shape = [int(round(original_shape[ax] * orig_spacing[ax] / iso_sp))
                     for ax in range(3)]
        direction = original_affine[:3, :3] / orig_spacing  # unit dirs
        iso_affine = np.eye(4)
        iso_affine[:3, :3] = direction * iso_sp
        iso_affine[:3, 3] = original_affine[:3, 3]

        merged = np.zeros(iso_shape, dtype=np.int16)

        for result, meta in items:
            pred_iso = result.get("pred_class_map_iso")
            if pred_iso is None:
                continue
            cn = result["casename"]
            sub_crop_lo, sub_crop_shape = _extract_sub_crop_info(result, cn)

            # BUG: no _deployment_index_shift(meta, observed_meta) here either.

            # The iso pred is a 64 mm sub-patch at iso spacing -- its
            # voxel shape differs from the disk-frame sub-patch (which is
            # at the disk patch's dense spacing). To place it in the iso
            # frame we recompute sub_crop positions in iso voxels by
            # converting through physical mm.
            disk_spacing = np.asarray(meta["patch_spacing"], dtype=np.float64)
            disk_shape = np.asarray(meta["patch_voxel_shape"], dtype=np.int64)
            sub_crop_lo_arr = np.asarray(sub_crop_lo, dtype=np.float64)

            # Sub-patch origin in disk-local mm (corner; voxel-center conv
            # adds an extra +spacing/2 but cancels with the iso voxel-
            # center sample, so corner-vs-corner mapping suffices).
            sub_origin_mm_in_disk = sub_crop_lo_arr * disk_spacing

            # Same physical region in iso-voxel coords inside the disk
            # patch's iso version. Disk patch in iso voxels:
            disk_iso_shape = np.maximum(
                np.round(disk_shape * disk_spacing / iso_sp).astype(np.int64),
                1,
            )
            sub_lo_iso_in_disk = np.round(sub_origin_mm_in_disk / iso_sp).astype(np.int64)

            patch = np.asarray(pred_iso).astype(np.int16, copy=False)
            patch = lcc_cleanup_with_warning(patch, casename=cn)

            # Compose: sub-patch (iso) -> disk patch (iso) -> iso volume.
            disk_iso = place_sub_patch_in_disk(
                patch, tuple(disk_iso_shape.tolist()), sub_lo_iso_in_disk.tolist(),
            )

            if meta["was_flipped"]:
                disk_iso = reverse_flip(disk_iso)
            disk_iso = reverse_reorient(disk_iso, meta["original_ornt"])

            # Disk patch position in iso voxels in the full iso volume:
            crop_slices_iso = []
            for ax in range(3):
                lo_phys = meta["crop_slices"][ax][0] * orig_spacing[ax]
                hi_phys = meta["crop_slices"][ax][1] * orig_spacing[ax]
                lo_iso = int(round(lo_phys / iso_sp))
                hi_iso = int(round(hi_phys / iso_sp))
                # Clamp to the iso volume bounds.
                lo_iso = max(0, min(lo_iso, iso_shape[ax]))
                hi_iso = max(lo_iso, min(hi_iso, iso_shape[ax]))
                crop_slices_iso.append([lo_iso, hi_iso])

            full_disk_iso = np.zeros(iso_shape, dtype=np.int16)
            i0, i1 = crop_slices_iso[0]
            j0, j1 = crop_slices_iso[1]
            k0, k1 = crop_slices_iso[2]
            pi = min(disk_iso.shape[0], i1 - i0)
            pj = min(disk_iso.shape[1], j1 - j0)
            pk = min(disk_iso.shape[2], k1 - k0)
            full_disk_iso[i0:i0+pi, j0:j0+pj, k0:k0+pk] = disk_iso[:pi, :pj, :pk]

            fg = full_disk_iso > 0
            merged[fg] = full_disk_iso[fg]

        merged = remap_canonical_to_original(merged, ref_meta).astype(np.int16)

        orig_name = Path(ref_meta["original_nifti_path"]).name
        stem = orig_name.replace(".nii.gz", "").replace(".nii", "")
        out_path = output_dir / f"{stem}{suffix}.nii.gz"

        nib.save(nib.Nifti1Image(merged, iso_affine), str(out_path))
        n_labels = len(set(np.unique(merged))) - 1
        print(f"  {source_id}: {len(items)} eye(s) -> {out_path.name} "
              f"(shape={iso_shape}, labels={n_labels})")
        output_paths.append(out_path)

    return output_paths
