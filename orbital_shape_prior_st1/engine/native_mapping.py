"""
Native-space mapping: reverse canonical alignment to place predictions
back into the original full-head volume coordinate system.

Inverse pipeline per eye (post inner-crop pipeline):

    64 mm pred sub-patch
        |
        | (1) LCC clean-up        -- defensive; should be a no-op because
        |                            canonical_align already enforces single-
        |                            eye patches and the MLP outputs one
        |                            connected shape per eye. Any voxels
        |                            stripped here trigger a WARNING since
        |                            it means the prior produced spurious
        |                            disconnected components.
        |
        | (2) Place into the 80 mm canonical disk patch using
        |     sub_crop_lo_vox_dense from the result dict (everything outside
        |     the sub-patch is zero).
        |
        | (3) Un-flip   (np.flip axis=0)  -- if was_flipped (OS)
        |
        | (4) Un-reorient (RAS → original) using meta["original_ornt"]
        |
        | (5) Place into the full-head volume using meta["crop_slices"]
        v
    Canonical-labels full-head per-eye volume

When both OD and OS exist for the same source, they are merged into one
volume. Because canonical_align's LCC cleanup guarantees a single-eye
patch and step (1) re-enforces it on the prediction side, two eyes can
NEVER overlap in the merged volume; the merge is therefore a simple
"foreground wins" union -- no need to consult either eye's metadata
during the merge step. Label remapping to the original scheme (nnunet /
labelfusion / atlas-offset) is deferred until AFTER the merge so the
union operates on the clean canonical {0..4} integer space.

Usage from other modules:
    from engine.native_mapping import map_results_to_native
    native_paths = map_results_to_native(results, meta_dir, output_dir)
"""

import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
from scipy import ndimage


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
    original_shape: Sequence[int],
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


def place_sub_patch_in_disk(
    sub_patch: np.ndarray,
    disk_shape: Sequence[int],
    sub_crop_lo_vox_dense: Sequence[int],
) -> np.ndarray:
    """Place a 64 mm sub-patch prediction into an 80 mm disk-patch zero volume.

    ``sub_crop_lo_vox_dense`` is the (lo_x, lo_y, lo_z) voxel offset of the
    sub-patch inside the disk patch's dense voxel grid (see
    ``engine.dataset.inner_crop_64mm``). Out-of-bounds clipping handles
    the rare edge case where the visible-LCC centroid sat near the disk
    patch boundary and the sub-patch was clamped during training.
    """
    disk = np.zeros(disk_shape, dtype=sub_patch.dtype)
    lo = [int(v) for v in sub_crop_lo_vox_dense]
    for ax in range(3):
        if lo[ax] >= disk_shape[ax] or (lo[ax] + sub_patch.shape[ax]) <= 0:
            # Entire sub-patch falls outside disk -- nothing to write.
            return disk
    sl_disk: List[slice] = []
    sl_sub: List[slice] = []
    for ax in range(3):
        d0 = max(lo[ax], 0)
        d1 = min(lo[ax] + sub_patch.shape[ax], int(disk_shape[ax]))
        s0 = d0 - lo[ax]
        s1 = s0 + (d1 - d0)
        sl_disk.append(slice(d0, d1))
        sl_sub.append(slice(s0, s1))
    disk[tuple(sl_disk)] = sub_patch[tuple(sl_sub)]
    return disk


def lcc_cleanup_with_warning(
    pred_class_map: np.ndarray,
    casename: str = "?",
) -> np.ndarray:
    """Keep only the largest connected component of the foreground.

    Under the inner-crop pipeline each eye's prediction is supposed to be
    one connected shape inside its 64 mm sub-patch. This helper enforces
    that invariant at inference time and prints a WARNING whenever it
    actually strips anything -- a non-empty strip means the prior emitted
    spurious foreground components which downstream consumers (paired
    Dice, merge) shouldn't see anyway.

    Returns a copy of ``pred_class_map`` with off-LCC voxels zeroed.
    """
    if pred_class_map.size == 0:
        return pred_class_map
    fg = pred_class_map > 0
    total_fg = int(fg.sum())
    if total_fg == 0:
        return pred_class_map

    struct = ndimage.generate_binary_structure(3, 3)  # 26-conn
    labeled, n_cc = ndimage.label(fg, structure=struct)
    if n_cc <= 1:
        return pred_class_map

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    lcc_id = int(sizes.argmax())
    lcc_count = int(sizes[lcc_id])
    stripped = total_fg - lcc_count
    if stripped == 0:
        return pred_class_map

    print(f"  [LCC] {casename}: {n_cc} foreground CCs in prediction; "
          f"stripped {stripped} voxels ({stripped / total_fg * 100:.1f}% "
          f"of fg) outside the largest CC ({lcc_count} voxels). The prior "
          f"should normally emit one CC per eye -- inspect the prediction "
          f"if this fires frequently.")
    out = pred_class_map.copy()
    out[(labeled != lcc_id) & fg] = 0
    return out


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
    sub_crop_lo_vox_dense: Sequence[int],
    sub_crop_shape_vox_dense: Optional[Sequence[int]] = None,
    casename: Optional[str] = None,
) -> np.ndarray:
    """
    Reverse canonical alignment for one eye's 64 mm sub-patch prediction.

    Args:
        pred_patch: [Nx, Ny, Nz] canonical-labels prediction in the 64 mm
            sub-patch frame.
        meta: alignment metadata json (one entry per disk patch).
        sub_crop_lo_vox_dense: voxel offset of the sub-patch inside the
            80 mm disk patch (dense voxel grid).
        sub_crop_shape_vox_dense: expected sub-patch voxel shape; only used
            for a sanity assert that ``pred_patch.shape`` matches.
        casename: optional name for the LCC WARNING message.

    Returns a CANONICAL-LABELS full-head volume (np.int16) with this eye's
    prediction placed at its native position. Label remapping to the
    original scheme is deferred until after OD/OS merge.
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


# ── Merge eyes and save ──────────────────────────────────────────

def merge_and_save_native(
    eye_volumes: List[np.ndarray],
    reference_meta: dict,
    output_path: Path,
):
    """
    Merge OD + OS canonical-labels full-volume renders, remap to the
    original label scheme, and save as NIfTI.

    Each ``eye_volumes[i]`` comes from ``invert_alignment_single_eye`` and
    is in CANONICAL labels {0..4} with zeros outside that eye's footprint
    (the LCC clean-up + single-eye disk patch invariant guarantees no
    contralateral voxels are written). The merge is therefore a simple
    "foreground union": any voxel where any eye is > 0 keeps that value.
    Since OD/OS footprints never overlap, the order of iteration does
    not change the result.

    Remapping (canonical → original scheme, possibly -1000 offset) runs
    once on the merged volume rather than per-eye, so the in-patch BG vs
    outside-patch sentinel ambiguity that bit the OLD merge is gone.
    """
    merged = np.zeros(reference_meta["original_shape"], dtype=np.int16)
    for vol in eye_volumes:
        if vol is None or not vol.any():
            continue
        # ``vol`` is in canonical labels with 0 = BG everywhere (both in-
        # patch and outside-patch). Foreground is strictly ``vol > 0``.
        fg = vol > 0
        merged[fg] = vol[fg]

    merged = remap_canonical_to_original(merged, reference_meta)

    affine = np.array(reference_meta["original_affine"])
    nib.save(nib.Nifti1Image(merged.astype(np.int16), affine), str(output_path))
    return output_path


# ── High-level: map a batch of results to native space ────────────

def _extract_sub_crop_info(result: dict, casename: str) -> Tuple[List[int], List[int]]:
    """Pull sub_crop position + shape out of a result dict, with a clear
    error when they're missing (= old cache without inner-crop sidecar)."""
    lo = result.get("sub_crop_lo_vox_dense")
    sh = result.get("sub_crop_shape_vox_dense")
    if lo is None or sh is None:
        raise ValueError(
            f"native_mapping: result for {casename} is missing "
            f"sub_crop_lo_vox_dense / sub_crop_shape_vox_dense. This "
            f"comes from the inner-crop pipeline (see engine/dataset.py "
            f"and diagnostics/resolution_sweep.py); old caches without "
            f"the sub_crop sidecar must be regenerated."
        )
    return [int(v) for v in lo], [int(v) for v in sh]


def map_results_to_native(
    results: List[dict],
    meta_dir: Path,
    output_dir: Path,
    suffix: str = "_cnisp",
    meta_path_for_casename: Optional[Callable[[str], Path]] = None,
    save_source_ids: Optional[set] = None,
) -> List[Path]:
    """
    Map inference results back to native space.

    Args:
        results: list of dicts from the resolution sweep, each with
            ``casename``, ``pred_class_map``, ``sub_crop_lo_vox_dense``,
            ``sub_crop_shape_vox_dense``.
        meta_dir: directory containing alignment metadata JSONs. Used as
            ``meta_dir/<casename>.json`` unless
            ``meta_path_for_casename`` is provided.
        output_dir: where to save ``_cnisp.nii.gz`` files.
        meta_path_for_casename: optional resolver ``(casename) -> Path``
            to support mixed metadata trees (Option C nnunet_pred mode
            uses ``metadata/`` for atlas cases and
            ``metadata_dataset835/`` for chk_* cases since those two
            share a Dice frame with different canonical crops). When
            None we fall back to ``meta_dir`` for every case so the
            legacy single-tree call sites keep working.
        save_source_ids: optional whitelist of ``source_id``s to actually
            write native masks for. ``None`` (default) writes all sources
            (back-compat). When provided, sources not in the set are
            skipped here -- their canonical-space Dice still lives in
            ``sweep_results.pkl`` / ``test_results.csv`` (the aggregate
            reads from there), so dropping the full-head mask only saves
            disk, not information.

    Returns:
        list of output file paths
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
        print(f"  {source_id}: {n_eyes} eye(s) → {out_path.name}")
        output_paths.append(out_path)

    return output_paths


# ── Isotropic patch export (with correct physical affine) ─────────

def map_iso_results_to_native(
    results: List[dict],
    meta_dir: Path,
    output_dir: Path,
    suffix: str = "_cnisp_iso",
    iso_mm: Optional[float] = None,
    meta_path_for_casename: Optional[Callable[[str], Path]] = None,
) -> List[Path]:
    """Save isotropic predictions merged into a full-volume at isotropic spacing.

    Under the inner-crop pipeline ``result["pred_class_map_iso"]`` is a
    64 mm sub-patch at isotropic spacing. We compose two layers exactly
    like the regular ``map_results_to_native`` does, but in iso voxel
    coordinates: sub-patch → 80 mm disk patch (in iso voxels) → full-
    head iso volume.

    OD/OS are merged with the same "foreground union" rule as the
    non-iso path. Remap to the original label scheme runs once on the
    merged iso volume.

    Args:
        iso_mm: FIXED isotropic spacing (mm) for the output head grid. When
            None, falls back to ``min(original per-axis spacing)`` (legacy).
            The nnUNet-C corrector passes ``0.5`` so the iso head grid is
            defined by the head FOV + 0.5 spacing and does NOT depend on the
            source's original resolution.
        meta_path_for_casename: optional ``(casename) -> Path`` resolver so
            Option C (atlas in ``metadata/``, chk_* in ``metadata_dataset835/``)
            resolves each case's metadata correctly. Falls back to
            ``meta_dir/<casename>.json`` when None (mirrors map_results_to_native).
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

            # Compose: sub-patch (iso) → disk patch (iso) → iso volume.
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
        print(f"  {source_id}: {len(items)} eye(s) → {out_path.name} "
              f"(shape={iso_shape}, labels={n_labels})")
        output_paths.append(out_path)

    return output_paths
