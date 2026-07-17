"""Anatomical plausibility metrics (computation layer).

Per-structure topology, cross-slice continuity, and shape regularity metrics
computed on multi-label segmentation masks. Designed for the two-layer
comparison:
  Layer 1: nnUNet raw pred vs CNISP pred (prior channel quality)
  Layer 2: arm B output vs Proposed output (cascade output quality)

Processing order per mask (critical for correctness):
  1. Load NIfTI multi-label volume
  2. Detect LR axis and through-plane axis; assert they differ
  3. Split volume into two eye halves along LR axis (multi-label!)
  4. Per eye half: remap to per-structure binary masks, then compute metrics

Depends on numpy, scipy.ndimage, nibabel. Optional: skimage (marching_cubes),
trimesh (curvature).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import ndimage

from simulation.evaluation.metrics import STRUCTURES, SCHEMES

PathLike = Union[str, Path]

# 26-connectivity structuring element for 3D connected components
_STRUCT_26 = np.ones((3, 3, 3), dtype=np.int32)


# ============================================================
# Axis detection
# ============================================================

def detect_through_plane_axis(spacing) -> int:
    """Through-plane axis = the axis with the largest voxel spacing."""
    return int(np.argmax(spacing))


def detect_lr_axis(affine: np.ndarray) -> int:
    """LR (left-right) array axis = axis most aligned with world-x (RAS)."""
    return int(np.argmax(np.abs(np.asarray(affine, dtype=float)[0, :3])))


# ============================================================
# Per-eye split
# ============================================================

def split_eyes(data: np.ndarray, affine: np.ndarray, spacing) -> List[Dict]:
    """Split a multi-label volume into two eye halves.

    Operates on the INTEGER multi-label volume (not per-structure binary).
    Split index = median LR-index of all foreground voxels.

    Returns list of dicts: [{"eye": "OD"/"OS", "volume": sub_array}, ...].
    OD/OS assigned by sign of world-x centroid (positive = patient left = OD).
    """
    lr_axis = detect_lr_axis(affine)
    fg = np.argwhere(data > 0)
    if fg.size == 0:
        split_idx = data.shape[lr_axis] // 2
    else:
        split_idx = int(np.median(fg[:, lr_axis]))

    # Slice into two halves
    sl_lo = [slice(None)] * data.ndim
    sl_hi = [slice(None)] * data.ndim
    sl_lo[lr_axis] = slice(0, split_idx)
    sl_hi[lr_axis] = slice(split_idx, None)
    half_lo = data[tuple(sl_lo)]
    half_hi = data[tuple(sl_hi)]

    # Determine OD/OS by world-x sign of each half's centroid
    # World position of voxel index i along lr_axis:
    #   world_x contribution = affine[0, lr_axis] * i + affine[0, 3]
    direction = float(affine[0, lr_axis])
    origin_x = float(affine[0, 3])

    def _world_x_centroid(half_vol, start_idx):
        fg_half = np.argwhere(half_vol > 0)
        if fg_half.size == 0:
            mid = start_idx + half_vol.shape[lr_axis] / 2.0
        else:
            mid = start_idx + float(fg_half[:, lr_axis].mean())
        return direction * mid + origin_x

    cx_lo = _world_x_centroid(half_lo, 0)
    cx_hi = _world_x_centroid(half_hi, split_idx)

    # In RAS: positive x = patient Left. Anatomical OD = right eye = patient right
    # = negative world-x. OS = left eye = patient left = positive world-x.
    # Compare the two halves: the one with LARGER world-x centroid is more to
    # patient-left (OS); the other is more patient-right (OD).
    if cx_lo > cx_hi:
        eyes = [
            {"eye": "OS", "volume": half_lo},
            {"eye": "OD", "volume": half_hi},
        ]
    else:
        eyes = [
            {"eye": "OD", "volume": half_lo},
            {"eye": "OS", "volume": half_hi},
        ]

    return eyes


# ============================================================
# Metric 1: Topology
# ============================================================

def compute_topology_metrics(
    masks: Dict[str, np.ndarray],
    spacing,
    min_cc_voxels: int = 5,
) -> Tuple[Dict[str, Dict], Dict[Tuple[str, str], Dict]]:
    """Topology violation indicators per structure.

    For each structure binary mask:
      - num_cc: connected components (26-conn), ignoring CCs < min_cc_voxels
      - has_multi_cc: num_cc > 1
      - num_holes: internal holes (background enclosed by structure)
      - has_holes: num_holes > 0
      - volume_mm3: total volume

    Also computes pairwise mutual overlap between structures.

    The min_cc_voxels filter handles CNISP rasterization artifacts (tiny
    single-voxel islands from discretizing a continuous auto-decoder output).
    """
    voxel_vol = float(np.prod(spacing))
    per_struct: Dict[str, Dict] = {}

    for name in STRUCTURES:
        mask = masks.get(name)
        if mask is None or mask.size == 0:
            per_struct[name] = {
                "num_cc": 0, "has_multi_cc": False,
                "num_holes": 0, "has_holes": False,
                "volume_mm3": 0.0,
            }
            continue

        # Connected components (26-connectivity)
        labeled, num_cc_raw = ndimage.label(mask, structure=_STRUCT_26)

        # Filter out tiny CCs (rasterization artifacts)
        if num_cc_raw > 1 and min_cc_voxels > 1:
            cc_sizes = ndimage.sum(mask, labeled, range(1, num_cc_raw + 1))
            num_cc = int(np.sum(np.asarray(cc_sizes) >= min_cc_voxels))
            num_cc = max(num_cc, 1) if mask.any() else 0
        else:
            num_cc = num_cc_raw

        # Internal holes
        filled = ndimage.binary_fill_holes(mask)
        holes = filled & (~mask)
        _, num_holes = ndimage.label(holes, structure=_STRUCT_26)

        volume_mm3 = float(np.sum(mask)) * voxel_vol

        per_struct[name] = {
            "num_cc": int(num_cc),
            "has_multi_cc": num_cc > 1,
            "num_holes": int(num_holes),
            "has_holes": num_holes > 0,
            "volume_mm3": volume_mm3,
        }

    # Pairwise mutual overlap
    overlap: Dict[Tuple[str, str], Dict] = {}
    struct_names = [s for s in STRUCTURES if s in masks and masks[s] is not None]
    for i, sa in enumerate(struct_names):
        for sb in struct_names[i + 1:]:
            ov = int(np.sum(masks[sa] & masks[sb]))
            overlap[(sa, sb)] = {
                "overlap_voxels": ov,
                "overlap_mm3": float(ov) * voxel_vol,
            }

    return per_struct, overlap


# ============================================================
# Metric 2: Cross-slice continuity
# ============================================================

def compute_cross_slice_continuity(
    masks: Dict[str, np.ndarray],
    spacing,
    axis: int,
) -> Dict[str, Dict]:
    """Cross-slice continuity metrics per structure along the given axis.

    For each structure on consecutive nonempty slices:
      - centroid displacement (in-plane, mm)
      - area relative change
      - gap count (slice present -> absent -> present)
    """
    in_plane_axes = [a for a in range(3) if a != axis]
    pixel_area = float(spacing[in_plane_axes[0]]) * float(spacing[in_plane_axes[1]])

    results: Dict[str, Dict] = {}

    for name in STRUCTURES:
        mask = masks.get(name)
        if mask is None or mask.size == 0:
            results[name] = {
                "max_centroid_jump_mm": 0.0,
                "mean_centroid_jump_mm": 0.0,
                "max_area_rel_change": 0.0,
                "mean_area_rel_change": 0.0,
                "num_gaps": 0,
                "num_nonempty_slices": 0,
            }
            continue

        n_slices = mask.shape[axis]
        centroids: List[Optional[Tuple[float, float]]] = []
        areas: List[Optional[float]] = []

        for z in range(n_slices):
            slc = [slice(None)] * 3
            slc[axis] = z
            slice_mask = mask[tuple(slc)]

            if not slice_mask.any():
                centroids.append(None)
                areas.append(None)
                continue

            coords = np.argwhere(slice_mask)
            centroid_mm = (
                float(coords[:, 0].mean()) * float(spacing[in_plane_axes[0]]),
                float(coords[:, 1].mean()) * float(spacing[in_plane_axes[1]]),
            )
            centroids.append(centroid_mm)
            areas.append(float(np.sum(slice_mask)) * pixel_area)

        # Compute consecutive-slice metrics
        centroid_jumps: List[float] = []
        area_changes: List[float] = []
        gap_count = 0
        prev_nonempty: Optional[int] = None

        for z in range(n_slices):
            if centroids[z] is not None:
                if prev_nonempty is not None:
                    if z - prev_nonempty > 1:
                        gap_count += 1

                    c_prev = centroids[prev_nonempty]
                    c_curr = centroids[z]
                    in_plane_dist = np.sqrt(
                        (c_curr[0] - c_prev[0]) ** 2
                        + (c_curr[1] - c_prev[1]) ** 2
                    )
                    centroid_jumps.append(float(in_plane_dist))

                    a_prev = areas[prev_nonempty]
                    a_curr = areas[z]
                    denom = max(a_prev, a_curr, 1e-6)
                    rel_change = abs(a_curr - a_prev) / denom
                    area_changes.append(float(rel_change))

                prev_nonempty = z

        num_nonempty = sum(1 for c in centroids if c is not None)

        results[name] = {
            "max_centroid_jump_mm": float(max(centroid_jumps)) if centroid_jumps else 0.0,
            "mean_centroid_jump_mm": float(np.mean(centroid_jumps)) if centroid_jumps else 0.0,
            "max_area_rel_change": float(max(area_changes)) if area_changes else 0.0,
            "mean_area_rel_change": float(np.mean(area_changes)) if area_changes else 0.0,
            "num_gaps": gap_count,
            "num_nonempty_slices": num_nonempty,
        }

    return results


# ============================================================
# Metric 3: Shape regularity (optional)
# ============================================================

def compute_shape_regularity(
    masks: Dict[str, np.ndarray],
    spacing,
) -> Dict[str, Dict]:
    """Surface smoothness: compactness (isoperimetric ratio) and curvature.

    Requires skimage for marching_cubes; trimesh for curvature (graceful skip).
    """
    try:
        from skimage.measure import marching_cubes
    except ImportError:
        return {s: {"surface_area_mm2": None, "compactness": None,
                    "mean_curvature_var": None} for s in STRUCTURES}

    results: Dict[str, Dict] = {}

    for name in STRUCTURES:
        mask = masks.get(name)
        if mask is None or int(np.sum(mask)) < 10:
            results[name] = {
                "surface_area_mm2": 0.0, "compactness": 0.0,
                "mean_curvature_var": None,
            }
            continue

        try:
            verts, faces, _, _ = marching_cubes(
                mask.astype(float), level=0.5, spacing=tuple(float(s) for s in spacing)
            )
        except Exception:
            results[name] = {
                "surface_area_mm2": 0.0, "compactness": 0.0,
                "mean_curvature_var": None,
            }
            continue

        # Surface area
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        sa = float(0.5 * np.sum(np.linalg.norm(cross, axis=1)))

        # Volume + compactness
        vol = float(np.sum(mask)) * float(np.prod(spacing))
        compactness = (36.0 * np.pi * vol ** 2) / (sa ** 3) if sa > 0 else 0.0

        # Mean curvature variance (optional)
        mcv = None
        try:
            import trimesh
            mesh = trimesh.Trimesh(vertices=verts, faces=faces)
            curvature = trimesh.curvature.discrete_mean_curvature_measure(
                mesh, mesh.vertices, radius=2.0
            )
            mcv = float(np.var(curvature))
        except (ImportError, Exception):
            pass

        results[name] = {
            "surface_area_mm2": sa,
            "compactness": compactness,
            "mean_curvature_var": mcv,
        }

    return results


# ============================================================
# Top-level per-case function
# ============================================================

def compute_case_plausibility(
    pred_path: PathLike,
    pred_scheme: str,
    offset_pred: int = 0,
    min_cc_voxels: int = 5,
    do_shape_reg: bool = False,
) -> List[Dict]:
    """Compute all plausibility metrics for one segmentation mask.

    Flow: load -> detect axes -> assert lr != tp -> split eyes ->
          per eye: binary_structures -> topology + continuity [+ shape reg].

    Returns a list of row dicts (one per eye per structure), or empty list
    if the case must be skipped (e.g. lr_axis == tp_axis).
    """
    import nibabel as nib

    path = Path(pred_path)
    img = nib.load(str(path))
    data = np.asarray(img.dataobj)
    if offset_pred:
        data = np.clip(data + offset_pred, 0, None)
    data = data.astype(np.int32)
    affine = np.asarray(img.affine, dtype=float)
    spacing = np.array(img.header.get_zooms()[:3], dtype=float)

    tp_axis = detect_through_plane_axis(spacing)
    lr_axis = detect_lr_axis(affine)

    if lr_axis == tp_axis:
        warnings.warn(
            f"LR axis ({lr_axis}) == through-plane axis ({tp_axis}) for "
            f"{path}; skipping this case (cannot split eyes and compute "
            f"continuity on the same axis).",
            stacklevel=2,
        )
        return []

    eyes = split_eyes(data, affine, spacing)

    rows: List[Dict] = []
    from simulation.evaluation.metrics import binary_structures

    for eye_info in eyes:
        eye_tag = eye_info["eye"]
        eye_vol = eye_info["volume"]

        masks = binary_structures(eye_vol, pred_scheme)

        topo, overlaps = compute_topology_metrics(masks, spacing, min_cc_voxels)
        continuity = compute_cross_slice_continuity(masks, spacing, axis=tp_axis)

        if do_shape_reg:
            shape_reg = compute_shape_regularity(masks, spacing)
        else:
            shape_reg = {s: {"surface_area_mm2": None, "compactness": None,
                             "mean_curvature_var": None} for s in STRUCTURES}

        for struct in STRUCTURES:
            row: Dict = {
                "eye": eye_tag,
                "structure": struct,
                # Topology
                "num_cc": topo[struct]["num_cc"],
                "has_multi_cc": topo[struct]["has_multi_cc"],
                "num_holes": topo[struct]["num_holes"],
                "has_holes": topo[struct]["has_holes"],
                "volume_mm3": topo[struct]["volume_mm3"],
                # Continuity
                "max_centroid_jump_mm": continuity[struct]["max_centroid_jump_mm"],
                "mean_centroid_jump_mm": continuity[struct]["mean_centroid_jump_mm"],
                "max_area_rel_change": continuity[struct]["max_area_rel_change"],
                "mean_area_rel_change": continuity[struct]["mean_area_rel_change"],
                "num_gaps": continuity[struct]["num_gaps"],
                "num_nonempty_slices": continuity[struct]["num_nonempty_slices"],
                # Shape regularity
                "surface_area_mm2": shape_reg[struct]["surface_area_mm2"],
                "compactness": shape_reg[struct]["compactness"],
                "mean_curvature_var": shape_reg[struct]["mean_curvature_var"],
            }

            # Pairwise overlaps for this structure
            for (sa, sb), ov_data in overlaps.items():
                if struct in (sa, sb):
                    other = sb if struct == sa else sa
                    row[f"overlap_with_{other}_voxels"] = ov_data["overlap_voxels"]

            rows.append(row)

    return rows


def build_plausibility_table(
    index: List[Dict],
    min_cc_voxels: int = 5,
    do_shape_reg: bool = False,
    save_csv: Optional[PathLike] = None,
    progress: bool = False,
):
    """Run plausibility metrics on all entries in a mask index subset.

    Each entry in ``index`` must have: case, arm, step, eff_res,
    pred_path, pred_scheme, offset_pred.

    Returns a pandas DataFrame with one row per (case, arm, step, eye, structure).
    """
    import time
    import pandas as pd

    recs: List[Dict] = []
    n = len(index)
    t0 = time.time()
    n_skipped = 0

    for i, it in enumerate(index, 1):
        if progress:
            print(
                f"[plausibility] {i}/{n}  {it.get('case')} / {it.get('arm')} "
                f"/ step{it.get('step')}  ({time.time() - t0:.0f}s elapsed)",
                file=sys.stderr, flush=True,
            )

        case_rows = compute_case_plausibility(
            it["pred_path"],
            it["pred_scheme"],
            offset_pred=it.get("offset_pred", 0),
            min_cc_voxels=min_cc_voxels,
            do_shape_reg=do_shape_reg,
        )

        if not case_rows:
            n_skipped += 1
            continue

        for r in case_rows:
            recs.append({
                "case": it["case"],
                "arm": it["arm"],
                "step": it["step"],
                "eff_res": it.get("eff_res"),
                **r,
            })

    if n_skipped:
        print(f"[plausibility] skipped {n_skipped}/{n} entries "
              f"(lr_axis == tp_axis or load failure)", file=sys.stderr)

    df = pd.DataFrame.from_records(recs)
    if save_csv and len(df) > 0:
        Path(save_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(str(save_csv), index=False)
    return df


__all__ = [
    "detect_through_plane_axis", "detect_lr_axis", "split_eyes",
    "compute_topology_metrics", "compute_cross_slice_continuity",
    "compute_shape_regularity", "compute_case_plausibility",
    "build_plausibility_table",
]
