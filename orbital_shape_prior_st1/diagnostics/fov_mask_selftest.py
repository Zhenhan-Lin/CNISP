"""
Self-tests for engine/fov_mask.py (project a FOV visible_box into a patch grid).

numpy-only synthetic-affine checks -- no data / model needed:
    - identity affines: mask == the box itself
    - subpatch_affine: cropping shifts only the origin (box tracks the crop)
    - axis permutation + flip source_affine: box maps to the permuted/flipped region
    - anisotropic scale between grids: physical box preserved
    - oblique source_affine: still exact (per-voxel rasterization, not corner transform)

Usage (run from anywhere):
    python orbital_shape_prior_st1/diagnostics/fov_mask_selftest.py

Loads fov_mask by file path so it needs only numpy (no engine/diagnostics __init__).
"""

import importlib.util
from pathlib import Path

import numpy as np

_FOV_MASK = Path(__file__).resolve().parents[1] / "engine" / "fov_mask.py"
_spec = importlib.util.spec_from_file_location("fov_mask", _FOV_MASK)
_fm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fm)
source_box_to_grid_mask = _fm.source_box_to_grid_mask
subpatch_affine = _fm.subpatch_affine


def _diag_affine(spacing, origin):
    A = np.eye(4)
    A[0, 0], A[1, 1], A[2, 2] = spacing
    A[:3, 3] = origin
    return A


def main():
    shape = (10, 12, 14)
    box = [[2, 6], [3, 9], [0, 14]]   # half-open; axis 2 uncut (full extent)

    # ── identity affines: mask == the box exactly ────────────────────────
    I = np.eye(4)
    m = source_box_to_grid_mask(shape, I, I, box)
    ref = np.zeros(shape, bool)
    ref[2:6, 3:9, 0:14] = True
    assert np.array_equal(m, ref), "identity affine must reproduce the box"
    assert m.sum() == 4 * 6 * 14
    print("identity: mask == box  (voxels:", int(m.sum()), ")")

    # ── subpatch_affine: crop shifts origin; box tracks into sub-grid ─────
    disk = _diag_affine([0.5, 0.5, 0.5], [-3.0, 1.0, 2.0])
    sub_lo = [1, 2, 3]
    A_sub = subpatch_affine(disk, sub_lo)
    assert np.allclose(A_sub[:3, :3], disk[:3, :3])                 # spacing unchanged
    assert np.allclose(A_sub[:3, 3], (disk @ np.array([1, 2, 3, 1]))[:3])
    # source == disk grid: a sub voxel v corresponds to disk voxel v+sub_lo, so the
    # box in disk-voxels appears shifted by -sub_lo in the sub grid.
    boxd = [[4, 8], [5, 10], [6, 11]]
    m_sub = source_box_to_grid_mask((8, 8, 8), A_sub, disk, boxd)
    ref_sub = np.zeros((8, 8, 8), bool)
    ref_sub[4 - 1:8 - 1, 5 - 2:10 - 2, 6 - 3:11 - 3] = True
    assert np.array_equal(m_sub, ref_sub), "subpatch crop must shift the box by -sub_lo"
    print("subpatch: box shifts by -sub_lo  (voxels:", int(m_sub.sum()), ")")

    # ── permutation + flip source_affine ─────────────────────────────────
    # source axis order (x,y,z) -> world picks (y, -x, z); grid is identity world.
    src = np.zeros((4, 4))
    src[0, 1] = 1.0            # world_x = src_y
    src[1, 0] = -1.0           # world_y = -src_x
    src[2, 2] = 1.0            # world_z = src_z
    src[3, 3] = 1.0
    src[:3, 3] = [0.0, 9.0, 0.0]   # keep source voxels non-negative over the grid
    m_p = source_box_to_grid_mask(shape, I, src, box)
    # verify against a direct per-voxel computation
    Minv = np.linalg.inv(src)
    ref_p = np.zeros(shape, bool)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                sv = np.rint((Minv @ [i, j, k, 1])[:3]).astype(int)
                ref_p[i, j, k] = all(box[a][0] <= sv[a] < box[a][1] for a in range(3))
    assert np.array_equal(m_p, ref_p), "permuted/flipped source must match direct calc"
    assert m_p.sum() > 0
    print("perm+flip: matches direct per-voxel calc  (voxels:", int(m_p.sum()), ")")

    # ── anisotropic scale: physical box preserved across a 2x finer grid ──
    src_s = _diag_affine([1.0, 1.0, 1.0], [0.0, 0.0, 0.0])
    grid_s = _diag_affine([0.5, 0.5, 0.5], [0.0, 0.0, 0.0])   # 2x finer than source
    box_s = [[2, 5], [1, 4], [0, 20]]
    m_s = source_box_to_grid_mask((20, 20, 20), grid_s, src_s, box_s)
    # a source voxel spans 2 fine voxels; half-open [2,5) source -> world [1.5, 4.5)
    # -> fine voxels round to source in [2,5): fine idx ~ [3..9] along x. Just assert
    # the physical extent is ~2x the source-voxel count and non-empty.
    assert m_s.sum() > 0
    xs = np.where(m_s.any(axis=(1, 2)))[0]
    assert (xs.max() - xs.min() + 1) in (6, 7), (xs.min(), xs.max())  # ~3 src * 2
    print("aniso-scale: physical box preserved  (x-extent voxels:",
          int(xs.max() - xs.min() + 1), ")")

    # ── oblique source_affine: exact (rasterized per voxel) ──────────────
    theta = 0.3
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta), 0],
                  [0, 0, 1.0]])
    src_o = np.eye(4)
    src_o[:3, :3] = R
    src_o[:3, 3] = [5.0, 5.0, 0.0]
    m_o = source_box_to_grid_mask(shape, I, src_o, box)
    Minv_o = np.linalg.inv(src_o)
    ref_o = np.zeros(shape, bool)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                sv = np.rint((Minv_o @ [i, j, k, 1])[:3]).astype(int)
                ref_o[i, j, k] = all(box[a][0] <= sv[a] < box[a][1] for a in range(3))
    assert np.array_equal(m_o, ref_o), "oblique source must match direct per-voxel calc"
    print("oblique: matches direct per-voxel calc  (voxels:", int(m_o.sum()), ")")

    print("\nALL FOV-MASK SELF-TESTS PASSED")


if __name__ == "__main__":
    main()
