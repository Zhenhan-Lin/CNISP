"""Project a FOV ``visible_box`` (source-CT voxel window) into a canonical patch grid.

The FOV-truncation builder records, per (case, pseudo-step), a ``visible_box`` =
per-source-axis half-open voxel windows ``[[x0,x1],[y0,y1],[z0,z1]]`` (a cut axis is
clipped, an uncut axis spans ``[0, shape]``). Voxels OUTSIDE that box were blanked to
air ("imaged-but-empty"); voxels inside are the acquired FOV.

To evaluate the CNISP fit ONLY inside the acquired FOV (plan C1: the geometric FOV
mask, NOT the segmentation), we need that box as a per-voxel mask on the canonical
patch grid the decoder queries. The mapping is a pure affine composition:

    world      = grid_affine @ [i, j, k, 1]          (patch voxel -> world mm)
    source_vox = inv(source_affine) @ world          (world mm -> source CT voxel)
    inside     = AND_axis( box_lo <= round(source_vox) < box_hi )

Canonical alignment only reorients (RAS) and resamples the source, both of which
preserve world coordinates, so ``inv(source_affine) @ grid_affine`` is the exact
patch-voxel -> source-voxel map -- valid even for an oblique ``source_affine`` (the
box is rasterized per voxel, not approximated by transforming corners).

numpy-only, no repo imports, so it is unit-testable with synthetic affines.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def subpatch_affine(disk_affine: np.ndarray, sub_lo_vox: Sequence[int]) -> np.ndarray:
    """Affine (voxel->world) of a sub-patch that is the voxel crop
    ``disk[sub_lo : sub_lo + shape]`` of a parent grid with ``disk_affine``.

    Cropping shifts only the origin: the direction/spacing block is unchanged and the
    new origin is the world position of the parent voxel ``sub_lo``.
    """
    disk_affine = np.asarray(disk_affine, dtype=np.float64)
    lo = np.asarray(sub_lo_vox, dtype=np.float64)
    out = disk_affine.copy()
    out[:3, 3] = (disk_affine @ np.append(lo, 1.0))[:3]
    return out


def source_box_to_grid_mask(
    grid_shape: Sequence[int],
    grid_affine: np.ndarray,
    source_affine: np.ndarray,
    visible_box: Sequence[Sequence[int]],
) -> np.ndarray:
    """Boolean mask over a target grid: True where the voxel maps into ``visible_box``.

    Args:
        grid_shape:    [3] target grid shape (the patch/sub-patch the decoder queries).
        grid_affine:   [4,4] target-voxel -> world mm.
        source_affine: [4,4] source-CT-voxel -> world mm (align metadata ``original_affine``).
        visible_box:   per-source-axis half-open ``[lo, hi)`` voxel windows (len == 3).

    Returns:
        bool ndarray of shape ``grid_shape`` (True == inside the acquired FOV).
    """
    grid_shape = tuple(int(s) for s in grid_shape)
    grid_affine = np.asarray(grid_affine, dtype=np.float64)
    source_affine = np.asarray(source_affine, dtype=np.float64)
    box = np.asarray(visible_box, dtype=np.float64)          # [3, 2]
    assert box.shape == (3, 2), f"visible_box must be 3x2, got {box.shape}"

    # Composed patch-voxel -> source-voxel affine (avoid two matmuls per voxel).
    M = np.linalg.inv(source_affine) @ grid_affine           # [4, 4]

    # Homogeneous index grid [N, 4] in (i, j, k, 1) order.
    ii, jj, kk = np.meshgrid(
        np.arange(grid_shape[0]), np.arange(grid_shape[1]),
        np.arange(grid_shape[2]), indexing="ij",
    )
    ones = np.ones(ii.size)
    idx = np.stack([ii.reshape(-1), jj.reshape(-1), kk.reshape(-1), ones], axis=1)

    src = idx @ M.T                                          # [N, 4]
    src_vox = np.rint(src[:, :3]).astype(np.int64)           # nearest source voxel

    lo = box[:, 0][None, :]
    hi = box[:, 1][None, :]
    inside = np.all((src_vox >= lo) & (src_vox < hi), axis=1)
    return inside.reshape(grid_shape)
