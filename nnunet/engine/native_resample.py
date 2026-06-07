#!/usr/bin/env python3
"""World-coordinate resampling of nnUNet plan/iso logits onto the native grid.

Extracted from ``predict_sparse_iso.py`` so the geometry can be unit-tested
WITHOUT a working nnUNet install (this module imports only numpy + nibabel).

Why world coordinates (the bug this fixes)
------------------------------------------
nnUNet's segmentation export resamples by array SHAPE, ignoring the affine.
Reconstructing the native mask by scaling the crop bbox by ``step`` and
re-running that resampler aligns the plan FOV's *extent* to the native crop's
*extent* -- two equal-width FOVs offset by half a coarse voxel -- so every
kept sparse slice lands at the CENTRE of its ``step``-wide slab: a through-
plane shift of ``(step-1)/2`` native voxels (grows with ``step``), silently
hurting Dice against the start=0 GT.

Here we instead resample the plan logits onto the native grid by WORLD
coordinates, with the plan grid anchored to the sparse CT's own affine (the
true start=0 sweep geometry). ``sparse voxel i`` then lands at
``native voxel i*step`` to sub-voxel precision, for any ``step`` parity and
for both thin and thick degradation.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to


def internal_to_nib_perm(
    transpose_forward: Sequence[int], io2nib: Sequence[int]
) -> List[int]:
    """Permutation ``P`` to reorder an INTERNAL-axis object into nibabel order.

    nnUNet keeps per-axis bookkeeping (logits spatial dims, crop bbox) in its
    INTERNAL order, where ``internal[i] == as_read[transpose_forward[i]]``.
    ``io2nib`` maps nibabel axis ``k`` to as-read axis ``io2nib[k]``. Hence
    nibabel axis ``k`` corresponds to internal axis
    ``transpose_forward.index(io2nib[k])``. The same ``P`` reorders both an
    array (``array.transpose(P)``) and a per-internal-axis list.
    """
    tf = list(transpose_forward)
    return [tf.index(int(io2nib[k])) for k in range(3)]


def plan_affine_nib(
    sparse_affine: np.ndarray,
    bbox_nib: Sequence[Sequence[int]],
    plan_shape_nib: Sequence[int],
) -> np.ndarray:
    """World affine (nibabel frame) of the plan/iso logits grid.

    Reconstructed from the sparse CT's nibabel affine, the nonzero crop bbox
    (nibabel order), and the plan grid shape, assuming the FOV-preserving
    (half-pixel) resize nnUNet uses on the forward sparse->plan pass:

      * crop origin    : sparse voxel ``bbox_nib[k][0]`` -> world via affine.
      * voxel-size scale ``f_k = n_crop_k / n_plan_k`` (FOV width preserved).
      * plan columns    ``v'_k = v_k * f_k``  (v_k = sparse step vector).
      * plan origin     ``o_crop + sum_k 0.5*(f_k - 1)*v_k`` (plan voxel-0
                         centre sits half a plan voxel in from the shared edge).
    """
    A = np.asarray(sparse_affine, dtype=np.float64)
    lo = np.array([bbox_nib[k][0] for k in range(3)], dtype=np.float64)
    o_crop = A[:3, :3] @ lo + A[:3, 3]
    n_crop = np.array(
        [bbox_nib[k][1] - bbox_nib[k][0] for k in range(3)], dtype=np.float64
    )
    n_plan = np.array(plan_shape_nib, dtype=np.float64)
    f = n_crop / n_plan
    plan_aff = np.eye(4, dtype=np.float64)
    plan_aff[:3, :3] = A[:3, :3] * f[np.newaxis, :]
    plan_aff[:3, 3] = o_crop + 0.5 * (A[:3, :3] @ (f - 1.0))
    return plan_aff


def resample_plan_to_native(
    plan_internal: np.ndarray,
    transpose_forward: Sequence[int],
    io2nib: Sequence[int],
    bbox_internal: Sequence[Sequence[int]],
    sparse_affine: np.ndarray,
    native_shape_nib: Tuple[int, int, int],
    native_affine_nib: np.ndarray,
    order: int = 0,
) -> np.ndarray:
    """Resample a plan-spacing label map onto the native CT grid (world-aware).

    Parameters
    ----------
    plan_internal : plan-spacing label map in nnUNet INTERNAL axis order
        (e.g. ``logits.argmax(0)``).
    transpose_forward, io2nib : axis-order maps (see ``internal_to_nib_perm``).
    bbox_internal : nnUNet ``bbox_used_for_cropping`` (internal order).
    sparse_affine : nibabel affine of the sparse input CT.
    native_shape_nib, native_affine_nib : the native CT's nibabel grid.
    order : interpolation order (0 = nearest, for labels).

    Returns the native-grid label map (uint8) in nibabel axis order.
    """
    perm = internal_to_nib_perm(transpose_forward, io2nib)
    plan_nib = np.ascontiguousarray(np.transpose(plan_internal, perm))
    bbox_nib = [list(map(int, bbox_internal[perm[k]])) for k in range(3)]
    plan_aff = plan_affine_nib(sparse_affine, bbox_nib, plan_nib.shape)
    plan_img = nib.Nifti1Image(plan_nib.astype(np.uint8), plan_aff)
    out = resample_from_to(
        plan_img,
        (tuple(int(x) for x in native_shape_nib), np.asarray(native_affine_nib)),
        order=order, mode="constant", cval=0,
    )
    return np.asarray(out.dataobj).astype(np.uint8)
