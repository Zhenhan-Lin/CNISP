"""
Degradation operators: thin (point-sample) and thick (profile-conv).

Both operators produce sparse volumes with the SAME affine convention
(centered conv preserves position), ensuring thin and thick test lines
are directly comparable and the sparse-to-native index mapping is exact.
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

from simulation.affine_ops import (
    assert_odd_kernel,
    assert_integer_step,
    compute_sparse_affine,
)


def degrade_thin(
    volume: torch.Tensor,
    spacing: torch.Tensor,
    offset: torch.Tensor,
    axis: int,
    step: int,
    start: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Point-sample degradation: keep every Nth slice.

    Identical semantics to the existing sparsen_volume(use_thick_slices=False).

    Parameters
    ----------
    volume : [D1, D2, D3] dense tensor (label or image)
    spacing : [3] voxel spacing
    offset : [3] first-voxel-center offset
    axis : which spatial axis to sparsify (0/1/2)
    step : keep every step-th slice
    start : first slice index to keep

    Returns
    -------
    (sparse_volume, new_spacing, new_offset)
    """
    assert_integer_step(step)
    if step == 1 and start == 0:
        return volume, spacing.clone(), offset.clone()

    slice_ids = torch.arange(start, volume.shape[axis], step)
    sparse = torch.index_select(volume, axis, slice_ids)
    new_spacing, new_offset = compute_sparse_affine(
        spacing, offset, axis, step, start
    )
    return sparse, new_spacing, new_offset


def degrade_thick(
    volume: torch.Tensor,
    spacing: torch.Tensor,
    offset: torch.Tensor,
    axis: int,
    step: int,
    start: int = 0,
    *,
    kernel: np.ndarray,
    is_label: bool = True,
    num_classes: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Physical thick-slice degradation: centered profile-conv + subsample.

    For labels: one-hot -> per-channel centered conv -> argmax.
    For images: direct centered conv on float intensity.

    The affine is identical to degrade_thin because centered convolution
    with an odd-length kernel does not shift the sample position.

    Parameters
    ----------
    volume : [D1, D2, D3] dense label tensor (int) or image tensor (float)
    spacing : [3] voxel spacing
    offset : [3] first-voxel-center offset
    axis : which spatial axis to sparsify (0/1/2)
    step : subsampling factor
    start : first slice index to keep
    kernel : 1D odd-length normalized profile kernel from slice_profile
    is_label : if True, use one-hot + argmax; if False, direct conv
    num_classes : number of label classes (including background)

    Returns
    -------
    (sparse_volume, new_spacing, new_offset)
    """
    assert_integer_step(step)
    assert_odd_kernel(kernel)

    if step == 1 and start == 0:
        return volume, spacing.clone(), offset.clone()

    if is_label:
        blurred = _convolve_label(volume, axis, kernel, num_classes)
    else:
        blurred = _convolve_image(volume, axis, kernel)

    slice_ids = torch.arange(start, blurred.shape[axis], step)
    sparse = torch.index_select(blurred, axis, slice_ids)

    new_spacing, new_offset = compute_sparse_affine(
        spacing, offset, axis, step, start
    )
    return sparse, new_spacing, new_offset


def _convolve_label(
    volume: torch.Tensor,
    axis: int,
    kernel: np.ndarray,
    num_classes: int,
) -> torch.Tensor:
    """One-hot -> per-channel centered conv -> argmax."""
    oh = F.one_hot(volume.long(), num_classes)  # [D1,D2,D3,C]
    oh = oh.permute(3, 0, 1, 2).float()        # [C,D1,D2,D3]
    k_t = torch.from_numpy(kernel).float()
    convolved = _conv1d_along_axis(oh, axis, k_t)  # [C,D1,D2,D3]
    return convolved.argmax(dim=0).to(volume.dtype)


def _convolve_image(
    volume: torch.Tensor,
    axis: int,
    kernel: np.ndarray,
) -> torch.Tensor:
    """Centered 1D conv on a float image, renormalized at the FOV boundary.

    BUG FIX (CT thick degradation): ``_conv1d_along_axis`` zero-pads. For a CT
    image the out-of-FOV value is AIR (~ -1000 HU), not 0, so a plain zero-padded
    box average at the first/last ``step//2`` through-plane slices blends real
    air (-1000) with 0 and lifts those whole slices toward gray. For LABELS this
    is fine (background class == 0); for IMAGES it corrupts the boundary slices.

    Fix: average ONLY the slices that actually exist (the physical thick-slice
    integrates the in-FOV extent), by dividing the zero-padded numerator by the
    zero-padded sum of kernel weights over valid (in-FOV) taps. Interior voxels
    see the full kernel (weights sum to 1) so they are UNCHANGED; only boundary
    slices are corrected (true partial average instead of a gray blend).
    """
    k_t = torch.from_numpy(kernel).float()
    vol = volume.float().unsqueeze(0)  # [1,D1,D2,D3]
    num = _conv1d_along_axis(vol, axis, k_t)
    # Per-position valid kernel weight: conv of an all-ones volume with the same
    # zero-padding. == 1.0 in the interior (kernel normalized), < 1.0 at edges.
    den = _conv1d_along_axis(torch.ones_like(vol), axis, k_t)
    out = num / den.clamp_min(1e-6)
    return out.squeeze(0)


def _conv1d_along_axis(
    tensor: torch.Tensor,
    axis: int,
    kernel_1d: torch.Tensor,
) -> torch.Tensor:
    """Apply centered 1D convolution along a spatial axis.

    Parameters
    ----------
    tensor : [C, D1, D2, D3] or [1, D1, D2, D3]
    axis : spatial axis (0, 1, or 2 — refers to D1/D2/D3, i.e. dims 1/2/3)
    kernel_1d : 1D kernel of odd length L

    The convolution uses zero-padding = (L-1)//2 on each side so that the
    output shape equals the input shape (centered, no shift).
    """
    C = tensor.shape[0]
    L = kernel_1d.shape[0]
    pad = (L - 1) // 2
    spatial_dim = axis + 1  # tensor dims are [C, D1, D2, D3]

    # Move target spatial axis to the last position for conv1d
    # conv1d operates on dim=-1 of a [batch, channels, length] tensor
    perm = list(range(tensor.ndim))
    perm[-1], perm[spatial_dim] = perm[spatial_dim], perm[-1]
    t = tensor.permute(perm)  # [..., target_axis_len] at dim -1

    # Reshape to [C * other_spatial, 1, target_axis_len] for grouped conv1d
    orig_shape = t.shape
    t = t.reshape(-1, 1, t.shape[-1])

    # Kernel shape for conv1d: [out_channels=1, in_channels=1, L]
    w = kernel_1d.reshape(1, 1, L).to(t.device)
    t = F.conv1d(t, w, padding=pad)

    # Restore shape
    t = t.reshape(orig_shape)
    # Undo permutation
    inv_perm = [0] * len(perm)
    for i, p in enumerate(perm):
        inv_perm[p] = i
    t = t.permute(inv_perm)
    return t
