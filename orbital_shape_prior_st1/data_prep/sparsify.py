"""
Synthetic sparsification to simulate anisotropic acquisition.

Reuses Amiranashvili's sparsen_volume logic but adapted for our pipeline.
This module is used both in training (to create sparse input from dense GT)
and in the diagnostic pipeline (to create controlled test conditions).
"""

from typing import Tuple

import torch


def sparsen_volume(
    volume: torch.Tensor,
    spacing: torch.Tensor,
    offset: torch.Tensor,
    axis: int,
    slice_step_size: int,
    slice_start_id: int = 0,
    use_thick_slices: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create a sparsified version of the volume by keeping every Nth slice.

    Directly adapted from Amiranashvili et al. data_generation.py.

    Args:
        volume: [D1, D2, D3] label tensor
        spacing: [3] voxel spacing
        offset: [3] spatial offset of first voxel center
        axis: which axis to sparsify (0=sag, 1=cor, 2=axial in LPS)
        slice_step_size: keep every Nth slice
        slice_start_id: which slice to start from
        use_thick_slices: average neighboring slices before selection

    Returns:
        (sparse_volume, new_spacing, new_offset)
    """
    if use_thick_slices:
        # Guard: thick slices average neighboring labels then threshold at 0.5.
        # This only makes sense for binary masks. For multi-class labels
        # (values 0,1,2,3,4), averaging produces meaningless fractional values
        # and thresholding destroys the class structure.
        n_unique = len(torch.unique(volume))
        if n_unique > 2:
            raise ValueError(
                f"use_thick_slices=True is incompatible with multi-class labels "
                f"({n_unique} unique values found). Set use_thick_slices=False."
            )
        volume = _compute_thick_slices(volume, axis, slice_step_size)

    if slice_step_size <= 1:
        return volume, spacing, offset

    slice_ids = torch.arange(slice_start_id, volume.shape[axis], slice_step_size)
    volume_sparse = torch.index_select(volume, axis, slice_ids)

    spacing_sparse = spacing.clone()
    spacing_sparse[axis] *= slice_step_size

    offset_sparse = offset.clone()
    offset_sparse[axis] += slice_start_id * spacing[axis]

    return volume_sparse, spacing_sparse, offset_sparse


def _compute_thick_slices(
    volume: torch.Tensor,
    axis: int,
    step_size: int,
) -> torch.Tensor:
    """Average neighboring slices to simulate thick-slice acquisition."""
    thick_slices = []
    half = round(step_size / 2) - 1 if step_size % 2 == 0 else round(step_size / 2 - 0.5)

    for center in range(volume.shape[axis]):
        start = max(0, center - half)
        end = min(start + step_size, volume.shape[axis])
        start = max(start, 0)
        subvol = torch.index_select(volume, axis, torch.arange(start, end))
        mean_slice = torch.mean(subvol.float(), dim=axis)
        mean_slice = (mean_slice >= 0.5).to(volume.dtype)
        thick_slices.append(mean_slice)

    return torch.stack(thick_slices, axis)
