"""
Affine helpers and hard assertions for the simulation module.

These guards enforce the geometric invariants that keep the
sparse-to-native position mapping exact by construction.
"""

from typing import Tuple

import numpy as np
import torch


def assert_start_zero(start: int) -> None:
    """Deployment/test path must use start=0 so sparse voxel i maps to
    native voxel i*step exactly."""
    if start != 0:
        raise ValueError(
            f"Deployment/test degradation requires start=0 for exact "
            f"sparse-to-native index correspondence. Got start={start}."
        )


def assert_odd_kernel(kernel: np.ndarray) -> None:
    """Centered conv position invariance requires odd-length kernels."""
    if kernel.shape[0] % 2 == 0:
        raise ValueError(
            f"Kernel must have odd length for centered-conv position "
            f"invariance. Got length {kernel.shape[0]}."
        )


def assert_integer_step(step: int) -> None:
    """Step must be a positive integer."""
    if not isinstance(step, int) or step < 1:
        raise ValueError(f"step must be a positive integer, got {step}")


def compute_sparse_affine(
    spacing: torch.Tensor,
    offset: torch.Tensor,
    axis: int,
    step: int,
    start: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the (spacing, offset) for a sparsified volume.

    This formula is valid for BOTH thin and thick degradation because
    centered convolution does not shift the sample position.

    Parameters
    ----------
    spacing : [3] dense voxel spacing
    offset : [3] dense first-voxel-center offset
    axis : sparsification axis (0/1/2)
    step : subsampling factor
    start : first slice index kept

    Returns
    -------
    (new_spacing, new_offset) tensors of shape [3].
    """
    new_spacing = spacing.clone()
    new_spacing[axis] *= step

    new_offset = offset.clone()
    new_offset[axis] += start * spacing[axis]

    return new_spacing, new_offset
