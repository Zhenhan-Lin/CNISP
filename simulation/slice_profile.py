"""
Slice profile kernel factory.

Dispatches by imaging modality:
  - CT:  rectangular SSP (box kernel, analytic)
  - MRI: Gaussian (default) or ESPRESO-estimated (.npy)

All kernels are 1D, odd-length, normalized, and centered. The odd-length
invariant guarantees that a centered convolution introduces zero positional
shift — the sample position stays at the kernel's center tap.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np


def ct_box_kernel(step: int) -> np.ndarray:
    """Rectangular SSP for CT.

    Width = 2*floor(step/2)+1 (always odd). This models the CT slice
    sensitivity profile as a uniform weight over a slab roughly equal
    to the slice thickness, with the constraint that the kernel is
    symmetric about its center tap (odd length).
    """
    width = 2 * (step // 2) + 1
    kernel = np.ones(width, dtype=np.float64)
    return kernel / kernel.sum()


def gaussian_kernel(fwhm_vox: float) -> np.ndarray:
    """Gaussian SSP for MRI (default when no estimated kernel available).

    Width = 2*round(fwhm)+1 (always odd).
    FWHM = 2*sqrt(2*ln2)*sigma ≈ 2.3548*sigma.
    """
    half = int(round(fwhm_vox))
    width = 2 * half + 1
    sigma = fwhm_vox / 2.3548200923244
    x = np.arange(width, dtype=np.float64) - half
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def load_estimated_kernel(path: Union[str, Path]) -> np.ndarray:
    """Load an ESPRESO-estimated PSF from a .npy file.

    Asserts the kernel is 1D and odd-length (required for centered-conv
    position invariance).
    """
    kernel = np.load(str(path)).squeeze().astype(np.float64)
    if kernel.ndim != 1:
        raise ValueError(
            f"Estimated kernel must be 1D, got shape {kernel.shape} "
            f"from {path}"
        )
    if kernel.shape[0] % 2 == 0:
        raise ValueError(
            f"Estimated kernel must have odd length for centered-conv "
            f"position invariance. Got length {kernel.shape[0]} from {path}. "
            f"Pad or trim by one sample."
        )
    return kernel / kernel.sum()


def get_kernel(
    modality: str,
    step: int,
    *,
    estimated_path: Optional[Union[str, Path]] = None,
    fwhm_vox: Optional[float] = None,
) -> np.ndarray:
    """Dispatch kernel construction by modality.

    Parameters
    ----------
    modality : "ct" or "mri"
    step : subsampling factor (used to set kernel width for CT box,
           and as default FWHM for MRI gaussian)
    estimated_path : path to a pre-cached ESPRESO .npy kernel (MRI only)
    fwhm_vox : explicit FWHM in voxels for MRI gaussian (overrides step)

    Returns
    -------
    1D numpy array, odd length, normalized, centered.
    """
    modality = modality.lower().strip()
    if modality == "ct":
        return ct_box_kernel(step)
    elif modality == "mri":
        if estimated_path is not None:
            return load_estimated_kernel(estimated_path)
        fwhm = fwhm_vox if fwhm_vox is not None else float(step)
        return gaussian_kernel(fwhm)
    else:
        raise ValueError(
            f"Unknown modality '{modality}'. Expected 'ct' or 'mri'."
        )
