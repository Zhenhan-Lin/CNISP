"""
Thin wrapper around ECLARE's ESPRESO implementation.

Estimates the slice selection profile (PSF) from a single anisotropic
MRI volume using a GAN that matches in-plane and through-plane patch
distributions. The estimated kernel is cached as a .npy file for reuse.

Dependency: `pip install eclare` (requires torch>=2.5, radifox-utils==1.0.3).

This module is DORMANT for CT-only cohorts — CT uses the analytic
ct_box_kernel from slice_profile.py instead.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch


def estimate_slice_profile(
    aniso_nifti_path: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    device: str = "cuda:0",
    force: bool = False,
) -> Path:
    """Run ESPRESO on one MRI volume and cache the PSF as .npy.

    Parameters
    ----------
    aniso_nifti_path : path to the anisotropic MRI NIfTI
    output_dir : directory to save the estimated kernel and diagnostic plot
    device : torch device string
    force : if True, re-estimate even if cached .npy exists

    Returns
    -------
    Path to the saved kernel .npy file.
    """
    from eclare.espreso import run_espreso
    from eclare.utils.train_set import TrainSet
    from eclare.utils.parse_image_file import parse_image

    aniso_nifti_path = Path(aniso_nifti_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = aniso_nifti_path.stem.replace(".nii", "")
    psf_path = output_dir / f"{stem}_psf.npy"
    plot_path = output_dir / f"{stem}_psf.png"

    if psf_path.exists() and not force:
        return psf_path

    image, slice_separation, scales, lr_axis, header, affine, orig_min, orig_max = (
        parse_image(aniso_nifti_path, normalize_image=True)
    )
    dataset = TrainSet(image=image, lr_axis=lr_axis, verbose=False)

    _g, _elapsed = run_espreso(
        slice_separation,
        dataset,
        torch.device(device),
        str(psf_path),
        str(plot_path),
    )
    return psf_path


def get_or_estimate_kernel(
    aniso_nifti_path: Union[str, Path],
    cache_dir: Union[str, Path],
    *,
    device: str = "cuda:0",
) -> np.ndarray:
    """Load cached kernel or estimate it, then return as numpy array.

    Convenience function for pipeline integration.
    """
    from simulation.slice_profile import load_estimated_kernel

    psf_path = estimate_slice_profile(
        aniso_nifti_path, cache_dir, device=device
    )
    return load_estimated_kernel(psf_path)
