"""Build a per-case reference grid at the Dataset835 plan spacing and resample
images/masks onto it.

This implements the pothole-2 (a-ii) decision: by resampling every channel +
label to a grid whose voxel spacing equals nnUNet's plan target spacing, the
preprocessing resample (`compute_new_shape` -> ratio 1 -> no change) becomes a
no-op, so the binary prelabel channels survive preprocessing as {0,1} instead of
being cubic-spline-interpolated into continuous values.

Geometry note: we keep each case's own direction cosines + origin and only set
the voxel-axis magnitudes to the target spacing. For the iso-0.5 Dataset835 plan
(isotropic) this matches nnUNet's post-transpose spacing exactly regardless of
orientation. For an anisotropic plan, confirm transpose_forward maps spacing
components to the same axes (the pothole-4 gate catches any residual resample).

Depends only on numpy + nibabel.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to


def read_plan_target_spacing(plan_json: Path, configuration: str) -> List[float]:
    """Read ``configurations.<configuration>.spacing`` from an nnUNet plan JSON."""
    with open(plan_json) as f:
        plan = json.load(f)
    try:
        return [float(x) for x in plan["configurations"][configuration]["spacing"]]
    except KeyError as e:  # noqa: BLE001
        raise KeyError(
            f"plan JSON {plan_json} has no configurations.{configuration}.spacing"
        ) from e


def resolve_target_spacing(cfg: Dict) -> List[float]:
    """Determine the target spacing for the no-op resample.

    Priority: explicit ``target_spacing`` in corrector.yaml, else read the 835
    plan JSON at ``${nnUNet_preprocessed}/Dataset{id}_{name}/{plan}.json``.
    """
    explicit = cfg.get("target_spacing")
    if explicit:
        return [float(x) for x in explicit]

    preproc = os.environ.get("nnUNet_preprocessed")
    if not preproc:
        raise RuntimeError(
            "target_spacing is null and $nnUNet_preprocessed is unset; cannot "
            "locate the 835 plan JSON. Set corrector.yaml::target_spacing "
            "explicitly, or export nnUNet_preprocessed on the GPU box."
        )
    ds = f"Dataset{int(cfg['reference_dataset_id']):03d}_{cfg['reference_dataset_name']}"
    ref_plan_name = cfg.get("reference_plan_json", cfg["reference_plan"])
    plan_json = Path(preproc) / ds / f"{ref_plan_name}.json"
    if not plan_json.is_file():
        raise FileNotFoundError(
            f"835 plan JSON not found: {plan_json}. Confirm reference_plan "
            f"(nnUNetPlans vs nnUNetPlans_iso05) on the GPU box."
        )
    return read_plan_target_spacing(plan_json, cfg["configuration"])


def voxel_spacing(affine: np.ndarray) -> np.ndarray:
    """Per-voxel-axis spacing = L2 norm of each affine column."""
    return np.sqrt(np.sum(np.asarray(affine)[:3, :3] ** 2, axis=0))


def compute_new_shape(
    old_shape: Sequence[int],
    old_spacing: Sequence[float],
    new_spacing: Sequence[float],
) -> Tuple[int, int, int]:
    """nnUNet's compute_new_shape: round(old_shape * old_spacing / new_spacing)."""
    return tuple(
        int(round(s * o / n))
        for s, o, n in zip(old_shape, old_spacing, new_spacing)
    )


def build_reference_grid(
    ref_img: "nib.Nifti1Image", target_spacing: Sequence[float]
) -> Tuple[Tuple[int, int, int], np.ndarray]:
    """Target (shape, affine) at ``target_spacing``, anchored to ``ref_img``.

    Keeps ``ref_img``'s direction cosines + origin; scales each voxel-axis to
    the target spacing; recomputes shape so the FOV is preserved.
    """
    old_affine = np.asarray(ref_img.affine, dtype=float)
    old_shape = ref_img.shape[:3]
    old_sp = voxel_spacing(old_affine)
    direction = old_affine[:3, :3] / old_sp  # unit columns
    new_affine = np.eye(4)
    new_affine[:3, :3] = direction * np.asarray(target_spacing, dtype=float)
    new_affine[:3, 3] = old_affine[:3, 3]
    new_shape = compute_new_shape(old_shape, old_sp, target_spacing)
    return new_shape, new_affine


def resample_to_grid(
    img: "nib.Nifti1Image",
    target_shape: Sequence[int],
    target_affine: np.ndarray,
    order: int,
) -> "nib.Nifti1Image":
    """World-coordinate resample of ``img`` onto (target_shape, target_affine).

    order=3 for CT intensities, order=0 (nearest) for label/binary masks.
    """
    return resample_from_to(
        img, (tuple(int(s) for s in target_shape), np.asarray(target_affine)),
        order=order,
    )
