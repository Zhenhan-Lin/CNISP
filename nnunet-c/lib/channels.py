"""Per-case channel assembly + geometry asserts for the corrector datasets.

Responsibilities:
  * ch0 degraded-source PIN: refuse any ch0 that is not a degraded sparse CT
    (pothole IV -- never let the corrector read the answer off a sharp image).
  * split a multi-class prelabel mask into per-structure binary channels.
  * assemble one case: resample ct/prelabel/gt to the 835 plan-spacing grid
    (pothole-2 a-ii), write _0000.._000N + label, assert identical geometry.

Depends on numpy + nibabel + lib.resample + lib.labels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import nibabel as nib

from lib import resample as _rs
from lib.labels import remap_to_nnunet


def assert_degraded_ct_path(ct_path: Path, experiment: str) -> None:
    """Pin ch0 to the degraded/thick sparse CT; reject native/dense inputs."""
    s = str(ct_path)
    marker = f"/{experiment}/sparse_step_"
    if marker not in s:
        raise AssertionError(
            f"ch0 must be a degraded sparse CT under '{marker}...', got {s!r}. "
            f"Refusing to build the corrector on a non-degraded image."
        )
    if "/input/native/" in s or s.rstrip("/").endswith("/native"):
        raise AssertionError(
            f"ch0 points at a NATIVE/dense CT ({s!r}); ch0 must be degraded."
        )


def split_mask_to_binaries(
    arr: np.ndarray, struct_to_value: Dict[str, int], structures: List[str]
) -> List[np.ndarray]:
    """One uint8 {0,1} channel per structure (in the fixed `structures` order)."""
    return [
        (arr == struct_to_value[name]).astype(np.uint8) for name in structures
    ]


def _save_like(arr: np.ndarray, affine: np.ndarray, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(arr, np.asarray(affine)), str(dst))


def _assemble_images(
    case_id: str,
    ct_path: Path,
    target_spacing: List[float],
    n_channels: int,
    structures: List[str],
    images_dir: Path,
    experiment: str,
    prelabel_path: Optional[Path],
    prelabel_struct_to_value: Optional[Dict[str, int]],
    file_ending: str,
):
    """Write ch0 (+ binary ch1..chN) to images_dir on the 835 plan-spacing grid.

    Returns (target_shape, target_affine, written_filenames). Shared by training
    assembly (assemble_case) and inference assembly (assemble_inference_case).
    """
    assert_degraded_ct_path(Path(ct_path), experiment)

    ct_img = nib.load(str(ct_path))
    target_shape, target_affine = _rs.build_reference_grid(ct_img, target_spacing)

    # ch0: CT, cubic spline.
    ct_rs = _rs.resample_to_grid(ct_img, target_shape, target_affine, order=3)
    ct_arr = np.asanyarray(ct_rs.dataobj).astype(np.float32)
    _save_like(ct_arr, target_affine, images_dir / f"{case_id}_0000{file_ending}")
    written = [f"{case_id}_0000{file_ending}"]

    # ch1..chN: binary prelabel channels, nearest.
    if n_channels > 1:
        if prelabel_path is None or prelabel_struct_to_value is None:
            raise ValueError(
                f"{case_id}: n_channels={n_channels} requires prelabel_path + "
                f"prelabel_struct_to_value"
            )
        if len(structures) != n_channels - 1:
            raise ValueError(
                f"{case_id}: n_channels-1={n_channels - 1} != len(structures)="
                f"{len(structures)}"
            )
        pre_img = nib.load(str(prelabel_path))
        pre_rs = _rs.resample_to_grid(pre_img, target_shape, target_affine, order=0)
        pre_arr = np.asanyarray(pre_rs.dataobj)
        binaries = split_mask_to_binaries(
            pre_arr, prelabel_struct_to_value, structures
        )
        for i, bin_arr in enumerate(binaries, start=1):
            name = f"{case_id}_{i:04d}{file_ending}"
            _save_like(bin_arr.astype(np.uint8), target_affine, images_dir / name)
            written.append(name)

    return target_shape, target_affine, written


def assemble_inference_case(
    case_id: str,
    ct_path: Path,
    target_spacing: List[float],
    n_channels: int,
    structures: List[str],
    images_dir: Path,
    experiment: str,
    prelabel_path: Optional[Path] = None,
    prelabel_struct_to_value: Optional[Dict[str, int]] = None,
    file_ending: str = ".nii.gz",
) -> Dict:
    """Assemble inference channels only (no GT/label) into images_dir."""
    target_shape, target_affine, written = _assemble_images(
        case_id, ct_path, target_spacing, n_channels, structures, images_dir,
        experiment, prelabel_path, prelabel_struct_to_value, file_ending,
    )
    _assert_geometry_images(images_dir, written, target_shape, target_affine)
    return {
        "case_id": case_id,
        "shape": [int(x) for x in target_shape],
        "spacing": [float(x) for x in target_spacing],
        "n_channels": n_channels,
        "image_files": written,
    }


def assemble_case(
    case_id: str,
    ct_path: Path,
    gt_path: Path,
    target_spacing: List[float],
    n_channels: int,
    structures: List[str],
    gt_struct_to_value: Dict[str, int],
    images_dir: Path,
    labels_dir: Path,
    experiment: str,
    prelabel_path: Optional[Path] = None,
    prelabel_struct_to_value: Optional[Dict[str, int]] = None,
    file_ending: str = ".nii.gz",
) -> Dict:
    """Assemble one nnUNet TRAINING case onto the 835 plan-spacing reference grid.

    Writes images_dir/{case}_0000{fe} (CT, order 3) and, for 5-channel controls,
    images_dir/{case}_0001..000N{fe} (binary prelabel channels, order 0), plus
    labels_dir/{case}{fe} (GT remapped to {1,2,3,4}, order 0). All share the
    reference grid exactly. Returns a summary dict for manifest/asserts.
    """
    target_shape, target_affine, written = _assemble_images(
        case_id, ct_path, target_spacing, n_channels, structures, images_dir,
        experiment, prelabel_path, prelabel_struct_to_value, file_ending,
    )

    # label: GT remapped to {1,2,3,4}, nearest.
    gt_img = nib.load(str(gt_path))
    gt_rs = _rs.resample_to_grid(gt_img, target_shape, target_affine, order=0)
    gt_arr = np.asanyarray(gt_rs.dataobj).astype(np.int32)
    label_arr = remap_to_nnunet(gt_arr, gt_struct_to_value, structures)
    label_name = f"{case_id}{file_ending}"
    _save_like(label_arr.astype(np.uint8), target_affine, labels_dir / label_name)

    # Geometry asserts: every written file shares the reference grid.
    _assert_geometry(images_dir, labels_dir, written, label_name, target_shape,
                     target_affine)

    return {
        "case_id": case_id,
        "shape": [int(x) for x in target_shape],
        "spacing": [float(x) for x in target_spacing],
        "n_channels": n_channels,
        "image_files": written,
        "label_file": label_name,
        "label_values": sorted(int(v) for v in np.unique(label_arr)),
    }


def _assert_geometry_images(
    images_dir: Path, image_files: List[str], target_shape, target_affine
) -> None:
    ref_shape = tuple(int(s) for s in target_shape)
    ref_aff = np.asarray(target_affine)
    for name in image_files:
        img = nib.load(str(images_dir / name))
        if img.shape[:3] != ref_shape:
            raise AssertionError(
                f"{name}: shape {img.shape[:3]} != reference {ref_shape}"
            )
        if not np.allclose(img.affine, ref_aff, atol=1e-4):
            raise AssertionError(f"{name}: affine mismatch vs reference grid")


def _assert_geometry(
    images_dir: Path,
    labels_dir: Path,
    image_files: List[str],
    label_file: str,
    target_shape,
    target_affine,
) -> None:
    _assert_geometry_images(images_dir, image_files, target_shape, target_affine)
    ref_shape = tuple(int(s) for s in target_shape)
    ref_aff = np.asarray(target_affine)
    lab = nib.load(str(labels_dir / label_file))
    if lab.shape[:3] != ref_shape:
        raise AssertionError(
            f"{label_file}: shape {lab.shape[:3]} != reference {ref_shape}"
        )
    if not np.allclose(lab.affine, ref_aff, atol=1e-4):
        raise AssertionError(f"{label_file}: affine mismatch vs reference grid")
