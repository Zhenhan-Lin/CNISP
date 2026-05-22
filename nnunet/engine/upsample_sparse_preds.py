#!/usr/bin/env python3
"""NN-upsample nnUNet sparse-CT predictions back to the native CT grid.

For each ``(source_id, step)`` in
``${work_dir}/nnunet_input_sparse_manifest.json`` (written by
``data_prep/sparsify_inputs.py``):

1. Read the sparse prediction from
   ``${work_dir}/nnunet_pred_native_step_{XX}/{sid}.nii.gz``.
2. Resample it onto the original CT's voxel grid using nearest-neighbour.
3. Write the result to
   ``${work_dir}/nnunet_pred_native_step_{XX}_upsampled/{sid}.nii.gz``.

step_01 (the dense baseline) is special-cased: no actual upsampling --
just symlink ``nnunet_pred_native/{sid}.nii.gz`` into the step_01
upsampled directory so the manifest is complete.

Why this exists
---------------
nnUNetv2's predict resamples the input to plan spacing for forward
inference, then resamples the prediction back to the *input* spacing on
save. Our input was a sparsified CT (one column of the affine scaled by
``step``), so the saved prediction is on that sparse grid. To compare
with the native-grid GT (which we deliberately keep untouched), we have
to project the prediction back to the dense native grid. Nearest
neighbour is the only honest choice for a discrete label map.

Output
------
* ``${work_dir}/nnunet_pred_native_step_{XX}_upsampled/{sid}.nii.gz``
* ``${work_dir}/nnunet_pred_native_sweep_manifest.json`` mirroring
  CNISP's ``native_sweep_manifest.json`` so ``compare_native.py`` can
  index nnUNet identically to how it indexes CNISP.

Usage
-----
    python nnunet/engine/upsample_sparse_preds.py --config nnunet/configs.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import nibabel as nib
import numpy as np
import yaml


def _load_yaml(path: Path) -> Dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _nn_upsample_along_axis(
    sparse_arr: np.ndarray,
    sparse_affine: np.ndarray,
    target_shape: tuple,
    target_affine: np.ndarray,
    step_axis: int,
    step_size: int,
    tol: float = 1e-3,
) -> np.ndarray:
    """Cheap NN resample when sparsification only touched ``step_axis``.

    Verifies the affines satisfy ``sparse_affine[:3, axis] ==
    target_affine[:3, axis] * step`` for ``axis == step_axis`` and are
    identical otherwise. Falls back to a hard error if not (so any
    nnUNetv2 behaviour change surfaces immediately instead of producing
    silently misaligned masks).
    """
    if sparse_arr.ndim != 3 or len(target_shape) != 3:
        raise ValueError(
            f"Expected 3D arrays; got sparse {sparse_arr.shape} target {target_shape}"
        )
    for ax in range(3):
        s_col = sparse_affine[:3, ax]
        t_col = target_affine[:3, ax]
        if ax == step_axis:
            expected = t_col * step_size
            if not np.allclose(s_col, expected, atol=tol):
                raise RuntimeError(
                    f"Sparse pred affine column {ax} = {s_col} disagrees with "
                    f"target * step ({step_size}) = {expected} (atol={tol}). "
                    f"nnUNet may have changed its resample behaviour, or the "
                    f"input CT was modified between sparsify and predict."
                )
        else:
            if not np.allclose(s_col, t_col, atol=tol):
                raise RuntimeError(
                    f"Sparse pred affine column {ax} = {s_col} disagrees with "
                    f"target {t_col} (atol={tol}). The non-step axes should be "
                    f"identical between sparse and dense grids."
                )
    if not np.allclose(sparse_affine[:3, 3], target_affine[:3, 3], atol=tol):
        raise RuntimeError(
            f"Origin mismatch between sparse {sparse_affine[:3, 3]} and "
            f"target {target_affine[:3, 3]} (atol={tol}). slice_start_id "
            f"should be 0 in data_prep/sparsify_inputs.py."
        )

    other_axes = [ax for ax in range(3) if ax != step_axis]
    for ax in other_axes:
        if sparse_arr.shape[ax] != target_shape[ax]:
            raise RuntimeError(
                f"Non-step axis {ax} differs: sparse {sparse_arr.shape[ax]} "
                f"vs target {target_shape[ax]}. nnUNet should preserve the "
                f"input grid on non-step axes."
            )

    native_idx = np.arange(target_shape[step_axis])
    sparse_idx = np.clip(
        np.round(native_idx / float(step_size)).astype(np.int64),
        0, sparse_arr.shape[step_axis] - 1,
    )
    return np.take(sparse_arr, sparse_idx, axis=step_axis)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="nnunet/configs.yaml")
    args = ap.parse_args()

    cfg = _load_yaml(Path(args.config))
    work_dir = Path(cfg["work_dir"])

    sparse_manifest = work_dir / "nnunet_input_sparse_manifest.json"
    if not sparse_manifest.exists():
        print(f"[upsample_sparse_preds] missing {sparse_manifest} -- "
              f"run nnunet/data_prep/sparsify_inputs.py first.",
              file=sys.stderr)
        return 2
    with open(sparse_manifest) as f:
        sparse_m = json.load(f)

    source_to_path = work_dir / "source_to_path.json"
    if not source_to_path.exists():
        print(f"[upsample_sparse_preds] missing {source_to_path} -- "
              f"run nnunet/data_prep/prepare_inputs.py first.",
              file=sys.stderr)
        return 2
    with open(source_to_path) as f:
        src_to_path = json.load(f)

    dense_pred_dir = work_dir / "nnunet_pred_native"

    out_steps: Dict[str, Dict[str, str]] = {}

    # ── step_01: just symlink the dense baseline ────────────────
    if dense_pred_dir.exists():
        out_01 = work_dir / "nnunet_pred_native_step_01_upsampled"
        out_01.mkdir(parents=True, exist_ok=True)
        step_01_map: Dict[str, str] = {}
        for sid in src_to_path:
            src_pred = dense_pred_dir / f"{sid}.nii.gz"
            if not src_pred.exists():
                print(f"  [step_01] {sid}: no dense pred at {src_pred}",
                      file=sys.stderr)
                continue
            dst = out_01 / f"{sid}.nii.gz"
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(src_pred.resolve())
            step_01_map[sid] = str(dst)
        out_steps["01"] = step_01_map
        print(f"[upsample_sparse_preds] step_01: symlinked "
              f"{len(step_01_map)} dense baseline(s) into {out_01}")
    else:
        print(f"[upsample_sparse_preds] note: dense baseline dir missing "
              f"({dense_pred_dir}); step_01 will be absent from the manifest.",
              file=sys.stderr)

    # ── step >= 2: real NN upsample ──────────────────────────────
    n_written = 0
    n_skipped = 0
    for step_tag in sorted(sparse_m.get("by_step", {}).keys()):
        step = int(step_tag)
        sparse_pred_dir = work_dir / f"nnunet_pred_native_step_{step_tag}"
        up_dir = work_dir / f"nnunet_pred_native_step_{step_tag}_upsampled"
        up_dir.mkdir(parents=True, exist_ok=True)

        step_map: Dict[str, str] = {}
        for sid, info in sparse_m["by_step"][step_tag].items():
            sparse_pred = sparse_pred_dir / f"{sid}.nii.gz"
            if not sparse_pred.exists():
                print(f"  [step_{step_tag}] {sid}: no sparse pred at "
                      f"{sparse_pred} -- skipping", file=sys.stderr)
                continue

            dst = up_dir / f"{sid}.nii.gz"
            if dst.exists():
                n_skipped += 1
                step_map[sid] = str(dst)
                continue

            ct_path = Path(src_to_path[sid]["ct_image_path"])
            ct_img = nib.load(str(ct_path))
            target_shape = tuple(int(x) for x in ct_img.shape[:3])
            target_affine = ct_img.affine

            sparse_img = nib.load(str(sparse_pred))
            sparse_arr = np.asarray(sparse_img.dataobj)
            if np.issubdtype(sparse_arr.dtype, np.floating):
                sparse_arr = np.rint(sparse_arr).astype(np.uint8)
            else:
                sparse_arr = sparse_arr.astype(np.uint8, copy=False)

            step_axis = int(info["step_axis"])

            up_arr = _nn_upsample_along_axis(
                sparse_arr=sparse_arr,
                sparse_affine=sparse_img.affine,
                target_shape=target_shape,
                target_affine=target_affine,
                step_axis=step_axis,
                step_size=step,
            )
            out_img = nib.Nifti1Image(up_arr.astype(np.uint8), target_affine)
            out_img.set_qform(target_affine)
            out_img.set_sform(target_affine)
            nib.save(out_img, str(dst))
            n_written += 1
            step_map[sid] = str(dst)

        if step_map:
            out_steps[step_tag] = step_map
            print(f"[upsample_sparse_preds] step_{step_tag}: "
                  f"{len(step_map)} upsampled in {up_dir}")

    sweep_manifest_path = work_dir / "nnunet_pred_native_sweep_manifest.json"
    with open(sweep_manifest_path, "w") as f:
        json.dump({"steps": out_steps}, f, indent=2)

    print(f"\n[upsample_sparse_preds] wrote {n_written} new file(s); "
          f"{n_skipped} already present.")
    print(f"[upsample_sparse_preds] manifest: {sweep_manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
