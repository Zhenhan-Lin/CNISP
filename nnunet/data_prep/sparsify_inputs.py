#!/usr/bin/env python3
"""Stage sparsified CT inputs for the nnUNet sweep, matched 1:1 to CNISP.

For each ``(source_id, step_size)`` that CNISP actually evaluated in
``sweep_results.pkl``, drop every Nth axial slice of that source's CT
along its through-plane axis (the axis with the largest voxel spacing),
write the result as ``{work_dir}/nnunet_input_step_{XX}/{sid}_0000.nii.gz``
(nnUNetv2's channel-0 naming convention), and bookkeep everything in
``{work_dir}/nnunet_input_sparse_manifest.json``.

step_size == 1 is intentionally skipped: that's the dense baseline and
nnunet/run_predict_native.sh already produces it under
``nnunet_pred_native/``. The upsample script later symlinks step_01 to
that directory so the sweep manifest is still complete.

Inputs
------
* ``${cnisp_output_basedir}/<cnisp_model_name>/sweep_results.pkl``
* ``${work_dir}/source_to_path.json`` (written by data_prep/prepare_inputs.py)
* Per-source CT NIfTIs referenced by source_to_path.json

Outputs
-------
* ``${work_dir}/nnunet_input_step_{XX}/{sid}_0000.nii.gz`` (one per
  source, per non-trivial step)
* ``${work_dir}/nnunet_input_sparse_manifest.json``

Safety
------
For each ``(source_id, step)`` we compare the new through-plane spacing
to CNISP's ``effective_resolution_mm`` for the same pair (averaged over
OD/OS) and fail loudly if they disagree by more than
``sparse_eff_res_tolerance`` (default 0.05 = 5 %). That catches sources
whose through-plane axis is *not* the highest-spacing voxel axis (rare
sagittal/coronal acquisitions) before we silently sparsify along the
wrong axis.

Usage
-----
    python nnunet/data_prep/sparsify_inputs.py --config nnunet/configs.yaml
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
import torch
import yaml


# This file lives at nnunet/data_prep/sparsify_inputs.py; repo root is two up.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from orbital_shape_prior_st1.data_prep.sparsify import sparsen_volume  # noqa: E402


def _load_yaml(path: Path) -> Dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _build_sweep_set(
    sweep_pkl: Path,
) -> Tuple[Dict[str, List[int]], Dict[Tuple[str, int], float]]:
    """Read sweep_results.pkl, group by source_id, drop step=1.

    Returns
    -------
    by_source : {source_id: sorted list of step_sizes > 1}
    eff_res   : {(source_id, step_size): mean effective_resolution_mm
                 averaged over the two eyes when both ran}
    """
    if not sweep_pkl.exists():
        raise FileNotFoundError(
            f"sweep_results.pkl not found: {sweep_pkl}\n"
            f"  Run `cnisp-infer` first so we know which steps to sparsify."
        )
    with open(sweep_pkl, "rb") as f:
        rows: List[dict] = pickle.load(f)

    eff_accum: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    for r in rows:
        cn = r.get("casename")
        if cn is None:
            continue
        if not (cn.endswith("_OD") or cn.endswith("_OS")):
            continue
        sid = cn[:-3]
        step = int(r["step_size"])
        eff_accum[(sid, step)].append(float(r["effective_resolution_mm"]))

    eff_res = {k: float(np.mean(v)) for k, v in eff_accum.items()}

    by_source: Dict[str, List[int]] = defaultdict(list)
    for (sid, step) in eff_res:
        if step <= 1:
            continue
        by_source[sid].append(step)
    for sid in by_source:
        by_source[sid] = sorted(set(by_source[sid]))
    return dict(by_source), eff_res


def _sparsify_one_ct(
    ct_path: Path,
    step_axis: int,
    step_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply sparsen_volume to a CT NIfTI; return (array, affine).

    The affine's ``step_axis`` column is scaled by ``step_size``; origin
    is unchanged because slice_start_id=0 keeps voxel 0 at the same
    physical location.
    """
    img = nib.load(str(ct_path))
    arr = np.asarray(img.dataobj)
    affine = img.affine.copy()
    spacing = np.asarray(img.header.get_zooms()[:3], dtype=np.float32)

    vol_t = torch.from_numpy(np.ascontiguousarray(arr))
    sp_t = torch.from_numpy(spacing)
    off_t = sp_t / 2.0  # offsets are cosmetic here; we rebuild the affine ourselves

    sparse_vol, _new_sp, _new_off = sparsen_volume(
        vol_t, sp_t, off_t,
        axis=step_axis,
        slice_step_size=step_size,
        slice_start_id=0,
        use_thick_slices=False,
    )
    sparse_arr = sparse_vol.numpy()

    # New affine: column `step_axis` of the rotation/scale 3x3 gets
    # multiplied by step_size; translation stays put.
    new_affine = affine.copy()
    new_affine[:3, step_axis] = new_affine[:3, step_axis] * step_size

    return sparse_arr, new_affine


def _eff_res_from_affine(affine: np.ndarray, axis: int) -> float:
    """Norm of the affine's column for ``axis`` = new physical spacing."""
    return float(np.linalg.norm(affine[:3, axis]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="nnunet/configs.yaml")
    args = ap.parse_args()

    cfg = _load_yaml(Path(args.config))
    cnisp_paths = _load_yaml(Path(cfg["cnisp_paths_yaml"]))

    work_dir = Path(cfg["work_dir"])
    cnisp_output_base = Path(cnisp_paths["output_basedir"]) / cfg["cnisp_model_name"]
    sweep_pkl = cnisp_output_base / "sweep_results.pkl"
    tol = float(cfg.get("sparse_eff_res_tolerance", 0.05))

    source_to_path = work_dir / "source_to_path.json"
    if not source_to_path.exists():
        print(f"[sparsify_inputs] {source_to_path} missing -- "
              f"run nnunet/data_prep/prepare_inputs.py first.",
              file=sys.stderr)
        return 2
    with open(source_to_path) as f:
        manifest_in = json.load(f)

    by_source, eff_res_idx = _build_sweep_set(sweep_pkl)

    print(f"[sparsify_inputs] sweep_results.pkl: {sweep_pkl}")
    print(f"[sparsify_inputs] sources in sweep:   {len(by_source)}")
    print(f"[sparsify_inputs] sources in manifest:{len(manifest_in)}")
    print(f"[sparsify_inputs] work_dir:           {work_dir}")
    print(f"[sparsify_inputs] eff-res tolerance:  {tol:.2%}")

    out_manifest: Dict = {
        "step_axis_per_source": {},
        "by_step": defaultdict(dict),
    }

    n_written = 0
    n_skipped_existing = 0
    issues: List[str] = []

    for sid, steps in sorted(by_source.items()):
        info = manifest_in.get(sid)
        if info is None:
            issues.append(f"{sid}: in sweep_results but not in source_to_path.json")
            continue
        ct_path = Path(info["ct_image_path"])
        if not ct_path.exists():
            issues.append(f"{sid}: ct_image_path missing on disk: {ct_path}")
            continue

        # Pick through-plane axis = highest-spacing voxel axis.
        img = nib.load(str(ct_path))
        zooms = np.asarray(img.header.get_zooms()[:3], dtype=np.float32)
        step_axis = int(np.argmax(zooms))
        out_manifest["step_axis_per_source"][sid] = step_axis
        base_spacing_axis = float(zooms[step_axis])

        for step in steps:
            step_dir = work_dir / f"nnunet_input_step_{step:02d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            dst = step_dir / f"{sid}_0000.nii.gz"

            cnisp_eff_res = eff_res_idx.get((sid, step))
            expected_eff_res = base_spacing_axis * step
            if cnisp_eff_res is None:
                issues.append(
                    f"{sid} step={step}: no eff_res row in sweep_results.pkl "
                    f"-- shouldn't happen since the step came from there."
                )
                continue
            rel = abs(expected_eff_res - cnisp_eff_res) / cnisp_eff_res
            if rel > tol:
                issues.append(
                    f"{sid} step={step}: through-plane axis check failed.\n"
                    f"    voxel argmax axis = {step_axis}, "
                    f"spacing[axis]*step = {expected_eff_res:.3f} mm, "
                    f"CNISP eff_res = {cnisp_eff_res:.3f} mm "
                    f"(rel diff {rel:.2%} > {tol:.2%}).\n"
                    f"    This source probably wasn't axial-acquired; "
                    f"sparsifying along voxel axis {step_axis} would "
                    f"degrade the wrong direction. Refusing to write."
                )
                continue

            if dst.exists():
                n_skipped_existing += 1
            else:
                sparse_arr, new_affine = _sparsify_one_ct(
                    ct_path=ct_path,
                    step_axis=step_axis,
                    step_size=step,
                )
                sanity_eff_res = _eff_res_from_affine(new_affine, step_axis)
                rel2 = abs(sanity_eff_res - cnisp_eff_res) / cnisp_eff_res
                assert rel2 <= tol + 1e-6, (
                    f"{sid} step={step}: post-write affine spacing "
                    f"{sanity_eff_res:.3f} disagrees with CNISP eff_res "
                    f"{cnisp_eff_res:.3f} (rel {rel2:.2%}). Bug in affine "
                    f"scaling."
                )
                out_img = nib.Nifti1Image(sparse_arr, new_affine)
                out_img.set_qform(new_affine)
                out_img.set_sform(new_affine)
                nib.save(out_img, str(dst))
                n_written += 1

            out_manifest["by_step"][f"{step:02d}"][sid] = {
                "input": str(dst),
                "eff_res_mm": round(cnisp_eff_res, 4),
                "step_axis": step_axis,
            }

    if issues:
        print(f"\n[sparsify_inputs] {len(issues)} issue(s):", file=sys.stderr)
        for line in issues:
            print(f"  - {line}", file=sys.stderr)
        # We refuse to finish if any sanity check failed. Missing-source
        # warnings alone shouldn't kill the phase, but axis-mismatch must.
        hard = [i for i in issues if "axis check failed" in i]
        if hard:
            return 3

    # Convert defaultdicts back to plain dict for JSON.
    out_manifest["by_step"] = dict(out_manifest["by_step"])

    manifest_path = work_dir / "nnunet_input_sparse_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(out_manifest, f, indent=2)

    print(f"\n[sparsify_inputs] wrote {n_written} sparse CT(s) "
          f"({n_skipped_existing} already on disk; not re-written).")
    print(f"[sparsify_inputs] manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
