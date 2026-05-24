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
* ``${cnisp_output_basedir}/<cnisp_model_name>/runs/<cnisp_sweep_source_run_tag>/sweep_results.pkl``
  -- defaults to ``runs/atlas_gt/`` (the ceiling curve). Override with
  ``--cnisp-sweep-source`` if your nnUNet sweep should track a
  different CNISP run's (source, step) set.
* ``${work_dir}/source_to_path.json`` (written by data_prep/prepare_inputs.py)
* Per-source CT NIfTIs referenced by source_to_path.json

Outputs
-------
* ``${work_dir}/nnunet_input_step_{XX}/{sid}_0000.nii.gz`` (one per
  source, per non-trivial step)
* ``${work_dir}/nnunet_input_sparse_manifest.json``

Safety
------
Two independent checks gate every write:

1. **Axis selection + obliqueness check** (per source).
   We *don't* use ``argmax(zooms)`` because that breaks for non-axial
   acquisitions (e.g. sagittal scans whose thick voxel axis is L-R,
   not S-I). Instead, we look at the raw CT's affine and pick the
   voxel axis whose physical direction best aligns with the RAS
   direction CNISP sparsified that source's canonical patch along::

       step_axis = argmax(|affine[ras_axis, :3]|)

   where ``ras_axis`` is per-source: it comes from each row's
   ``step_axis`` field in ``sweep_results.pkl`` (CNISP writes the
   patch voxel axis it actually used). Under legacy CNISP runs with a
   uniform ``slice_step_axis: <int>`` config, every source's
   ``ras_axis`` equals that int -- equivalent to the old behaviour.
   Under ``slice_step_axis: auto`` (per-case CNISP mode), each source
   gets its own ``ras_axis``, matching its natural through-plane
   direction. If a row predates CNISP's per-case write-out, the
   config knob ``cnisp_slice_step_axis`` is used as a fallback.
   Sparsifying the chosen voxel axis then degrades the same physical
   direction CNISP did, regardless of original acquisition orientation.
   We also require the chosen axis to be *dominantly* aligned
   (projection >= ``sparse_axis_alignment_min``, default 0.95) so
   oblique grids that don't cleanly map to any voxel axis are dropped
   rather than silently mis-sparsified.
2. **Magnitude check** (per source × step). After axis selection,
   compare ``zooms[step_axis] * step`` to CNISP's
   ``effective_resolution_mm`` (mean over OD/OS):
   * ≤ ``sparse_eff_res_tolerance`` (default 5%): silently OK.
   * ≤ ``sparse_eff_res_max_drift`` (default 30%): proceed with a warn.
     This handles cases where CNISP's canonical patch lives on a
     different grid than the raw CT (e.g. chk_* QA-kept old-nnUNet
     preds were resampled to ~1.25 mm iso while the raw CT is 1 mm iso).
     The nnUNet sparse-CT then sits at the patch's eff_res row in the
     manifest but the *actual* sparsified spacing differs slightly.
   * Above ``sparse_eff_res_max_drift``: hard-skip the step.

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
from typing import Dict, List, Optional, Tuple

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
) -> Tuple[
    Dict[str, List[int]],
    Dict[Tuple[str, int], float],
    Dict[Tuple[str, int], Optional[int]],
]:
    """Read sweep_results.pkl, group by source_id, drop step=1.

    Returns
    -------
    by_source : {source_id: sorted list of step_sizes > 1}
    eff_res   : {(source_id, step_size): mean effective_resolution_mm
                 averaged over the two eyes when both ran}
    step_axis : {(source_id, step_size): canonical RAS axis CNISP used
                 (= patch voxel axis after canonical alignment). Same
                 across OD/OS for a given source (canonical_align
                 preserves orientation). ``None`` if the sweep was
                 produced by a pre-per-case CNISP build that didn't
                 emit ``step_axis`` on rows -- callers should fall back
                 to the cnisp_slice_step_axis config knob.}
    """
    if not sweep_pkl.exists():
        raise FileNotFoundError(
            f"sweep_results.pkl not found: {sweep_pkl}\n"
            f"  Run `cnisp-infer` first so we know which steps to sparsify."
        )
    with open(sweep_pkl, "rb") as f:
        rows: List[dict] = pickle.load(f)

    eff_accum: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    axis_accum: Dict[Tuple[str, int], List[int]] = defaultdict(list)
    for r in rows:
        cn = r.get("casename")
        if cn is None:
            continue
        if not (cn.endswith("_OD") or cn.endswith("_OS")):
            continue
        sid = cn[:-3]
        step = int(r["step_size"])
        eff_accum[(sid, step)].append(float(r["effective_resolution_mm"]))
        if "step_axis" in r and r["step_axis"] is not None:
            axis_accum[(sid, step)].append(int(r["step_axis"]))

    eff_res = {k: float(np.mean(v)) for k, v in eff_accum.items()}

    # Per (sid, step) consensus axis: OD/OS should agree because
    # canonical_align reorients both eyes to the same RAS axes.
    step_axis: Dict[Tuple[str, int], Optional[int]] = {}
    for key in eff_res:
        axes = axis_accum.get(key, [])
        if not axes:
            step_axis[key] = None
        elif len(set(axes)) == 1:
            step_axis[key] = axes[0]
        else:
            # Eyes disagree (shouldn't happen if canonical alignment is
            # correct). Picking the mode is safer than failing the whole
            # sweep; surface a warning so the user notices.
            from collections import Counter
            most_common, _ = Counter(axes).most_common(1)[0]
            print(
                f"[sparsify_inputs] WARN: {key} has inconsistent "
                f"step_axis values across rows ({sorted(set(axes))}); "
                f"using mode={most_common}.",
                file=sys.stderr,
            )
            step_axis[key] = most_common

    by_source: Dict[str, List[int]] = defaultdict(list)
    for (sid, step) in eff_res:
        if step <= 1:
            continue
        by_source[sid].append(step)
    for sid in by_source:
        by_source[sid] = sorted(set(by_source[sid]))
    return dict(by_source), eff_res, step_axis


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
    ap.add_argument("--cnisp-sweep-source", default="atlas_gt",
                    help="run_tag under output_basedir/<model>/runs/ "
                         "whose sweep_results.pkl drives the (source, "
                         "step) set. Default atlas_gt: the deployment "
                         "curve always re-uses the ceiling curve's "
                         "sweep so a single nnUNet sparse-CT sweep "
                         "covers both stories.")
    args = ap.parse_args()

    cfg = _load_yaml(Path(args.config))
    cnisp_paths = _load_yaml(Path(cfg["cnisp_paths_yaml"]))

    work_dir = Path(cfg["work_dir"])
    cnisp_run_base = (
        Path(cnisp_paths["output_basedir"])
        / cfg["cnisp_model_name"]
        / "runs"
        / args.cnisp_sweep_source
    )
    sweep_pkl = cnisp_run_base / "sweep_results.pkl"
    # Backward compat: pre-Option-C runs wrote sweep_results.pkl directly
    # under output_basedir/<model>/ without a runs/<run_tag>/ wrapper.
    if not sweep_pkl.exists():
        legacy = (Path(cnisp_paths["output_basedir"])
                  / cfg["cnisp_model_name"] / "sweep_results.pkl")
        if legacy.exists():
            print(f"[sparsify_inputs] {sweep_pkl} not found; "
                  f"falling back to legacy layout at {legacy}")
            sweep_pkl = legacy
    soft_tol = float(cfg.get("sparse_eff_res_tolerance", 0.05))
    drift_tol = float(cfg.get("sparse_eff_res_max_drift", 0.30))
    canonical_axis = int(cfg.get("cnisp_slice_step_axis", 2))
    align_min = float(cfg.get("sparse_axis_alignment_min", 0.95))

    source_to_path = work_dir / "source_to_path.json"
    if not source_to_path.exists():
        print(f"[sparsify_inputs] {source_to_path} missing -- "
              f"run nnunet/data_prep/prepare_inputs.py first.",
              file=sys.stderr)
        return 2
    with open(source_to_path) as f:
        manifest_in = json.load(f)

    by_source, eff_res_idx, sweep_axis_idx = _build_sweep_set(sweep_pkl)

    n_with_axis = sum(1 for v in sweep_axis_idx.values() if v is not None)
    print(f"[sparsify_inputs] sweep_results.pkl: {sweep_pkl}")
    print(f"[sparsify_inputs] sources in sweep:   {len(by_source)}")
    print(f"[sparsify_inputs] sources in manifest:{len(manifest_in)}")
    print(f"[sparsify_inputs] work_dir:           {work_dir}")
    print(f"[sparsify_inputs] step_axis from sweep rows: "
          f"{n_with_axis}/{len(sweep_axis_idx)} (rest fall back to "
          f"config cnisp_slice_step_axis = {canonical_axis})")
    print(f"[sparsify_inputs] axis alignment min: {align_min:.2%} "
          f"(reject oblique grids below this)")
    print(f"[sparsify_inputs] eff-res soft tol:   {soft_tol:.2%} "
          f"(silently OK)")
    print(f"[sparsify_inputs] eff-res max drift:  {drift_tol:.2%} "
          f"(warn-and-proceed; hard skip above this)")

    out_manifest: Dict = {
        "step_axis_per_source": {},
        "by_step": defaultdict(dict),
    }

    n_written = 0
    n_skipped_existing = 0
    n_skipped_oblique = 0  # sources whose grid is too oblique to RAS
    n_skipped_drift = 0    # steps whose magnitude drift exceeds drift_tol
    n_drift_warn = 0       # steps in (soft_tol, drift_tol] -> warn-and-proceed
    issues: List[str] = []
    warnings: List[str] = []

    for sid, steps in sorted(by_source.items()):
        info = manifest_in.get(sid)
        if info is None:
            issues.append(f"{sid}: in sweep_results but not in source_to_path.json")
            continue
        ct_path = Path(info["ct_image_path"])
        if not ct_path.exists():
            issues.append(f"{sid}: ct_image_path missing on disk: {ct_path}")
            continue

        # Per-source RAS direction CNISP sparsified along. Under legacy
        # ``slice_step_axis: <int>`` this is the same for every source;
        # under ``slice_step_axis: auto`` each source has its own value
        # (= patch argmax(spacing), the scan's natural through-plane
        # direction after canonical alignment). All steps for a given
        # source share the same canonical patch and thus the same axis.
        per_step_axes = {s: sweep_axis_idx.get((sid, s)) for s in steps}
        ras_axes_present = sorted({a for a in per_step_axes.values() if a is not None})
        if not ras_axes_present:
            # Legacy sweep_results.pkl without step_axis on rows: fall
            # back to the single config knob.
            ras_axis = canonical_axis
        elif len(ras_axes_present) == 1:
            ras_axis = ras_axes_present[0]
        else:
            issues.append(
                f"{sid}: inconsistent RAS axis across steps "
                f"{per_step_axes}. Skipping source -- this should not "
                f"happen since all steps share the same canonical patch."
            )
            continue

        img = nib.load(str(ct_path))
        zooms = np.asarray(img.header.get_zooms()[:3], dtype=np.float32)
        affine = img.affine

        # Find the raw CT voxel axis whose physical direction best aligns
        # with the RAS direction CNISP sparsified the patch along.
        # Affine row `ras_axis` projects each voxel column onto that
        # world direction; we pick the voxel axis with the largest
        # projection (normalised by column norm to take out spacing).
        col_norms = np.linalg.norm(affine[:3, :3], axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            projections = np.where(
                col_norms > 0,
                np.abs(affine[ras_axis, :3]) / col_norms,
                0.0,
            )
        step_axis = int(np.argmax(projections))
        alignment = float(projections[step_axis])
        if alignment < align_min:
            # Oblique voxel grid: no single voxel axis cleanly maps to
            # the chosen RAS direction, so any per-axis sparsification
            # would degrade a mix of physical directions.
            issues.append(
                f"{sid}: oblique voxel grid -- best axis is {step_axis} "
                f"but its alignment to RAS axis {ras_axis} is only "
                f"{alignment:.2%} < {align_min:.2%}. Skipping all "
                f"{len(steps)} step(s) for this source."
            )
            n_skipped_oblique += 1
            continue
        out_manifest["step_axis_per_source"][sid] = step_axis
        out_manifest.setdefault("ras_axis_per_source", {})[sid] = ras_axis
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
            if rel > drift_tol:
                # Magnitude drift too large even after axis OK: skip just
                # this step (other steps for the same source might still
                # be fine).
                issues.append(
                    f"{sid} step={step}: eff-res drift {rel:.2%} > "
                    f"max_drift {drift_tol:.2%}. raw CT spacing[axis "
                    f"{step_axis}] * step = {expected_eff_res:.3f} mm, "
                    f"CNISP eff_res = {cnisp_eff_res:.3f} mm. Skipping "
                    f"this (source, step)."
                )
                n_skipped_drift += 1
                continue
            if rel > soft_tol:
                # Axis matches but spacings differ -- usually because
                # CNISP's canonical patch was derived from a resampled
                # label (e.g. chk_* QA-kept old-nnUNet pred on a coarser
                # grid). Sparsifying the raw CT here is still physically
                # correct (same S-I direction); only the numeric eff_res
                # drifts slightly relative to CNISP's row.
                warnings.append(
                    f"{sid} step={step}: eff-res drift {rel:.2%} "
                    f"(raw {expected_eff_res:.3f} mm vs CNISP "
                    f"{cnisp_eff_res:.3f} mm). axis OK, proceeding."
                )
                n_drift_warn += 1

            if dst.exists():
                n_skipped_existing += 1
            else:
                sparse_arr, new_affine = _sparsify_one_ct(
                    ct_path=ct_path,
                    step_axis=step_axis,
                    step_size=step,
                )
                # Post-write sanity: the affine column we scaled by
                # step_size must yield spacing == base_spacing*step.
                # Independent of CNISP's eff_res.
                sanity_eff_res = _eff_res_from_affine(new_affine, step_axis)
                rel2 = abs(sanity_eff_res - expected_eff_res) / expected_eff_res
                assert rel2 <= 1e-3, (
                    f"{sid} step={step}: post-write affine spacing "
                    f"{sanity_eff_res:.3f} mm disagrees with "
                    f"base_spacing*step={expected_eff_res:.3f} mm "
                    f"(rel {rel2:.2%}). Bug in affine scaling."
                )
                out_img = nib.Nifti1Image(sparse_arr, new_affine)
                out_img.set_qform(new_affine)
                out_img.set_sform(new_affine)
                nib.save(out_img, str(dst))
                n_written += 1

            out_manifest["by_step"][f"{step:02d}"][sid] = {
                "input": str(dst),
                # eff_res_mm reflects CNISP's row so summary buckets line
                # up across methods; actual_eff_res_mm preserves the raw
                # CT's true post-sparsification spacing for traceability.
                "eff_res_mm": round(cnisp_eff_res, 4),
                "actual_eff_res_mm": round(expected_eff_res, 4),
                "step_axis": step_axis,
            }

    if warnings:
        print(
            f"\n[sparsify_inputs] {len(warnings)} eff-res drift warning(s) "
            f"(axis OK, proceeded):", file=sys.stderr,
        )
        for line in warnings:
            print(f"  · {line}", file=sys.stderr)
    if issues:
        print(f"\n[sparsify_inputs] {len(issues)} issue(s):", file=sys.stderr)
        for line in issues:
            print(f"  - {line}", file=sys.stderr)

    # Convert defaultdicts back to plain dict for JSON.
    out_manifest["by_step"] = dict(out_manifest["by_step"])

    manifest_path = work_dir / "nnunet_input_sparse_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(out_manifest, f, indent=2)

    print(
        f"\n[sparsify_inputs] wrote {n_written} sparse CT(s) "
        f"({n_skipped_existing} already on disk; not re-written).\n"
        f"[sparsify_inputs] oblique sources skipped:           {n_skipped_oblique}\n"
        f"[sparsify_inputs] drift-skipped (source, step) pairs:  {n_skipped_drift}\n"
        f"[sparsify_inputs] drift-warn (proceeded) pairs:        {n_drift_warn}"
    )
    print(f"[sparsify_inputs] manifest: {manifest_path}")
    # Exit non-zero only if we ended up with nothing to compare; otherwise
    # let downstream phases handle the (source, step) gaps gracefully.
    if not out_manifest["by_step"]:
        print(
            "[sparsify_inputs] manifest is empty -- nothing to sweep. "
            "Check that sweep_results.pkl matches your test cases.",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
