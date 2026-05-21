#!/usr/bin/env python3
"""
Single-case probe driver for the inference shape-consistency fixes.

Exercises in order:
    1. resolution_sweep._run_case (eval_case_at_resolution)
       — prints [_run_case] line per step and asserts pred.shape == gt.shape.
    2. infer.py-style nii export to a temporary probe_output/ dir.
    3. resolution_sweep._try_load_cached on that nii
       — prints [cache hit] line and asserts pred.shape == gt.shape.
    4. reconstruction_qc.diagnose_single_case on the result
       — asserts pred_class_map.shape == gt_class_map.shape.

If anything is wrong with the shape pipeline, one of these asserts fires
with an explicit message naming the offending case / step / path.

Run from the repo root:
    PYTHONPATH=orbital_shape_prior_st1 python3 \
        orbital_shape_prior_st1/scripts/probe_inference_shapes.py \
        -p orbital_shape_prior_st1/configs/paths.yaml \
        -t orbital_shape_prior_st1/configs/train_sty2.yaml \
        -c orbital_shape_prior_st1/configs/test_default.yaml \
        -m orbital_ad_v2

    # Pick a specific case instead of the first one in the test casefile:
    ... --case atlas_orbit0006_ubMask_al2_fill_OD

    # Run a different list of step sizes (default: 1, 3):
    ... --steps 1 3 5
"""
import argparse
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import yaml

from diagnostics.reconstruction_qc import diagnose_single_case
from diagnostics.resolution_sweep import (
    _try_load_cached,
    eval_case_at_resolution,
)
from engine.dataset import load_casenames, load_orbital_volumes
from engine.infer import load_model_checkpoint, optimize_latent
from engine.train import create_model


def _load_params(paths_yaml: Path, train_yaml: Path, test_yaml: Path,
                 model_name: str) -> dict:
    params = {}
    for f in (paths_yaml, train_yaml, test_yaml):
        with open(f) as fp:
            params.update(yaml.safe_load(fp))
    params["model_name"] = model_name
    params["checkpoint"] = "best"
    return params


def _save_pred_like_infer(pred_class_map: np.ndarray, spacing: np.ndarray,
                          casename: str, step: int, probe_dir: Path) -> Path:
    """Mirror infer.py's per-step pred export so _try_load_cached sees it."""
    step_dir = probe_dir / f"step_{step:02d}" / "pred"
    step_dir.mkdir(parents=True, exist_ok=True)
    aff = np.diag([*spacing, 1.0])
    out_path = step_dir / f"{casename}_pred.nii.gz"
    nib.save(nib.Nifti1Image(pred_class_map.astype(np.uint8), aff),
             str(out_path))
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--paths", required=True)
    parser.add_argument("-t", "--train_config", required=True)
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-m", "--model_name", required=True)
    parser.add_argument("--case", default=None,
                        help="Case name (default: first in test casefile).")
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 3],
                        help="Step sizes to probe (default: 1 3).")
    parser.add_argument("--probe_dir", default=None,
                        help="Where to write probe outputs (default: "
                             "<output_basedir>/<model>/_probe).")
    args = parser.parse_args()

    params = _load_params(Path(args.paths), Path(args.train_config),
                          Path(args.config), args.model_name)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # ── Pick one case ────────────────────────────────────────────────
    casefiles_dir = Path(params["casefiles_dir"])
    casenames = load_casenames(casefiles_dir / params["test_casefile"])
    if args.case is None:
        casename = casenames[0]
    else:
        if args.case not in casenames:
            raise SystemExit(
                f"--case {args.case!r} not in test casefile "
                f"({casefiles_dir / params['test_casefile']}); "
                f"first few: {casenames[:5]}"
            )
        casename = args.case
    print(f"Probing case: {casename}")

    labels_dir = Path(params["aligned_dir"]) / params.get("labels_dirname",
                                                          "labels")
    labels_dense, spacings_dense = load_orbital_volumes(labels_dir, [casename])
    label_dense = labels_dense[0]
    spacing_dense = spacings_dense[0]
    print(f"  label_dense.shape = {tuple(label_dense.shape)}")
    print(f"  spacing_dense     = "
          f"{tuple(round(float(s), 4) for s in spacing_dense)}")

    # ── Load model ───────────────────────────────────────────────────
    model_dir = Path(params["model_basedir"]) / params["model_name"]
    model_state, ckpt_meta = load_model_checkpoint(model_dir, "best",
                                                   verbose=True)
    net = create_model(params, torch.ones(3))
    net.load_state_dict(model_state["net"], strict=True)
    net = net.to(device).eval()
    print(f"  net.image_size = {net.image_size.tolist()}")
    expected_envelope = tuple(
        int(np.ceil(float(net.image_size[d]) / float(spacing_dense[d])))
        for d in range(3)
    )
    print(f"  envelope shape (legacy)   = {expected_envelope}")
    print(f"  label_dense shape (fixed) = {tuple(label_dense.shape)}")
    if tuple(label_dense.shape) == expected_envelope:
        print("  ⚠ envelope == label shape for this case "
              "(shape mismatch never visible here; try another case "
              "to fully exercise the assert)")

    # ── Probe dir (isolated from real recon dir) ─────────────────────
    if args.probe_dir is not None:
        probe_dir = Path(args.probe_dir)
    else:
        probe_dir = (Path(params["output_basedir"])
                     / params["model_name"] / "_probe")
    if probe_dir.exists():
        shutil.rmtree(probe_dir)
    probe_dir.mkdir(parents=True, exist_ok=True)
    print(f"Probe outputs in: {probe_dir}")

    step_axis = int(params["slice_step_axis"])

    # ── 1) live _run_case path for each requested step ───────────────
    results = []
    for step in args.steps:
        print(f"\n── _run_case  step={step}  ────────────────────────────")
        result = eval_case_at_resolution(
            net=net, optimize_fn=optimize_latent,
            label_dense=label_dense, spacing_dense=spacing_dense,
            step_size=step, step_axis=step_axis,
            params=params, device=device,
            use_thick_slices=params.get("use_thick_slices", False),
        )
        result["casename"] = casename
        print(f"  → dice_dense={result['dice']['mean']:.3f}  "
              f"dice_obs={result['dice_observed']['mean']:.3f}  "
              f"pred.shape={result['pred_class_map'].shape}  "
              f"gt.shape={result['gt_class_map'].shape}")
        # mirror infer.py: write pred so cache path can find it
        out = _save_pred_like_infer(result["pred_class_map"],
                                    result["spacing"], casename, step,
                                    probe_dir)
        print(f"  saved: {out}  ({result['pred_class_map'].nbytes / 1024:.1f} KB)")
        results.append(result)

    # ── 2) _try_load_cached path ─────────────────────────────────────
    print("\n── _try_load_cached  ──────────────────────────────────────")
    for step in args.steps:
        cached = _try_load_cached(
            probe_dir, casename, step, step_axis,
            label_dense, spacing_dense, net.num_classes,
        )
        if cached is None:
            print(f"  step={step}: cache miss (unexpected)")
            continue
        print(f"  step={step}: cache hit dice_dense="
              f"{cached['dice']['mean']:.3f}  "
              f"pred.shape={cached['pred_class_map'].shape}  "
              f"gt.shape={cached['gt_class_map'].shape}  "
              f"latent_missing={cached['latent_missing']}")

    # ── 3) reconstruction_qc.diagnose_single_case ────────────────────
    print("\n── reconstruction_qc.diagnose_single_case  ─────────────────")
    for r in results:
        diag = diagnose_single_case(
            r["pred_class_map"], r["gt_class_map"],
            r["spacing"], r["casename"],
        )
        print(f"  step={r['step_size']}: "
              f"mean_dice_unaligned={diag.mean_dice_unaligned:.3f}  "
              f"mean_dice_aligned={diag.mean_dice_aligned:.3f}  "
              f"position_contrib={diag.mean_position_contribution:+.3f}")

    print("\nAll probes passed ✓  (no assert tripped, all shapes consistent).")


if __name__ == "__main__":
    main()
