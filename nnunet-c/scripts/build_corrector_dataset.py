#!/usr/bin/env python3
"""Build the corrector's 5-channel nnUNet raw dataset from the data/ tree.

Consumes the self-contained corrector data layout (NOT the work_dir sweep):
    ch0  = data/images/{case}_step{XX}_0000.nii.gz      (degraded CT, pinned)
    ch1..ch4 = control prelabel split into per-class binaries:
        control B -> data/nnunet_pred/{case}_step{XX}.nii.gz   (835 pred, {1,2,3,4})
        control C -> data/cnisp_pred/{case}_step{XX}.nii.gz    (CNISP,   {1,2,3,4})
    label = full-res Dataset835 prediction (manifest gt_candidate_pred), the
            pseudo-GT target (keep=False images have no manual GT), {0..4}.
Everything is resampled to the 835 plan-spacing grid (pothole-2 a-ii) so nnUNet's
preprocess resample is a no-op and the binary channels stay {0,1}.

Each (case_id, step) -> one nnUNet case `corr_{case_id}_step{XX}`. Only samples
whose ch0 + prelabel (+ optionally cnisp, to mirror C) + gt all exist are built,
so a capped CNISP run (e.g. --max-samples 300) yields a matching dataset.

Usage:
    python nnunet-c/scripts/build_corrector_dataset.py --control C
    python nnunet-c/scripts/build_corrector_dataset.py --control B --require-cnisp
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.config import add_repo_to_syspath, load_corrector_config, get_control  # noqa: E402

add_repo_to_syspath(__file__)

import numpy as np  # noqa: E402
import nibabel as nib  # noqa: E402

from engine.convert import convert_case  # noqa: E402  (the SINGLE converter)
from engine.build_dataset import _raw_root, _dataset_dir, _write_dataset_json  # noqa: E402

_DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "corrector.yaml"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(_DEFAULT_CONFIG))
    ap.add_argument("--control", required=True, choices=["B", "C", "b", "c"])
    ap.add_argument("--require-cnisp", action="store_true",
                    help="also require data/cnisp_pred to exist for each sample "
                         "(use for control B so it matches C's capped case set).")
    ap.add_argument("--raw-root", default=None, help="override $nnUNet_raw")
    args = ap.parse_args()

    cfg = load_corrector_config(args.config, caller_file=__file__)
    control = get_control(cfg, args.control)
    if control.get("external"):
        raise RuntimeError(f"control {args.control.upper()} is external (Dataset"
                           f"{control['dataset_id']}); nothing to build.")
    if int(control["n_channels"]) != 5:
        raise RuntimeError("this builder is for the 5-channel controls (B/C).")

    cd = cfg["corrector_data"]
    res = cfg["_resolved"]
    data_root = Path(cd["data_root"])
    data_root = data_root if data_root.is_absolute() else (res["repo_root"] / data_root)
    images_dirname = cd.get("images_dirname", "images")
    images_dir = data_root / images_dirname
    pre_dirname = (cd.get("nnunet_pred_dirname", "nnunet_pred")
                   if control["prelabel_source"] == "nnunet"
                   else cd.get("cnisp_pred_dirname", "cnisp_pred"))
    prelabel_dir = data_root / pre_dirname
    cnisp_dir = data_root / cd.get("cnisp_pred_dirname", "cnisp_pred")
    manifest_path = data_root / "corrector_data_manifest.json"
    if not manifest_path.is_file():
        print(f"[build] {manifest_path} missing -- run build_corrector_data.py first.",
              file=sys.stderr)
        return 2
    manifest = json.load(open(manifest_path))

    raw = _raw_root(args.raw_root)
    ds_dir = _dataset_dir(raw, control)
    images_out = ds_dir / "imagesTr"
    labels_out = ds_dir / "labelsTr"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    print(f"[build] control={args.control.upper()} -> {ds_dir}")
    print(f"[build] ch0={images_dir}  prelabel={prelabel_dir}  grid=GT original (per source)")
    print(f"[build]   (nnUNet resamples this original grid -> iso 0.5 plan at preprocess)")

    assembled, skipped = [], 0
    for case_id, entry in sorted(manifest["cases"].items()):
        gt = entry.get("gt_candidate_pred", "")
        if not gt or not Path(gt).exists():
            skipped += 1
            continue
        # Common grid = the GT's ORIGINAL native grid: label = GT (shared across
        # steps), ch1-4 = CNISP native mask (already on this grid), ch0 = degraded
        # upsampled to it. nnUNet then resamples original -> iso 0.5 at preprocess.
        gt_img = nib.load(str(gt))
        ref_grid = (gt_img.shape[:3], np.asarray(gt_img.affine))
        for step_s, sinfo in entry.get("steps", {}).items():
            if not sinfo.get("kept"):
                continue
            step = int(step_s)
            ct = images_dir / f"{case_id}_step{step:02d}_0000.nii.gz"
            pre = prelabel_dir / f"{case_id}_step{step:02d}.nii.gz"
            if not ct.exists() or not pre.exists():
                skipped += 1
                continue
            if args.require_cnisp and not (cnisp_dir / f"{case_id}_step{step:02d}.nii.gz").exists():
                skipped += 1
                continue
            cid = f"corr_{case_id}_step{step:02d}"
            summary = convert_case(
                case_id=cid, ct_path=ct, prelabel_path=pre, ref_grid=ref_grid,
                experiment=cfg["experiment"], images_dir=images_out,
                gt_path=Path(gt), labels_dir=labels_out,
                degraded_marker=f"/{images_dirname}/",
            )
            assembled.append(summary)
            print(f"  {cid}: shape={summary['shape']} labels={summary['label_values']}")

    _write_dataset_json(ds_dir, control, cfg, num_training=len(assembled))
    with open(ds_dir / "corrector_build_manifest.json", "w") as f:
        json.dump({"control": args.control.upper(), "n": len(assembled),
                   "cases": assembled}, f, indent=2)
    print(f"[build] wrote {len(assembled)} case(s); skipped {skipped} (missing files).")
    print(f"[build] dataset -> {ds_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
