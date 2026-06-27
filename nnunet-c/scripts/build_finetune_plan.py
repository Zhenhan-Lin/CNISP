#!/usr/bin/env python3
"""CLI: build the finetune plan for control B/C (potholes 1 & 3).

Run AFTER `nnUNetv2_plan_and_preprocess -d <855|845>` (which produces a valid
5-channel plan) and BEFORE `nnUNetv2_preprocess` with the merged plan.

Merges Dataset835's ch0 intensity stats + target spacing + architecture into the
855/845 plan, writes it under a new plan identifier (default nnUNetPlansFinetune),
and dumps plan_before.json / plan_after.json for inspection.

Usage:
    python nnunet-c/scripts/build_finetune_plan.py --control B
    # then on the GPU box:
    nnUNetv2_preprocess -d 855 -plans_name nnUNetPlansFinetune -c 3d_fullres
    nnUNetv2_train 855 3d_fullres <fold> -p nnUNetPlansFinetune -pretrained_weights <adapted.pth>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.config import add_repo_to_syspath, load_corrector_config, get_control  # noqa: E402

add_repo_to_syspath(__file__)

from engine.plan_merge import load_plan, save_plan, merge_finetune_plan  # noqa: E402

_DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "corrector.yaml"


def _preproc_dataset_dir(dataset_id: int, dataset_name: str) -> Path:
    preproc = os.environ.get("nnUNet_preprocessed")
    if not preproc:
        raise RuntimeError("$nnUNet_preprocessed is unset (need it on the GPU box).")
    return Path(preproc) / f"Dataset{int(dataset_id):03d}_{dataset_name}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(_DEFAULT_CONFIG))
    ap.add_argument("--control", required=True, choices=["B", "C", "b", "c"])
    ap.add_argument("--ref-plan", default=None,
                    help="override 835 plan JSON path")
    ap.add_argument("--target-plan", default=None,
                    help="override 855/845 plan JSON path (default: <preproc>/"
                         "Dataset.../nnUNetPlans.json)")
    ap.add_argument("--target-plan-name", default="nnUNetPlans",
                    help="name of the freshly-planned target plan (default: %(default)s)")
    ap.add_argument("--out-plan-name", default="nnUNetPlansFinetune",
                    help="plan identifier to write (default: %(default)s)")
    ap.add_argument("--configuration", default=None,
                    help="override configuration (default: from config)")
    ap.add_argument("--binary-resampling", dest="binary_resampling",
                    action="store_true", default=True,
                    help="use the per-channel resampler (ch0 order 3, ch1-N order "
                         "0) so binary prelabels survive preprocess (default ON).")
    ap.add_argument("--no-binary-resampling", dest="binary_resampling",
                    action="store_false",
                    help="use nnUNet's default order-3 resampling for all channels "
                         "(ch1-N become soft).")
    ap.add_argument("--report-json", default=None)
    args = ap.parse_args()

    cfg = load_corrector_config(args.config, caller_file=__file__)
    control = get_control(cfg, args.control)
    configuration = args.configuration or cfg["configuration"]

    # ref (835) plan JSON to MERGE PARAMETERS from = reference_plan_json (iso05),
    # NOT reference_plan (which names the results/checkpoint folder).
    ref_plan_name = cfg.get("reference_plan_json", cfg["reference_plan"])
    if args.ref_plan:
        ref_plan_path = Path(args.ref_plan)
    else:
        ref_dir = _preproc_dataset_dir(cfg["reference_dataset_id"],
                                       cfg["reference_dataset_name"])
        ref_plan_path = ref_dir / f"{ref_plan_name}.json"

    # target (855/845) plan JSON
    target_dir = _preproc_dataset_dir(control["dataset_id"], control["dataset_name"])
    if args.target_plan:
        target_plan_path = Path(args.target_plan)
    else:
        target_plan_path = target_dir / f"{args.target_plan_name}.json"

    for p in (ref_plan_path, target_plan_path):
        if not p.is_file():
            raise FileNotFoundError(f"plan JSON not found: {p}")

    ref = load_plan(ref_plan_path)
    target = load_plan(target_plan_path)
    merged, overrides = merge_finetune_plan(ref, target, configuration)

    # Patch plan identity so preprocess/train pick up the merged plan cleanly.
    merged["plans_name"] = args.out_plan_name
    cfgs = merged["configurations"]
    old_di = cfgs[configuration].get("data_identifier", f"{args.target_plan_name}_{configuration}")
    cfgs[configuration]["data_identifier"] = f"{args.out_plan_name}_{configuration}"

    # Per-channel data resampling: ch0 order 3, binary ch1-N order 0. Requires the
    # custom fn installed under nnunetv2/preprocessing/resampling/ (see
    # nnunet-c/engine/corrector_resampling.py). The seg (label) resampler is left
    # at nnUNet's default.
    if args.binary_resampling:
        cfgs[configuration]["resampling_fn_data"] = "resample_corrector_data_to_shape"
        overrides.append(
            f"configurations.{configuration}.resampling_fn_data=resample_corrector_data_to_shape")

    out_path = target_dir / f"{args.out_plan_name}.json"
    save_plan(target, target_dir / "plan_before.json")
    save_plan(merged, target_dir / "plan_after.json")
    save_plan(merged, out_path)

    report = {
        "control": args.control.upper(),
        "ref_plan": str(ref_plan_path),
        "target_plan": str(target_plan_path),
        "out_plan": str(out_path),
        "out_plan_name": args.out_plan_name,
        "configuration": configuration,
        "old_data_identifier": old_di,
        "new_data_identifier": cfgs[configuration]["data_identifier"],
        "overrides": overrides,
        "plan_before": str(target_dir / "plan_before.json"),
        "plan_after": str(target_dir / "plan_after.json"),
    }
    print("── finetune plan merge ────────────────────────────────────")
    print(f"  ref    : {report['ref_plan']}")
    print(f"  target : {report['target_plan']}")
    print(f"  out    : {report['out_plan']}  (plans_name={args.out_plan_name})")
    print(f"  overrides ({len(overrides)}):")
    for o in overrides:
        print(f"    - {o}")
    print(f"  before/after dumped to {target_dir}/plan_before.json|plan_after.json")
    print("───────────────────────────────────────────────────────────")
    if args.report_json:
        with open(args.report_json, "w") as f:
            json.dump(report, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
