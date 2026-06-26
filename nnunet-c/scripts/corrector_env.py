#!/usr/bin/env python3
"""Print shell-evalable KEY=VALUE lines for a corrector control.

Used by the shell wrappers so all paths/identities have a single source of truth
(corrector.yaml). Example:

    eval "$(python3 nnunet-c/scripts/corrector_env.py --control B)"
    echo "$CTRL_DATASET_ID $REF_CKPT"

Resolves the Dataset835 checkpoint path from $nnUNet_results when available.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _q(v) -> str:
    return f'"{v}"'


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from lib.config import load_corrector_config, get_control  # lazy

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config",
                    default=str(Path(__file__).resolve().parents[1] / "configs" / "corrector.yaml"))
    ap.add_argument("--control", required=True)
    args = ap.parse_args()

    cfg = load_corrector_config(args.config, caller_file=__file__)
    ctrl = get_control(cfg, args.control)
    res = cfg["_resolved"]

    ref_id = int(cfg["reference_dataset_id"])
    ref_name = cfg["reference_dataset_name"]
    ref_plan = cfg["reference_plan"]
    configuration = cfg["configuration"]
    trainer = cfg["trainer"]
    ref_fold = cfg["reference_fold"]

    ref_ckpt = ""
    results = os.environ.get("nnUNet_results")
    if results:
        ref_ckpt = str(
            Path(results)
            / f"Dataset{ref_id:03d}_{ref_name}"
            / f"{trainer}__{ref_plan}__{configuration}"
            / f"fold_{ref_fold}" / "checkpoint_final.pth"
        )

    lines = {
        "CONTROL": args.control.upper(),
        "CTRL_DATASET_ID": int(ctrl["dataset_id"]),
        "CTRL_DATASET_NAME": ctrl["dataset_name"],
        "N_CHANNELS": int(ctrl["n_channels"]),
        "PRELABEL_SOURCE": ctrl["prelabel_source"],
        "EXTERNAL": "1" if ctrl.get("external") else "0",
        "REF_DATASET_ID": ref_id,
        "REF_DATASET_NAME": ref_name,
        "REF_PLAN": ref_plan,
        "CONFIGURATION": configuration,
        "TRAINER": trainer,
        "REF_FOLD": ref_fold,
        "REF_CKPT": ref_ckpt,
        "EXPERIMENT": cfg["experiment"],
        "RUN_TAG": cfg.get("run_tag", ""),
        "CNISP_MODEL_NAME": cfg["cnisp_model_name"],
        "CNISP_TRAIN_YAML": cfg["cnisp_train_yaml"],
        "CNISP_TEST_YAML": cfg.get("cnisp_test_yaml", "test_corrector.yaml"),
        "CASEFILES_DIR": str(res["casefiles_dir"]),
        "CORRECTOR_TRAIN_CASEFILE": cfg["corrector_train_casefile"],
        "WORK_DIR": str(res["work_dir"]),
        "CNISP_PATHS_YAML": str(res["cnisp_paths_yaml"]),
        "CNISP_MODEL_BASEDIR": str(res["cnisp_model_basedir"]),
        "CNISP_OUTPUT_BASEDIR": str(res["cnisp_output_basedir"]),
        "NNUNET_CONFIG_YAML": str(
            (res["repo_root"] / cfg["nnunet_config_yaml"]).resolve()
        ),
    }

    cd = cfg.get("corrector_data")
    if cd:
        root = res["repo_root"]
        data_root = Path(cd["data_root"])
        data_root = data_root if data_root.is_absolute() else (root / data_root)
        lines["DATA_ROOT"] = str(data_root)
        lines["DATA_IMAGES"] = str(data_root / cd.get("images_dirname", "images"))
        lines["DATA_NNUNET_PRED"] = str(data_root / cd.get("nnunet_pred_dirname", "nnunet_pred"))
        lines["DATA_CNISP_PRED"] = str(data_root / cd.get("cnisp_pred_dirname", "cnisp_pred"))
    for k, v in lines.items():
        print(f"{k}={_q(v)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
