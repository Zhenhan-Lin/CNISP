#!/usr/bin/env python3
"""CLI: assemble the corrector's inference channels (imagesTs) for a control.

The cascade for a control is:
    degraded CT --(nnUNet 835)--> sparse mask --[C: CNISP test-opt]--> prelabel
    --> assemble 5ch imagesTs (this script) --> nnUNetv2_predict(control model)

This script performs the genuinely new middle step: resolve each (source, step)'s
degraded CT + prelabel (via the SAME lib.prelabel logic as training) and assemble
the channels onto the 835 plan-spacing grid (no GT). The upstream nnUNet/CNISP
predictions are produced by run_full_pipeline.sh Stages 1 & 3; the final
nnUNetv2_predict is launched by run_predict.sh.

Usage:
    python nnunet-c/scripts/predict_cascade.py --control C --split test \
        --out-images-dir /tmp/corrC_imagesTs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.config import add_repo_to_syspath, load_corrector_config, get_control, structures  # noqa: E402

add_repo_to_syspath(__file__)

from lib import caselist as _cl  # noqa: E402
from lib import labels as _lab  # noqa: E402
from lib import prelabel as _pl  # noqa: E402
from lib import resample as _rs  # noqa: E402
from lib import channels as _ch  # noqa: E402
from lib import staging as _st  # noqa: E402

_DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "corrector.yaml"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(_DEFAULT_CONFIG))
    ap.add_argument("--control", required=True, choices=["A", "B", "C", "a", "b", "c"])
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--out-images-dir", required=True,
                    help="where to write {case}_0000..000N.nii.gz for nnUNetv2_predict")
    args = ap.parse_args()

    cfg = load_corrector_config(args.config, caller_file=__file__)
    control = get_control(cfg, args.control)
    structs = structures(cfg)
    target_spacing = _rs.resolve_target_spacing(cfg)

    sources = (_cl.test_sources(cfg) if args.split == "test"
               else _cl.corrector_train_sources(cfg))
    source_infos = _lab.resolve_source_infos(cfg, sources)

    out_images = Path(args.out_images_dir)
    out_images.mkdir(parents=True, exist_ok=True)
    manifest = {}
    n = 0
    for sid in sources:
        si = source_infos[sid]
        for step in _pl.available_steps(cfg, sid, control, si):
            cid = _st.case_id(sid, step)
            prelabel_path = None
            prelabel_stv = None
            if control["prelabel_source"] != "none":
                pre = _pl.resolve_prelabel(cfg, control, sid, step, si)
                prelabel_path = Path(pre["path"])
                prelabel_stv = pre["struct_to_value"]
            summary = _ch.assemble_inference_case(
                case_id=cid,
                ct_path=_pl.degraded_ct_path(cfg, sid, step),
                target_spacing=target_spacing,
                n_channels=int(control["n_channels"]),
                structures=structs,
                images_dir=out_images,
                experiment=cfg["experiment"],
                prelabel_path=prelabel_path,
                prelabel_struct_to_value=prelabel_stv,
            )
            manifest[cid] = {"source_id": sid, "step": step, **summary}
            n += 1
            print(f"  [assemble-infer] {cid}: shape={summary['shape']}")

    mf_path = out_images / "predict_cascade_manifest.json"
    with open(mf_path, "w") as f:
        json.dump({"control": args.control.upper(), "split": args.split,
                   "cases": manifest}, f, indent=2)
    print(f"[predict_cascade] assembled {n} case(s) -> {out_images}")
    print(f"[predict_cascade] manifest -> {mf_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
