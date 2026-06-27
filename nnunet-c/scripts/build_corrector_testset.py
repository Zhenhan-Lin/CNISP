#!/usr/bin/env python3
"""Assemble the 5-channel nnUNet-C TEST input for the CNISP test set.

The CNISP test cases already have (from the earlier work_dir / run_pipeline
sweep): degraded CTs under work_dir/input, Dataset835 sparse preds under
work_dir/prediction, and canonical-aligned patches + metadata under the CNISP
aligned_dir. CNISP test inference (032 / run_corrector_cnisp) writes the per
(source, step) mask remapped to nnUNet {1,2,3,4}.

This script gathers those into ``nnunet-c/test_input/<control>/imagesTs`` in the
nnUNet naming scheme, MIRRORING the training build exactly:

    ref grid = the source's ORIGINAL/dense grid (= CNISP native-mask grid = GT grid)
    ch0      = degraded CT (work_dir/input) resampled UP to that grid (order 3)
    ch1..ch4 = the prelabel mask split into per-structure binaries (order 0):
                 control B -> work_dir/prediction/<exp>/sparse_step_XX_native/<sid>.nii.gz
                 control C -> <cnisp-test-dir>/<gtstem>_step{XX}.nii.gz   (from 032)

nnUNet then resamples original -> iso 0.5 (nnUNetPlansFinetune) at predict via
the per-channel resampler (ch0 order 3, ch1-4 order 0) -- identical to training.

A sidecar ``test_cases_map.json`` records corr_case -> {source_id, step,
gt_label_path} so predictions can be scored against the native GT afterwards.

Usage:
    python nnunet-c/scripts/build_corrector_testset.py --control C \
        --cnisp-test-dir nnunet-c/data/cnisp_pred_test
    python nnunet-c/scripts/build_corrector_testset.py --control B
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.config import load_corrector_config, get_control, add_repo_to_syspath  # noqa: E402

add_repo_to_syspath(__file__)

import numpy as np  # noqa: E402
import nibabel as nib  # noqa: E402

from lib import channels as _ch  # noqa: E402
from lib import prelabel as _pre  # noqa: E402
from lib.labels import resolve_source_infos, NNUNET_LABELS  # noqa: E402

_DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "corrector.yaml"
_STRUCTS = ["ON", "Recti", "Globe", "Fat"]   # fixed nnUNet channel order


def _read_source_ids(casefile: Path) -> list:
    """Read a CNISP casefile (casenames) -> unique source_ids (strip _OD/_OS)."""
    sids = []
    for line in casefile.read_text().splitlines():
        cn = line.strip()
        if not cn or cn.startswith("#"):
            continue
        sid = cn[:-3] if cn.endswith(("_OD", "_OS")) else cn
        if sid not in sids:
            sids.append(sid)
    return sids


def _gt_stem(gt_label_path: Path) -> str:
    return gt_label_path.name.replace(".nii.gz", "").replace(".nii", "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(_DEFAULT_CONFIG))
    ap.add_argument("--control", required=True, choices=["A", "B", "C", "a", "b", "c"])
    ap.add_argument("--casefile", default=None,
                    help="casefile under casefiles_dir (default: CNISP test_cases.txt)")
    ap.add_argument("--steps", default="3,6,9,12",
                    help="step sizes to assemble (default 3,6,9,12)")
    ap.add_argument("--cnisp-test-dir", default=None,
                    help="control C: dir of 032 test masks <gtstem>_step{XX}.nii.gz "
                         "(default: data/cnisp_pred_test)")
    ap.add_argument("--out", default=None,
                    help="output root (default: nnunet-c/test_input)")
    args = ap.parse_args()

    cfg = load_corrector_config(args.config, caller_file=__file__)
    control = get_control(cfg, args.control)
    # A is the single-channel pure-nnUNet baseline (external Dataset835): no image
    # assembly, but we STILL emit a test_cases_map.json so A/B/C share one eval.
    # The A "prediction" is the stock 835 run on the degraded test CTs, i.e. the
    # native-grid nnUNet pred already on disk (= control B's prelabel source).
    is_A = control["prelabel_source"] == "none"
    if not is_A and int(control["n_channels"]) != 5:
        raise RuntimeError("this builder assembles the 5-channel controls (B/C); "
                           "use --control A only for the map/eval baseline.")

    res = cfg["_resolved"]
    steps = [int(s) for s in args.steps.split(",") if s.strip()]
    casefile = (Path(args.casefile) if (args.casefile and Path(args.casefile).is_absolute())
                else res["casefiles_dir"] / (args.casefile or "test_cases.txt"))
    if not casefile.is_file():
        print(f"[testset] casefile not found: {casefile}", file=sys.stderr)
        return 2
    source_ids = _read_source_ids(casefile)

    out_root = (Path(args.out) if args.out
                else res["nnunet_c_root"] / "test_input")
    ctl_dir = out_root / control["dataset_name"]
    images_out = ctl_dir / "imagesTs"
    if not is_A:
        images_out.mkdir(parents=True, exist_ok=True)
    else:
        ctl_dir.mkdir(parents=True, exist_ok=True)

    is_cnisp = control["prelabel_source"] == "cnisp"
    cnisp_test_dir = None
    if is_cnisp:
        cnisp_test_dir = (Path(args.cnisp_test_dir) if args.cnisp_test_dir
                          else res["repo_root"] / "nnunet-c" / "data" / "cnisp_pred_test")
        if not cnisp_test_dir.is_dir():
            print(f"[testset] CNISP test-mask dir missing: {cnisp_test_dir}\n"
                  f"  run the CNISP test inference first, e.g.:\n"
                  f"    OUT_DIR={cnisp_test_dir} ALIGNED_DIR={res['aligned_dir']} \\\n"
                  f"    CASEFILE={casefile.name} MAX_SAMPLES=0 GPUS=\"0 1\" \\\n"
                  f"    bash nnunet-c/run_corrector_cnisp.sh", file=sys.stderr)
            return 2

    src_label = ("nnUNet on degraded (A baseline; abs pred paths)" if is_A
                 else ("CNISP " + str(cnisp_test_dir) if is_cnisp
                       else "nnUNet (work_dir)"))
    print(f"[testset] control={args.control.upper()} -> {ctl_dir}")
    print(f"[testset] sources={len(source_ids)} steps={steps} casefile={casefile.name}")
    print(f"[testset] prelabel/pred source={src_label}")

    infos = resolve_source_infos(cfg, source_ids)

    assembled, skipped = [], 0
    case_map = {}
    for sid in source_ids:
        info = infos[sid]
        gt_path = Path(info.gt_label_path)
        if not gt_path.exists():
            print(f"  {sid}: GT/ref grid missing ({gt_path}); skip source")
            skipped += 1
            continue
        # struct_to_value stored in the map so the SHARED eval can remap GT to
        # {1,2,3,4} without re-running resolve_gt (keeps A/B/C eval identical).
        gt_stv = {k: int(v) for k, v in info.gt_struct_to_value.items()}
        gt_img = nib.load(str(gt_path))
        ref_grid = (gt_img.shape[:3], np.asarray(gt_img.affine))
        gstem = _gt_stem(gt_path)
        for step in steps:
            ct = _pre.degraded_ct_path(cfg, sid, step)
            if not ct.exists():
                skipped += 1
                continue
            cid = f"corr_{sid}_step{step:02d}"
            if is_A:
                # A = stock 835 on the degraded CT, native grid (= B's prelabel
                # source). No assembly; record the ABSOLUTE pred path for eval.
                pred = _pre._b_prelabel_path(cfg, sid, step)
                if not Path(pred).exists():
                    skipped += 1
                    continue
                case_map[cid] = {"source_id": sid, "step": step,
                                 "gt_label_path": str(gt_path),
                                 "gt_struct_to_value": gt_stv,
                                 "pred_file": str(pred)}
                assembled.append(cid)
                print(f"  {cid}: pred={Path(pred).name}")
                continue
            if is_cnisp:
                pre = cnisp_test_dir / f"{gstem}_step{step:02d}.nii.gz"
            else:
                pre = _pre._b_prelabel_path(cfg, sid, step)
            if not Path(pre).exists():
                skipped += 1
                continue
            summary = _ch.assemble_inference_case(
                case_id=cid, ct_path=ct, target_spacing=None, ref_grid=ref_grid,
                n_channels=5, structures=_STRUCTS,
                images_dir=images_out, experiment=cfg["experiment"],
                prelabel_path=Path(pre),
                prelabel_struct_to_value=dict(NNUNET_LABELS),
            )
            assembled.append(summary)
            # pred_file is RELATIVE -> eval joins it with --pred-dir (nnUNet out).
            case_map[cid] = {"source_id": sid, "step": step,
                             "gt_label_path": str(gt_path),
                             "gt_struct_to_value": gt_stv,
                             "pred_file": f"{cid}.nii.gz"}
            print(f"  {cid}: shape={summary['shape']}")

    with open(ctl_dir / "test_cases_map.json", "w") as f:
        json.dump({"control": args.control.upper(), "casefile": casefile.name,
                   "structures": _STRUCTS, "labels": dict(NNUNET_LABELS),
                   "n": len(assembled), "cases": case_map}, f, indent=2)
    print(f"[testset] wrote {len(assembled)} case(s); skipped {skipped} (missing files).")
    if not is_A:
        print(f"[testset] imagesTs -> {images_out}")
    print(f"[testset] map     -> {ctl_dir / 'test_cases_map.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
