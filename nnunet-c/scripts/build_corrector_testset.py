#!/usr/bin/env python3
"""Convert CNISP's deployment-curve test output into nnUNet-C TEST input.

CNISP inference for the test set is CNISP's OWN existing run -- the thick-mode
``nnunet_pred`` deployment curve via ``03_infer.py``:

    runs/<experiment>/<run_tag>/native_space_step_XX/<gtstem>...nii.gz  (+ manifest.json)

(test cases + adaptive sweep are defined by the CNISP test yaml; nnUNet-C does
NOT run CNISP or invent a sweep). This script ONLY converts that output into the
5-channel nnUNet-C input, using the SAME ``engine/convert.py::convert_case`` the
train builder uses:

    ref grid = source's original/dense GT grid
    ch0      = degraded CT (work_dir/input) upsampled to it (order 3)
    ch1..ch4 = prelabel split into per-structure binaries (order 0)
               control C -> CNISP native mask (runs/.../native_space_step_XX/),
                            remapped BY NAME from its native scheme to nnUNet {1,2,3,4}
               control B -> work_dir Dataset835 native sparse pred (already {1,2,3,4})

A sidecar ``test_cases_map.json`` records corr_case -> {source/step/GT/scheme}
for the shared eval.

Usage:
    python nnunet-c/scripts/build_corrector_testset.py --control C
    python nnunet-c/scripts/build_corrector_testset.py --control B
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.config import load_corrector_config, get_control, add_repo_to_syspath  # noqa: E402

add_repo_to_syspath(__file__)

import numpy as np  # noqa: E402
import nibabel as nib  # noqa: E402

from engine.convert import convert_case, STRUCTS  # noqa: E402  (the SINGLE converter)
from lib import prelabel as _pre  # noqa: E402
from lib.labels import resolve_source_infos, remap_to_nnunet, NNUNET_LABELS  # noqa: E402
from lib.resample import build_reference_grid  # noqa: E402

_DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "corrector.yaml"
_NATIVE_DIR_RE = re.compile(r"^sparse_step_(\d+)_native$")
_RUN_STEP_RE = re.compile(r"^native_space_step_(\d+)$")


def _read_source_ids(casefile: Path) -> list:
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


def _discover_steps_cnisp_runs(cfg, sid: str) -> list:
    """Steps where CNISP's deployment run produced this source (via manifest)."""
    run_dir = _pre._cnisp_run_dir(cfg)
    if not run_dir.is_dir():
        return []
    steps = []
    for d in sorted(run_dir.glob("native_space_step_*")):
        m = _RUN_STEP_RE.match(d.name)
        if not m:
            continue
        mf = d / "manifest.json"
        if not mf.is_file():
            continue
        data = json.load(open(mf))
        by_sid = data.get("by_source_id", data)
        if sid in by_sid:
            steps.append(int(m.group(1)))
    return sorted(set(steps))


def _discover_steps_cnisp_iso(cfg, sid: str) -> list:
    """Steps where CNISP's ISO-0.5 deployment output produced this source.

    Mirrors _discover_steps_cnisp_runs but reads the iso prelabel root
    (nnunet-c/data/cnisp_pred_test_iso/native_space_step_XX/manifest.json).
    """
    root = _pre._cnisp_iso_root(cfg)
    if not root.is_dir():
        return []
    steps = []
    for d in sorted(root.glob("native_space_step_*")):
        m = _RUN_STEP_RE.match(d.name)
        if not m:
            continue
        mf = d / "manifest.json"
        if not mf.is_file():
            continue
        data = json.load(open(mf))
        by_sid = data.get("by_source_id", data)
        if sid in by_sid:
            steps.append(int(m.group(1)))
    return sorted(set(steps))


def _discover_steps_nnunet(cfg, sid: str) -> list:
    """Steps with a native-grid Dataset835 sparse pred for this source (A/B)."""
    root = cfg["_resolved"]["nnunet_pred_root"]
    exp_dir = root / cfg["experiment"]
    if not exp_dir.is_dir():
        return []
    steps = []
    for d in sorted(exp_dir.glob("sparse_step_*_native")):
        m = _NATIVE_DIR_RE.match(d.name)
        if m and (d / f"{sid}.nii.gz").exists():
            steps.append(int(m.group(1)))
    return sorted(set(steps))


def _nn_prelabel(pre_info: dict, cid: str, stage_dir: Path) -> Path:
    """Return a path to the prelabel in nnUNet {1,2,3,4} scheme.

    B's work_dir pred is already {1,2,3,4} -> use as-is. C's CNISP native mask is
    in the source's ORIGINAL scheme -> remap BY NAME (struct_to_value) to {1,2,3,4}
    and stage it (so the single converter always receives {1,2,3,4}).
    """
    src = Path(pre_info["path"])
    if pre_info.get("scheme") == "nnunet":
        return src
    img = nib.load(str(src))
    arr = np.asanyarray(img.dataobj)
    nn = remap_to_nnunet(arr, dict(pre_info["struct_to_value"]), STRUCTS)
    stage_dir.mkdir(parents=True, exist_ok=True)
    out = stage_dir / f"{cid}.nii.gz"
    nib.save(nib.Nifti1Image(nn.astype(np.uint8), img.affine), str(out))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(_DEFAULT_CONFIG))
    ap.add_argument("--control", required=True, choices=["A", "B", "C", "a", "b", "c"])
    ap.add_argument("--casefile", default=None,
                    help="casefile under casefiles_dir (default: CNISP test_cases.txt)")
    ap.add_argument("--steps", default="auto",
                    help="'auto' (default): discover whichever (source,step) CNISP "
                         "produced (C) / has nnUNet preds (A/B). Or an explicit list.")
    ap.add_argument("--prelabel-grid", choices=["iso", "gt"], default="iso",
                    help="iso (default): assemble the 5ch case on the iso-0.5 head "
                         "grid built from the DEGRADED image (ch0) FOV; control C's "
                         "ch1..4 come from CNISP's iso-0.5 prelabels (no native "
                         "round-trip, no GT grid). gt: legacy -- assemble on the GT "
                         "native grid, ch1..4 from CNISP native_space masks.")
    ap.add_argument("--iso-mm", type=float, default=0.5,
                    help="iso spacing (mm) for --prelabel-grid iso (default 0.5).")
    ap.add_argument("--out", default=None, help="output root (default: nnunet-c/test_input)")
    args = ap.parse_args()

    cfg = load_corrector_config(args.config, caller_file=__file__)
    control = get_control(cfg, args.control)
    is_A = control["prelabel_source"] == "none"
    is_cnisp = control["prelabel_source"] == "cnisp"
    grid_iso = (args.prelabel_grid == "iso")
    if not is_A and int(control["n_channels"]) != 5:
        raise RuntimeError("this builder assembles the 5-channel controls (B/C); "
                           "use --control A only for the map/eval baseline.")

    res = cfg["_resolved"]
    auto_steps = args.steps.strip().lower() == "auto"
    explicit_steps = (None if auto_steps
                      else [int(s) for s in args.steps.split(",") if s.strip()])
    casefile = (Path(args.casefile) if (args.casefile and Path(args.casefile).is_absolute())
                else res["casefiles_dir"] / (args.casefile or "test_cases.txt"))
    if not casefile.is_file():
        print(f"[testset] casefile not found: {casefile}", file=sys.stderr)
        return 2
    source_ids = _read_source_ids(casefile)

    out_root = (Path(args.out) if args.out else res["nnunet_c_root"] / "test_input")
    ctl_dir = out_root / control["dataset_name"]
    images_out = ctl_dir / "imagesTs"
    stage_dir = ctl_dir / "_prelabel_nn"
    (images_out if not is_A else ctl_dir).mkdir(parents=True, exist_ok=True)

    if is_cnisp:
        _need = _pre._cnisp_iso_root(cfg) if grid_iso else _pre._cnisp_run_dir(cfg)
        if not _need.is_dir():
            _hint = ("EMIT_ISO=1 bash nnunet-c/run_corrector_predict.sh C 0  "
                     "(or 03_infer.py --emit-iso-prelabel-dir "
                     "nnunet-c/data/cnisp_pred_test_iso)") if grid_iso else (
                     "bash nnunet-c/run_corrector_predict.sh C 0  "
                     "(or 03_infer.py --test-label-source nnunet_pred --run-tag "
                     f"{cfg['run_tag']} --experiment {cfg['experiment']})")
            print(f"[testset] CNISP {'iso' if grid_iso else 'native'} output "
                  f"missing: {_need}\n  Run CNISP test first, e.g.:\n    {_hint}",
                  file=sys.stderr)
            return 2

    print(f"[testset] control={args.control.upper()} -> {ctl_dir}")
    print(f"[testset] sources={len(source_ids)} "
          f"steps={'auto-discover' if auto_steps else explicit_steps} "
          f"casefile={casefile.name}")
    if is_cnisp:
        print(f"[testset] CNISP runs: {_pre._cnisp_run_dir(cfg)}")

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
        gt_stv = {k: int(v) for k, v in info.gt_struct_to_value.items()}
        # GT is used ONLY for the eval map (resample pred->GT grid at eval) and,
        # in legacy gt mode, as the assembly ref grid. In iso mode the case is
        # assembled on the degraded-image 0.5 head grid (built per step below),
        # so the source's original resolution never touches the corrector input.
        ref_grid_gt = None
        if not grid_iso:
            gt_img = nib.load(str(gt_path))
            ref_grid_gt = (gt_img.shape[:3], np.asarray(gt_img.affine))

        if explicit_steps is not None:
            src_steps = explicit_steps
        elif is_cnisp:
            src_steps = (_discover_steps_cnisp_iso(cfg, sid) if grid_iso
                         else _discover_steps_cnisp_runs(cfg, sid))
        else:                                   # A / B -> nnUNet native preds
            src_steps = _discover_steps_nnunet(cfg, sid)

        for step in src_steps:
            ct = _pre.degraded_ct_path(cfg, sid, step)
            if not ct.exists():
                skipped += 1
                continue
            cid = f"corr_{sid}_step{step:02d}"

            if is_A:
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

            # Reference grid for the 5ch case:
            #   iso -> iso-mm head grid from the DEGRADED image (ch0) FOV; B and
            #          C share it for the same (sid, step) so they stay
            #          structurally identical (fair B-vs-C). No GT grid.
            #   gt  -> legacy GT native grid.
            if grid_iso:
                ref_grid = build_reference_grid(nib.load(str(ct)),
                                                [args.iso_mm] * 3)
            else:
                ref_grid = ref_grid_gt

            # Resolve ch1..4 prelabel.
            #   C + iso -> CNISP iso-0.5 head mask (original scheme; _nn_prelabel
            #              remaps to {1,2,3,4} by name -- no native round-trip).
            #   C + gt / B -> resolve_prelabel (native_space C / nnUNet pred B).
            if is_cnisp and grid_iso:
                try:
                    _iso_path = _pre._c_iso_prelabel_path(cfg, sid, step)
                except (FileNotFoundError, KeyError):
                    skipped += 1
                    continue
                pre_info = {"path": _iso_path,
                            "struct_to_value": dict(info.gt_struct_to_value),
                            "scheme": info.gt_scheme}
            else:
                try:
                    pre_info = _pre.resolve_prelabel(cfg, control, sid, step, info)
                except (FileNotFoundError, KeyError):
                    skipped += 1
                    continue
            if pre_info is None or not Path(pre_info["path"]).exists():
                skipped += 1
                continue
            pre_nn = _nn_prelabel(pre_info, cid, stage_dir)   # -> {1,2,3,4}
            summary = convert_case(
                case_id=cid, ct_path=ct, prelabel_path=pre_nn, ref_grid=ref_grid,
                experiment=cfg["experiment"], images_dir=images_out,
            )
            assembled.append(summary)
            case_map[cid] = {"source_id": sid, "step": step,
                             "gt_label_path": str(gt_path),
                             "gt_struct_to_value": gt_stv,
                             "pred_file": f"{cid}.nii.gz"}
            print(f"  {cid}: shape={summary['shape']}")

    with open(ctl_dir / "test_cases_map.json", "w") as f:
        json.dump({"control": args.control.upper(), "casefile": casefile.name,
                   "structures": STRUCTS, "labels": dict(NNUNET_LABELS),
                   "n": len(assembled), "cases": case_map}, f, indent=2)
    print(f"[testset] wrote {len(assembled)} case(s); skipped {skipped} (missing files).")
    if not is_A:
        print(f"[testset] imagesTs -> {images_out}")
    print(f"[testset] map     -> {ctl_dir / 'test_cases_map.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
