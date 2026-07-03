#!/usr/bin/env python3
# NOTE: originally created under nnunet-c/diagnostics/ ; moved to nnunet-c/debugger/.
# Imports resolve via parents[1] == nnunet-c, so it runs from either location.
"""Probe: why are the TEST prelabel channels (ch1..ch4) empty?

Reproduces build_corrector_testset.py's prelabel resolution for a source and
prints every intermediate, so we can see EXACTLY where the mask goes to zero:

  1) info.gt_struct_to_value / gt_scheme  (the map _nn_prelabel remaps FROM)
  2) the CNISP prelabel SOURCE file (iso or native): unique values + counts
     -> is the source itself empty? what scheme/values does it actually carry?
  3) remap_to_nnunet(source, gt_struct_to_value): per-structure nonzero
     -> does the remap drop structures because the source scheme != gt scheme?
  4) resample the remapped mask onto the assembly ref grid (order 0)
     -> does world misalignment between the prelabel grid and the ref grid
        (degraded-CT 0.5 FOV in iso mode) zero it out?

Hypothesis under test: the CNISP decode emits CANONICAL {1,2,3,4} but the iso
branch interprets it with the source's labelfusion scheme {1,3,5,7}, so the
remap finds nothing (esp. Globe=5 / Fat=7) and the channels go empty.

``--step`` is OPTIONAL: omit it to sweep EVERY step present for --sid (steps are
auto-discovered from the CNISP run's native_space_step_XX manifests), so one call
walks all step_sizes for one source. Give --step to inspect a single step.

Usage:
    # all steps for one source (default):
    python nnunet-c/debugger/debug_test_prelabel.py \
        --control C --grid iso --sid atlas_orbit0001_ubMask_al2_fill
    # a single step:
    python nnunet-c/debugger/debug_test_prelabel.py \
        --control C --grid iso --sid atlas_orbit0001_ubMask_al2_fill --step 3
    # also try --grid gt to compare the native-mask path
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

from lib import prelabel as _pre  # noqa: E402
from lib.labels import (  # noqa: E402
    resolve_source_infos, remap_to_nnunet, remap_native_to_nnunet,
)
from lib.resample import build_reference_grid, resample_to_grid, voxel_spacing  # noqa: E402

STRUCTS = ["ON", "Recti", "Globe", "Fat"]
_DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "corrector.yaml"
_RUN_STEP_RE = re.compile(r"^native_space_step_(\d+)$")


def _uniq(arr) -> str:
    v, c = np.unique(arr, return_counts=True)
    pairs = [f"{int(vv)}:{int(cc)}" for vv, cc in zip(v, c)]
    return "{" + ", ".join(pairs[:12]) + ("" if len(pairs) <= 12 else " ...") + "}"


def _discover_steps(cfg, sid: str, grid_iso: bool) -> list:
    """Steps present for ``sid`` in the CNISP iso (grid_iso) or native run dir.

    Scans ``native_space_step_XX/manifest.json`` (same source of truth
    build_corrector_testset uses) and returns the sorted step list for sid.
    """
    root = _pre._cnisp_iso_root(cfg) if grid_iso else _pre._cnisp_run_dir(cfg)
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
        try:
            data = json.load(open(mf))
        except (OSError, json.JSONDecodeError):
            continue
        by_sid = data.get("by_source_id", data)
        if sid in by_sid:
            steps.append(int(m.group(1)))
    return sorted(set(steps))


def walk_one(cfg, control, info, grid_iso: bool, sid: str, step: int,
             iso_mm: float) -> None:
    """Print the full 2->3->4 prelabel-resolution walk for one (sid, step)."""
    print("\n" + "#" * 72)
    print(f"# sid={sid}  step={step}  grid={'iso' if grid_iso else 'gt'}")
    print("#" * 72)

    # 2) CNISP prelabel SOURCE file
    print("[2] CNISP prelabel source")
    try:
        src = (_pre._c_iso_prelabel_path(cfg, sid, step) if grid_iso
               else _pre._c_prelabel_path(cfg, sid, step))
    except (FileNotFoundError, KeyError) as e:
        print(f"    [!] cannot resolve prelabel: {e}")
        return
    print(f"    path = {src}")
    if not Path(src).exists():
        print("    [!] source file does not exist.")
        return
    img = nib.load(str(src))
    arr = np.asanyarray(img.dataobj)
    print(f"    shape={arr.shape} dtype={arr.dtype} "
          f"spacing={np.round(voxel_spacing(img.affine),3).tolist()}")
    print(f"    UNIQUE values:counts = {_uniq(arr)}")
    print(f"    nonzero = {int((arr != 0).sum())}")

    # 3) remap exactly like _nn_prelabel does (FROM gt scheme TO {1,2,3,4})
    print("[3] remap_to_nnunet(source, gt_struct_to_value) -- as the builder does")
    nn = remap_to_nnunet(arr, dict(info.gt_struct_to_value), STRUCTS)
    print(f"    remapped UNIQUE = {_uniq(nn)}")
    for i, name in enumerate(STRUCTS, start=1):
        print(f"      ch{i} {name:6s}: nonzero={int((nn == i).sum())}")

    # 3b) what a BY-NAME (canonical) remap would give, for contrast
    print("[3b] for contrast: remap assuming CNISP canonical {ON:1,Recti:2,Globe:3,Fat:4}")
    canon = {"ON": 1, "Recti": 2, "Globe": 3, "Fat": 4}
    nn2 = remap_to_nnunet(arr, canon, STRUCTS)
    for i, name in enumerate(STRUCTS, start=1):
        print(f"      ch{i} {name:6s}: nonzero={int((nn2 == i).sum())}")

    # 3c) what the FIXED builder now does: auto-detect scheme + remap BY NAME.
    print("[3c] FIXED builder path: remap_native_to_nnunet(scheme='auto')")
    nn3, det_scheme, det_off = remap_native_to_nnunet(arr, STRUCTS, scheme="auto")
    print(f"    detected scheme={det_scheme!r} offset={det_off}")
    for i, name in enumerate(STRUCTS, start=1):
        print(f"      ch{i} {name:6s}: nonzero={int((nn3 == i).sum())}")

    # 4) resample remapped mask onto the assembly ref grid
    print("[4] resample remapped mask onto the assembly ref grid (order 0)")
    ct = _pre.degraded_ct_path(cfg, sid, step)
    print(f"    degraded CT (ch0) = {ct}  exists={Path(ct).exists()}")
    if grid_iso:
        if not Path(ct).exists():
            print("    [!] degraded CT missing; cannot build iso ref grid.")
            return
        ref_shape, ref_aff = build_reference_grid(nib.load(str(ct)), [iso_mm] * 3)
    else:
        gt_img = nib.load(str(info.gt_label_path))
        ref_shape, ref_aff = gt_img.shape[:3], np.asarray(gt_img.affine)
    print(f"    ref grid shape={tuple(int(s) for s in ref_shape)} "
          f"spacing={np.round(voxel_spacing(ref_aff),3).tolist()}")
    print(f"    src  grid origin={np.round(img.affine[:3,3],1).tolist()}  "
          f"ref grid origin={np.round(np.asarray(ref_aff)[:3,3],1).tolist()}")
    nn_img = nib.Nifti1Image(nn.astype(np.uint8), img.affine)
    rs = resample_to_grid(nn_img, ref_shape, ref_aff, order=0)
    rs_arr = np.asanyarray(rs.dataobj)
    for i, name in enumerate(STRUCTS, start=1):
        print(f"      ch{i} {name:6s}: nonzero AFTER regrid = {int((rs_arr == i).sum())}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(_DEFAULT_CONFIG))
    ap.add_argument("--control", default="C")
    ap.add_argument("--grid", choices=["iso", "gt"], default="iso")
    ap.add_argument("--sid", required=True)
    ap.add_argument("--step", type=int, default=None,
                    help="single step to walk; OMIT to sweep ALL steps present "
                         "for --sid (auto-discovered).")
    ap.add_argument("--iso-mm", type=float, default=0.5)
    args = ap.parse_args()

    cfg = load_corrector_config(args.config, caller_file=__file__)
    control = get_control(cfg, args.control)
    grid_iso = (args.grid == "iso")
    sid = args.sid

    print("=" * 72)
    print(f"control={args.control} grid={args.grid} sid={sid} "
          f"step={args.step if args.step is not None else 'ALL'}")
    print("=" * 72)

    # 1) source info / GT scheme (what _nn_prelabel remaps FROM) -- step-independent
    info = resolve_source_infos(cfg, [sid])[sid]
    print("\n[1] source GT scheme")
    print(f"    gt_scheme           = {getattr(info, 'gt_scheme', '?')}")
    print(f"    gt_struct_to_value  = {dict(info.gt_struct_to_value)}")
    print(f"    gt_label_path       = {info.gt_label_path}")
    print("    >>> these are the (name->value) keys remap_to_nnunet will search "
          "for in the CNISP mask. If the CNISP mask uses DIFFERENT values, the "
          "matching voxels are 0 -> empty channels.")

    if args.step is not None:
        steps = [args.step]
    else:
        steps = _discover_steps(cfg, sid, grid_iso)
        if not steps:
            print(f"\n[!] no steps found for sid={sid} under "
                  f"{_pre._cnisp_iso_root(cfg) if grid_iso else _pre._cnisp_run_dir(cfg)}",
                  file=sys.stderr)
            return 1
        print(f"\n[sweep] steps present for {sid}: {steps}")

    for step in steps:
        walk_one(cfg, control, info, grid_iso, sid, step, args.iso_mm)

    print("\n[done] Per step, walk [2] -> [3] -> [4]: the first place a "
          "structure becomes 0 is the bug site.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
