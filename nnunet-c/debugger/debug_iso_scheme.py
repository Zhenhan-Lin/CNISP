#!/usr/bin/env python3
# NOTE: originally created under nnunet-c/diagnostics/ ; moved to nnunet-c/debugger/.
# Imports resolve via parents[1] == nnunet-c, so it runs from either location.
"""Decide the TRUE label scheme of the CNISP iso prelabel by cross-tabulating it
against the source GT.

The unique value set {0,1,2,3,4} is ambiguous: it is the SAME whether the mask
is in CNISP canonical order (1=ON,2=Globe,3=Fat,4=Recti) or already remapped to
nnUNet by name (1=ON,2=Recti,3=Globe,4=Fat) -- both are permutations of {1..4}.
To pin which structure each iso VALUE actually is, we overlap each iso value
region with each GT structure (whose value->name map we KNOW from resolve_gt)
and report the best-matching GT structure per iso value.

Output: for each iso value v in {1,2,3,4}, the GT structure it overlaps most
(by intersection voxel count), so we can read off the real iso scheme:
    v=1 -> ON, v=2 -> ?, v=3 -> ?, v=4 -> ?

Usage:
    python nnunet-c/debugger/debug_iso_scheme.py \
        --control C --grid iso --sid atlas_orbit0001_ubMask_al2_fill --step 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.config import load_corrector_config, get_control, add_repo_to_syspath  # noqa: E402

add_repo_to_syspath(__file__)

import numpy as np  # noqa: E402
import nibabel as nib  # noqa: E402

from lib import prelabel as _pre  # noqa: E402
from lib.labels import resolve_source_infos  # noqa: E402
from lib.resample import resample_to_grid, voxel_spacing  # noqa: E402

_DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "corrector.yaml"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(_DEFAULT_CONFIG))
    ap.add_argument("--control", default="C")
    ap.add_argument("--grid", choices=["iso", "gt"], default="iso")
    ap.add_argument("--sid", required=True)
    ap.add_argument("--step", type=int, required=True)
    args = ap.parse_args()

    cfg = load_corrector_config(args.config, caller_file=__file__)
    _ = get_control(cfg, args.control)
    sid, step = args.sid, args.step
    info = resolve_source_infos(cfg, [sid])[sid]

    print("=" * 72)
    print(f"iso-scheme cross-tab: sid={sid} step={step} grid={args.grid}")
    print("=" * 72)
    print(f"gt_scheme          = {getattr(info, 'gt_scheme', '?')}")
    print(f"gt_struct_to_value = {dict(info.gt_struct_to_value)}")

    # iso (or native) CNISP mask
    src = (_pre._c_iso_prelabel_path(cfg, sid, step) if args.grid == "iso"
           else _pre._c_prelabel_path(cfg, sid, step))
    iso_img = nib.load(str(src))
    iso = np.asanyarray(iso_img.dataobj).astype(np.int32)
    print(f"\niso mask  : {src}")
    print(f"  shape={iso.shape} spacing={np.round(voxel_spacing(iso_img.affine),3).tolist()} "
          f"unique={np.unique(iso).tolist()}")

    # GT label, resampled onto the iso grid (order 0) so they share voxels.
    gt_img = nib.load(str(info.gt_label_path))
    print(f"GT label  : {info.gt_label_path}")
    print(f"  shape={gt_img.shape} spacing={np.round(voxel_spacing(gt_img.affine),3).tolist()} "
          f"unique={np.unique(np.asanyarray(gt_img.dataobj)).tolist()[:12]}")
    gt_rs = resample_to_grid(gt_img, iso.shape, iso_img.affine, order=0)
    gt = np.asanyarray(gt_rs.dataobj).astype(np.int32)

    # value -> structure name for the GT (invert gt_struct_to_value)
    val2name = {int(v): k for k, v in info.gt_struct_to_value.items()}
    print(f"\nGT value->name = {val2name}")
    gt_fg_total = int((gt != 0).sum())
    print(f"GT foreground voxels on iso grid = {gt_fg_total}")
    if gt_fg_total == 0:
        print("[!] GT has no foreground on the iso grid -- world misalignment? "
              "cross-tab impossible.")
        return 1

    print("\nFor each iso VALUE, overlap with each GT structure (intersection "
          "voxel count); the best match names that iso value:")
    canonical = {1: "ON", 2: "Globe", 3: "Fat", 4: "Recti"}     # native_mapping
    nnunet = {1: "ON", 2: "Recti", 3: "Globe", 4: "Fat"}        # by-name target
    for v in [1, 2, 3, 4]:
        iso_v = (iso == v)
        nv = int(iso_v.sum())
        if nv == 0:
            print(f"  iso value {v}: (absent)")
            continue
        overlaps = {}
        for gv, gname in val2name.items():
            inter = int(np.logical_and(iso_v, gt == gv).sum())
            overlaps[gname] = inter
        best = max(overlaps, key=overlaps.get)
        frac = overlaps[best] / nv if nv else 0.0
        ov_str = " ".join(f"{n}={c}" for n, c in
                          sorted(overlaps.items(), key=lambda kv: -kv[1]))
        print(f"  iso value {v} (n={nv}): best GT match = {best} "
              f"({frac*100:.1f}% of the value's voxels)  [{ov_str}]")
        print(f"      canonical says {canonical[v]:6s} | nnunet-by-name says {nnunet[v]}")

    print("\n>>> Read each 'best GT match' against the canonical/nnunet guesses. "
          "If best matches the nnunet column, the iso mask is ALREADY nnUNet "
          "{1,2,3,4} by name -> the builder should remap with NNUNET_LABELS. "
          "If it matches canonical, the builder must apply the canonical->nnUNet "
          "by-name remap {1:1,2:3,3:4,4:2}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
