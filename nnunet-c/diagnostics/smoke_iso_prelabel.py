#!/usr/bin/env python3
"""Smoke-test the newly activated iso-0.5 prelabel link (map_iso_results_to_native).

Checks, for ONE source/step, that the emitted iso-0.5 full-head mask
(``*_cnisp_iso_step{XX}.nii.gz`` from 03_infer --emit-iso-prelabel-dir):

  1. has isotropic 0.5 mm spacing,
  2. is geometrically co-located with the degraded image (same direction +
     origin + FOV extent) -- i.e. it lands on the head grid defined by the
     degraded image FOV + 0.5, NOT some shifted/native grid,
  3. places foreground in the SAME physical place as the trusted NATIVE CNISP
     mask: we resample the native mask onto the iso grid (order 0) and compute
     per-label Dice. High Dice => the iso placement (place_sub_patch_in_disk ->
     reverse_flip -> reverse_reorient -> place_patch_in_volume, all in iso
     voxels) is consistent with the native path. A low Dice would mean a
     placement/offset bug in the activated iso link -- STOP.

This does NOT need a GPU or the CNISP model: it only reads the two produced
NIfTIs (+ optionally the degraded CT). Run it after a
``EMIT_ISO=1 ... run_corrector_predict.sh C`` (or a 03_infer --emit-iso-prelabel-dir).

Usage:
    python nnunet-c/diagnostics/smoke_iso_prelabel.py \
        --iso-mask    nnunet-c/data/cnisp_pred_test_iso/<stem>_cnisp_iso_step03.nii.gz \
        --native-mask <output_basedir>/<model>/runs/thick/corrector_gt/native_space_step_03/<stem>_cnisp_step03.nii.gz \
        --degraded-ct <work_dir>/input/thick/sparse_step_03/<sid>_0000.nii.gz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to


def _spacing(affine: np.ndarray) -> np.ndarray:
    return np.sqrt((np.asarray(affine)[:3, :3] ** 2).sum(axis=0))


def _direction(affine: np.ndarray) -> np.ndarray:
    a = np.asarray(affine)[:3, :3]
    return a / _spacing(affine)


def _fov_mm(img) -> np.ndarray:
    return np.asarray(img.shape[:3], dtype=float) * _spacing(img.affine)


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    pa, pb = float(a.sum()), float(b.sum())
    if pa + pb == 0.0:
        return 1.0
    return 2.0 * float(np.logical_and(a, b).sum()) / (pa + pb)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iso-mask", required=True)
    ap.add_argument("--native-mask", required=True)
    ap.add_argument("--degraded-ct", default=None)
    ap.add_argument("--iso-mm", type=float, default=0.5)
    ap.add_argument("--dice-warn", type=float, default=0.90,
                    help="warn if native-vs-iso Dice below this (default 0.90)")
    args = ap.parse_args()

    iso = nib.load(args.iso_mask)
    nat = nib.load(args.native_mask)
    iso_arr = np.asanyarray(iso.dataobj)
    nat_arr = np.asanyarray(nat.dataobj)

    ok = True
    print("=" * 66)
    print(f"[smoke] iso    : {Path(args.iso_mask).name}  shape={iso.shape}")
    print(f"[smoke] native : {Path(args.native_mask).name}  shape={nat.shape}")
    print("=" * 66)

    # 1) iso spacing == iso_mm on all axes.
    isp = _spacing(iso.affine)
    print(f"[1] iso spacing = {np.round(isp,4).tolist()} (expect {args.iso_mm})")
    if not np.allclose(isp, args.iso_mm, atol=1e-3):
        print("    FAIL: iso mask spacing is not isotropic at iso_mm.")
        ok = False

    # 2) geometry vs degraded image (same head grid: direction/origin/FOV).
    if args.degraded_ct:
        deg = nib.load(args.degraded_ct)
        d_dir, i_dir = _direction(deg.affine), _direction(iso.affine)
        d_org = np.asarray(deg.affine)[:3, 3]
        i_org = np.asarray(iso.affine)[:3, 3]
        d_fov, i_fov = _fov_mm(deg), _fov_mm(iso)
        print(f"[2] degraded FOV(mm) = {np.round(d_fov,1).tolist()}  "
              f"iso FOV(mm) = {np.round(i_fov,1).tolist()}")
        print(f"    origin  deg={np.round(d_org,2).tolist()}  "
              f"iso={np.round(i_org,2).tolist()}")
        if not np.allclose(d_dir, i_dir, atol=1e-2):
            print("    WARN: direction cosines differ from degraded image.")
        if not np.allclose(d_org, i_org, atol=1.0):
            print("    WARN: origin differs from degraded image by >1 mm.")
        if not np.allclose(d_fov, i_fov, rtol=0.05, atol=2.0):
            print("    WARN: FOV extent differs from degraded image by >5%.")
    else:
        print("[2] (skipped; pass --degraded-ct to check head-grid co-location)")

    # 3) foreground present + placement consistent with the native mask.
    labels = sorted(int(v) for v in np.unique(iso_arr) if int(v) != 0)
    print(f"[3] iso foreground labels = {labels}  "
          f"fg voxels = {int((iso_arr>0).sum())}")
    if not labels:
        print("    FAIL: iso mask has NO foreground.")
        ok = False

    # Resample native -> iso grid (order 0) and Dice per label.
    nat_on_iso = np.asanyarray(
        resample_from_to(nat, (iso.shape[:3], iso.affine), order=0).dataobj
    ).astype(np.int16)
    all_labels = sorted(set(labels) | {int(v) for v in np.unique(nat_on_iso)
                                       if int(v) != 0})
    print("    native(->iso) vs iso Dice per label "
          "(placement consistency check):")
    dices = []
    for lab in all_labels:
        d = _dice(iso_arr == lab, nat_on_iso == lab)
        dices.append(d)
        flag = "" if d >= args.dice_warn else "  <-- LOW"
        print(f"      label {lab}: Dice = {d:.4f}{flag}")
    mean_d = float(np.mean(dices)) if dices else 0.0
    print(f"    MEAN Dice = {mean_d:.4f}")
    if mean_d < args.dice_warn:
        print(f"    WARN: mean Dice < {args.dice_warn}. Either real native->0.5 "
              f"resolution difference (expected, small) OR an iso PLACEMENT bug "
              f"(large drop / one label near 0). Inspect overlay before trusting.")

    print("-" * 66)
    print(f"[smoke] {'PASS (hard checks)' if ok else 'FAIL (hard checks)'}; "
          f"native-vs-iso mean Dice = {mean_d:.4f}")
    print("    NOTE: native vs iso are not expected to be identical (different "
          "sampling); this checks they agree in PLACEMENT, not bit-equality.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
