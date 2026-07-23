#!/usr/bin/env python3
"""Mask the Dataset835 nnUNet predictions to the acquired FOV (truncation-aware).

nnUNet stage-1 runs on the FOV-truncated CT (out-of-FOV = air). Near the truncation
boundary it under-segments AND spills foreground into the blanked region -- but there
is no image evidence there, so no prediction should exist. This zeroes every
``nnunet_pred`` voxel OUTSIDE that scan's acquired FOV, so:

  * the prediction (and its visualization) has NO label in the truncated region;
  * CNISP's test-time fit sees the clean "predicted-region" observation and only
    optimizes where nnUNet predicted (bg or fg) AND the image has data (= inside FOV).

The truncation preserves the CT grid, so the sidecar ``visible_box`` (per-source-axis
half-open ``[lo, hi)`` voxel windows, keyed by ``[case_id][step]`` in
``fov_truncation_manifest.json``) indexes the prediction voxels DIRECTLY -- a pure
voxel-window crop, no affine/resampling. ``source_shape`` guards the grid match.

Usage:
    python nnunet-c/scripts/mask_nnunet_pred_fov.py --config nnunet-c/configs/corrector_fov.yaml
    python nnunet-c/scripts/mask_nnunet_pred_fov.py --self-test
    # explicit dirs (skip the config):
    python nnunet-c/scripts/mask_nnunet_pred_fov.py \
        --pred-dir <data_root>/nnunet_pred --trunc-manifest <data_root>/fov_truncation_manifest.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
_PRED_RE = re.compile(r"^(?P<case>.+)_step(?P<step>\d+)\.nii\.gz$")


def fov_keep_mask(shape, visible_box) -> np.ndarray:
    """Bool array over ``shape`` (True inside the half-open ``visible_box`` per axis)."""
    keep = np.zeros(tuple(int(s) for s in shape), dtype=bool)
    sl = tuple(slice(int(lo), int(hi)) for lo, hi in visible_box)
    keep[sl] = True
    return keep


def mask_pred_to_fov(pred: np.ndarray, visible_box, source_shape=None) -> np.ndarray:
    """Return ``pred`` with everything outside ``visible_box`` set to 0 (background)."""
    if source_shape is not None and tuple(int(s) for s in pred.shape) != \
            tuple(int(s) for s in source_shape):
        raise ValueError(
            f"pred shape {pred.shape} != sidecar source_shape {tuple(source_shape)}; "
            f"grid mismatch -- refusing to mask (is this the truncated-CT grid?).")
    keep = fov_keep_mask(pred.shape, visible_box)
    out = pred.copy()
    out[~keep] = 0
    return out


def run(args) -> int:
    if args.trunc_manifest and args.pred_dir:
        pred_dir = Path(args.pred_dir)
        trunc = json.load(open(args.trunc_manifest))
    else:
        sys.path.insert(0, str(REPO / "nnunet-c"))
        from lib.config import load_corrector_config
        cfg = load_corrector_config(str(REPO / args.config) if not Path(args.config).is_absolute()
                                    else args.config, caller_file=str(Path(__file__)))
        cd = cfg.get("corrector_data", {}) or {}
        dr = Path(cd.get("data_root", "nnunet-c/data"))
        dr = dr if dr.is_absolute() else (cfg["_resolved"]["repo_root"] / dr)
        pred_dir = dr / cd.get("nnunet_pred_dirname", "nnunet_pred")
        tm = Path(args.trunc_manifest) if args.trunc_manifest else (dr / "fov_truncation_manifest.json")
        if not tm.is_file():
            print(f"[mask-fov] sidecar not found: {tm}", file=sys.stderr)
            return 2
        trunc = json.load(open(tm))

    import nibabel as nib
    if not pred_dir.is_dir():
        print(f"[mask-fov] pred dir not found: {pred_dir}", file=sys.stderr)
        return 2
    out_dir = Path(args.out_dir) if args.out_dir else pred_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    n_ok = n_skip = n_changed = 0
    for p in sorted(pred_dir.glob("*.nii.gz")):
        m = _PRED_RE.match(p.name)
        if not m:
            continue
        case, step = m.group("case"), str(int(m.group("step")))
        info = (trunc.get(case, {}) or {}).get(step)
        if not info or "visible_box" not in info:
            print(f"  {p.name}: no sidecar visible_box (case={case} step={step}); SKIP")
            n_skip += 1
            continue
        img = nib.load(str(p))
        pred = np.asanyarray(img.dataobj)
        try:
            masked = mask_pred_to_fov(pred, info["visible_box"], info.get("source_shape"))
        except ValueError as e:
            print(f"  {p.name}: {e}; SKIP")
            n_skip += 1
            continue
        removed = int((pred != 0).sum() - (masked != 0).sum())
        if removed != 0 or out_dir != pred_dir:
            nib.save(nib.Nifti1Image(masked.astype(pred.dtype), img.affine, img.header),
                     str(out_dir / p.name))
            n_changed += 1 if removed != 0 else 0
        print(f"  {p.name}: removed {removed} out-of-FOV label voxel(s)")
        n_ok += 1
    print(f"[mask-fov] masked {n_ok} pred(s) ({n_changed} changed, {n_skip} skipped) -> {out_dir}")
    return 0 if n_ok else 1


def _self_test() -> int:
    shape = (12, 14, 16)
    visible_box = [[2, 8], [0, 14], [3, 12]]           # axis 0 & 2 clipped, axis 1 full
    pred = np.zeros(shape, np.int16)
    pred[5, 7, 8] = 3                                   # inside the box -> kept
    pred[0, 7, 8] = 2                                   # outside axis-0 window -> removed
    pred[5, 7, 1] = 4                                   # outside axis-2 window -> removed
    out = mask_pred_to_fov(pred, visible_box, source_shape=shape)
    assert out[5, 7, 8] == 3, "inside-FOV label must survive"
    assert out[0, 7, 8] == 0 and out[5, 7, 1] == 0, "out-of-FOV labels must be zeroed"
    assert (out != 0).sum() == 1
    # grid-mismatch guard
    try:
        mask_pred_to_fov(pred, visible_box, source_shape=(9, 9, 9))
        raise AssertionError("should have raised on shape mismatch")
    except ValueError:
        pass
    # keep-mask sanity
    keep = fov_keep_mask(shape, visible_box)
    assert keep.sum() == 6 * 14 * 9 and keep[2, 0, 3] and not keep[1, 0, 3]
    print("mask_pred_to_fov: inside kept, outside zeroed, shape-guard OK")
    print("\nALL MASK-NNUNET-PRED-FOV SELF-TESTS PASSED")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="nnunet-c/configs/corrector_fov.yaml",
                    help="corrector config (resolves data_root -> nnunet_pred + sidecar).")
    ap.add_argument("--pred-dir", default=None, help="explicit nnunet_pred dir (skips --config).")
    ap.add_argument("--trunc-manifest", default=None, help="explicit fov_truncation_manifest.json.")
    ap.add_argument("--out-dir", default=None,
                    help="write masked preds here (default: in place, overwriting pred-dir).")
    ap.add_argument("--self-test", action="store_true")
    return ap


if __name__ == "__main__":
    a = build_parser().parse_args()
    sys.exit(_self_test() if a.self_test else run(a))
