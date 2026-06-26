#!/usr/bin/env python3
"""Smoke-test a few assembled RAW cases before nnUNet preprocessing.

Checks (on ${nnUNet_raw}/Dataset{ID}_.../imagesTr+labelsTr):
  * the expected number of channels (_0000.._000{N-1}) exist per case,
  * all channels + label share one shape and affine,
  * ch1..ch4 are binary {0,1},
  * label values are a subset of {0,1,2,3,4}.

This runs on the RAW dataset (pre-preprocess); the post-preprocess gate is
diagnostics/check_preprocessed.py. Heavy deps imported lazily.

Usage:
    python nnunet-c/diagnostics/smoke_dataset.py --control B --n 3
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    import numpy as np      # lazy
    import nibabel as nib   # lazy

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from lib.config import load_corrector_config, get_control  # lazy

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config",
                    default=str(Path(__file__).resolve().parents[1] / "configs" / "corrector.yaml"))
    ap.add_argument("--control", required=True, choices=["B", "C", "b", "c"])
    ap.add_argument("--n", type=int, default=3, help="cases to inspect")
    ap.add_argument("--raw-root", default=None, help="override $nnUNet_raw")
    args = ap.parse_args()

    cfg = load_corrector_config(args.config, caller_file=__file__)
    ctrl = get_control(cfg, args.control)
    n_channels = int(ctrl["n_channels"])

    raw = args.raw_root or os.environ.get("nnUNet_raw")
    if not raw:
        raise RuntimeError("set $nnUNet_raw or pass --raw-root")
    ds = Path(raw) / f"Dataset{int(ctrl['dataset_id']):03d}_{ctrl['dataset_name']}"
    images, labels = ds / "imagesTr", ds / "labelsTr"
    fe = ".nii.gz"

    cases = sorted(p.name[: -len(f"_0000{fe}")]
                   for p in images.glob(f"*_0000{fe}"))[: args.n]
    if not cases:
        raise FileNotFoundError(f"no cases in {images}")
    print(f"[smoke] control={args.control.upper()} dataset={ds.name} "
          f"n_channels={n_channels} inspecting {len(cases)} case(s)")

    failures = []
    for case in cases:
        ref = nib.load(str(images / f"{case}_0000{fe}"))
        ref_shape, ref_aff = ref.shape[:3], ref.affine
        print(f"  - {case}: ch0 shape={ref_shape}")
        for c in range(1, n_channels):
            f = images / f"{case}_{c:04d}{fe}"
            if not f.exists():
                failures.append(f"{case}: missing channel {c}")
                continue
            img = nib.load(str(f))
            if img.shape[:3] != ref_shape:
                failures.append(f"{case} ch{c}: shape {img.shape[:3]} != {ref_shape}")
            if not np.allclose(img.affine, ref_aff, atol=1e-4):
                failures.append(f"{case} ch{c}: affine mismatch")
            uniq = np.unique(np.asanyarray(img.dataobj))
            if not np.all(np.isin(uniq, [0, 1])):
                failures.append(f"{case} ch{c}: not binary (unique={uniq[:6]})")
        lab = nib.load(str(labels / f"{case}{fe}"))
        if lab.shape[:3] != ref_shape:
            failures.append(f"{case} label: shape {lab.shape[:3]} != {ref_shape}")
        luniq = sorted(int(v) for v in np.unique(np.asanyarray(lab.dataobj)))
        if set(luniq) - {0, 1, 2, 3, 4}:
            failures.append(f"{case} label: unexpected values {luniq}")
        print(f"    label values={luniq}")

    print("───────────────────────────────────────────────────────────")
    if failures:
        print("[smoke] FAIL:")
        for m in failures:
            print(f"  - {m}")
        return 1
    print("[smoke] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
