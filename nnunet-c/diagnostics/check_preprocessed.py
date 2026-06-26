#!/usr/bin/env python3
"""Pothole-4 HARD GATE: validate one preprocessed 855/845 case before finetune.

After `nnUNetv2_preprocess` with the merged plan, this loads one preprocessed
case from ${nnUNet_preprocessed}/Dataset{ID}_.../<data_identifier>/ and asserts:

  1. ch0 (CT) is z-scored with the 835 stats: its min/max match the expected
     normalized clip bounds (percentile_00_5/99_5 - mean)/std from the 835 plan.
  2. ch1..ch4 are still {0,1} (pothole-2 a-ii kept the binaries intact).
  3. all channels share one spatial shape (single array) == the seg shape.
  4. label values are a subset of {0,1,2,3,4}.

Exits non-zero on any violation so run_full_pipeline.sh can block finetune.
Heavy deps (blosc2 / numpy) imported lazily.

Usage:
    python nnunet-c/diagnostics/check_preprocessed.py --control B
    python nnunet-c/diagnostics/check_preprocessed.py --control C --plan-name nnUNetPlansFinetune
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple


def _resolve_paths(config: str, control: str, plan_name: str, configuration: Optional[str]):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from lib.config import load_corrector_config, get_control  # lazy

    cfg = load_corrector_config(config, caller_file=__file__)
    ctrl = get_control(cfg, control)
    configuration = configuration or cfg["configuration"]
    preproc = os.environ.get("nnUNet_preprocessed")
    if not preproc:
        raise RuntimeError("$nnUNet_preprocessed unset (need it on the GPU box).")

    ds_dir = Path(preproc) / f"Dataset{int(ctrl['dataset_id']):03d}_{ctrl['dataset_name']}"
    data_dir = ds_dir / f"{plan_name}_{configuration}"
    ref_dir = (Path(preproc)
               / f"Dataset{int(cfg['reference_dataset_id']):03d}_{cfg['reference_dataset_name']}")
    ref_plan_json = ref_dir / f"{cfg['reference_plan']}.json"
    return cfg, ctrl, data_dir, ref_plan_json


def _load_case(data_dir: Path, case: Optional[str]):
    """Load (data[C,*spatial], seg[1,*spatial]) from a preprocessed case.

    Supports blosc2 (.b2nd), npz, and npy layouts across nnUNet versions.
    """
    import numpy as np  # lazy

    # discover a case stem
    def _stems(suffix: str):
        return sorted(p.name[: -len(suffix)] for p in data_dir.glob(f"*{suffix}")
                      if not p.name.endswith(f"_seg{suffix}"))

    for suffix, loader in ((".b2nd", "blosc2"), (".npz", "npz"), (".npy", "npy")):
        stems = _stems(suffix)
        if not stems:
            continue
        stem = case or stems[0]
        if loader == "blosc2":
            import blosc2  # lazy
            data = blosc2.open(str(data_dir / f"{stem}{suffix}"))[:]
            seg = blosc2.open(str(data_dir / f"{stem}_seg{suffix}"))[:]
        elif loader == "npz":
            npz = np.load(str(data_dir / f"{stem}{suffix}"))
            data, seg = npz["data"], npz["seg"]
        else:  # npy
            data = np.load(str(data_dir / f"{stem}{suffix}"))
            seg = np.load(str(data_dir / f"{stem}_seg{suffix}"))
        return stem, np.asarray(data), np.asarray(seg)

    raise FileNotFoundError(
        f"no preprocessed case (.b2nd/.npz/.npy) found in {data_dir}. "
        f"Did nnUNetv2_preprocess run with the merged plan?"
    )


def _expected_ct_bounds(ref_plan_json: Path) -> Tuple[float, float, Dict]:
    with open(ref_plan_json) as f:
        plan = json.load(f)
    fip = plan["foreground_intensity_properties_per_channel"]["0"]
    mean = float(fip["mean"])
    std = float(fip["std"])
    lo = float(fip.get("percentile_00_5", fip.get("percentile_0_5")))
    hi = float(fip.get("percentile_99_5"))
    return (lo - mean) / std, (hi - mean) / std, fip


def main() -> int:
    import numpy as np  # lazy

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config",
                    default=str(Path(__file__).resolve().parents[1] / "configs" / "corrector.yaml"))
    ap.add_argument("--control", required=True, choices=["B", "C", "b", "c"])
    ap.add_argument("--plan-name", default="nnUNetPlansFinetune")
    ap.add_argument("--configuration", default=None)
    ap.add_argument("--case", default=None, help="specific case stem (default: first)")
    ap.add_argument("--tol", type=float, default=0.15,
                    help="relative tolerance for CT normalized-bound check")
    args = ap.parse_args()

    cfg, ctrl, data_dir, ref_plan_json = _resolve_paths(
        args.config, args.control, args.plan_name, args.configuration
    )
    n_channels = int(ctrl["n_channels"])
    print(f"[check] control={args.control.upper()} data_dir={data_dir}")

    stem, data, seg = _load_case(data_dir, args.case)
    print(f"[check] case={stem} data.shape={data.shape} seg.shape={seg.shape}")

    failures = []

    # (3) shapes
    if data.shape[0] != n_channels:
        failures.append(f"channel count {data.shape[0]} != expected {n_channels}")
    if tuple(data.shape[1:]) != tuple(seg.shape[1:]):
        failures.append(f"data spatial {data.shape[1:]} != seg spatial {seg.shape[1:]}")

    # (1) ch0 CT normalization consistent with 835
    exp_lo, exp_hi, fip = _expected_ct_bounds(ref_plan_json)
    ch0 = data[0].astype(np.float64)
    ch0_min, ch0_max, ch0_mean, ch0_std = (float(ch0.min()), float(ch0.max()),
                                           float(ch0.mean()), float(ch0.std()))
    span = max(abs(exp_hi - exp_lo), 1e-6)
    lo_ok = abs(ch0_min - exp_lo) <= args.tol * span
    hi_ok = abs(ch0_max - exp_hi) <= args.tol * span
    print(f"[check] ch0 CT: min={ch0_min:.3f} (exp {exp_lo:.3f}) "
          f"max={ch0_max:.3f} (exp {exp_hi:.3f}) mean={ch0_mean:.3f} std={ch0_std:.3f}")
    print(f"        835 stats: mean={fip['mean']:.3f} std={fip['std']:.3f}")
    if not (lo_ok and hi_ok):
        failures.append(
            "ch0 normalized bounds do not match 835 stats (pothole 1): "
            f"min/max=({ch0_min:.3f},{ch0_max:.3f}) vs expected "
            f"({exp_lo:.3f},{exp_hi:.3f})"
        )

    # (2) ch1..ch4 binary
    if n_channels > 1:
        for c in range(1, n_channels):
            uniq = np.unique(data[c])
            is_binary = np.all(np.isin(uniq, [0, 1]))
            rng = (float(data[c].min()), float(data[c].max()))
            print(f"[check] ch{c}: unique<= {uniq[:6]}{' ...' if uniq.size > 6 else ''} "
                  f"range={rng} binary={bool(is_binary)}")
            if not is_binary:
                failures.append(
                    f"ch{c} is NOT binary (pothole 2 broken): range={rng}, "
                    f"{uniq.size} unique values"
                )

    # (4) label values subset of {0,1,2,3,4}
    seg_uniq = np.unique(seg)
    allowed = {0, 1, 2, 3, 4, -1}   # -1 = nnUNet ignore label, tolerated
    bad = sorted(set(int(v) for v in seg_uniq) - allowed)
    print(f"[check] label values: {sorted(int(v) for v in seg_uniq)}")
    if bad:
        failures.append(f"label has unexpected values {bad} (expected subset of 0..4)")

    print("───────────────────────────────────────────────────────────")
    if failures:
        print("[check] FAIL (pothole-4 gate):")
        for m in failures:
            print(f"  - {m}")
        return 1
    print("[check] PASS: CT normalization, binary channels, shapes, and labels OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
