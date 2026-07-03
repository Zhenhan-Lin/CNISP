#!/usr/bin/env python3
"""Driver: MASK_INDEX -> tidy per-structure metrics CSV (the shared interface).

Analogous to ``simulation/comparison/compare_native.py`` (which writes the paired
CSV): this reads a MASK_INDEX json and writes ``metrics_long.csv``, the interface
every ``*_summary`` driver consumes. Run this ONCE, then point the summaries at
the CSV (they can also build it themselves via --mask-index, but building once is
cheaper when several figures share it).

MASK_INDEX json = a list of dicts, one per (case, arm, step, mode) mask::

    {"case": "10058_20330227", "arm": "nnU-Net", "step": 5, "mode": "thick",
     "eff_res": 2.5, "pred_path": ".../sparse_step_05_native/...nii.gz",
     "gt_path": ".../atlas_labels/...nii.gz",
     "pred_scheme": "nnunet", "gt_scheme": "labelfusion", "offset_gt": 1000}

arms: nnU-Net / CNISP / nnU->nnU / Proposed / Oracle.

Usage:
    python simulation/evaluation/build_metrics.py \
        --mask-index index.json --out-csv comparison/viz/evaluation__thick/metrics_long.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simulation.evaluation.metrics import build_metrics_table, DEFAULT_TAU_MM


def run(args) -> int:
    with open(args.mask_index) as f:
        index = json.load(f)
    if not index:
        print(f"[build_metrics] empty MASK_INDEX in {args.mask_index}", file=sys.stderr)
        return 2
    df = build_metrics_table(index, tau=args.tau_mm, save_csv=args.out_csv)
    print(f"[build_metrics] {len(index)} mask(s) -> {len(df)} row(s) -> {args.out_csv}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mask-index", required=True, help="JSON list of per-mask entries.")
    ap.add_argument("--out-csv", required=True, help="destination metrics_long.csv.")
    ap.add_argument("--tau-mm", type=float, default=DEFAULT_TAU_MM,
                    help=f"Surface-Dice tolerance mm (default {DEFAULT_TAU_MM}).")
    return ap


if __name__ == "__main__":
    sys.exit(run(build_parser().parse_args()))
