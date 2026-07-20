#!/usr/bin/env python3
"""Driver: 5-arm Dice-vs-effective-resolution figure from metrics_long.csv.

Single-source-of-truth replacement for the old comparison-track ``combined__thick``
Dice curves. Reads the SAME ``metrics_long.csv`` every other evaluation figure
uses -- native-mask Dice for all 5 arms (nnUNet / Cascade UNet / CNISP / Proposed
/ Oracle, incl. Oracle) on the shared eff_res buckets -- so it needs NO paired
CSVs, no re-inference, and no recompute. Given an existing metrics_long.csv it is
a pure re-plot (resume).

Note (1a decision): CNISP (C) and Oracle (E) Dice here are the NATIVE reconstructed
-mask-vs-GT Dice from metrics_long, NOT the old canonical per-eye Dice from
sweep_results.pkl. Arms A/B/D are identical to the old convention by construction.

Usage:
    python simulation/evaluation/dice_quality_summary.py \
        --out comparison/viz/evaluation__thick --mode thick \
        --metrics-csv comparison/viz/evaluation__thick/metrics_long.csv
    # or build on the fly from a MASK_INDEX:
    python simulation/evaluation/dice_quality_summary.py \
        --out comparison/viz/evaluation__thick --mask-index .../mask_index.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simulation.evaluation import aggregate, plots
from simulation.evaluation.metrics import load_metrics_df, DEFAULT_TAU_MM


def run(args) -> int:
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    index = None
    if args.mask_index:
        with open(args.mask_index) as f:
            index = json.load(f)
    df = load_metrics_df(args.metrics_csv, index, args.tau_mm)
    if df is None:
        raise SystemExit(
            "[dice_quality_summary] no metrics -- pass --metrics-csv (an existing "
            "metrics_long.csv) or --mask-index. This driver has no synthetic mode.")
    if args.common_samples:
        df = aggregate.restrict_to_common(df)

    # Optional config-driven legend override (arm -> display string). Keyed by the
    # canonical METHODS names, so a config can rename the legend without code edits.
    legend_map = None
    if args.config:
        from nnunet.helpers.config import load_yaml
        cfg = load_yaml(Path(args.config))
        legend_map = cfg.get("eval_arm_labels") or None

    edges = None
    if args.bucket_edges:
        edges = [float(x) for x in args.bucket_edges.split(",") if x.strip()]

    bucket_order, by_arm_bucket, eff_by_bucket = aggregate.dice_vs_eff_res(
        df, args.mode, edges)
    if not bucket_order:
        raise SystemExit(f"[dice_quality_summary] no eff_res buckets for mode="
                         f"{args.mode!r} (is 'eff_res' populated in the metrics?).")
    p = out / "dice_vs_eff_res.png"
    plots.dice_vs_eff_res_figure(bucket_order, by_arm_bucket, eff_by_bucket, p,
                                 delta_arm=args.delta_arm, baseline=args.baseline,
                                 legend_map=legend_map, ymin=args.ymin)
    print(f"[dice_quality_summary] wrote {p}  "
          f"(arms={sorted({a for a, _ in by_arm_bucket})}, buckets={len(bucket_order)})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output dir for the figure.")
    ap.add_argument("--metrics-csv", default=None, help="prebuilt metrics_long.csv (resume).")
    ap.add_argument("--mask-index", default=None, help="MASK_INDEX json (built on the fly).")
    ap.add_argument("--mode", default=aggregate.DEFAULT_MODE,
                    help=f"sweep mode to aggregate (default {aggregate.DEFAULT_MODE}).")
    ap.add_argument("--delta-arm", default="Proposed",
                    help="arm for the head-to-head delta panel (default Proposed).")
    ap.add_argument("--baseline", default="nnUNet",
                    help="baseline arm the delta is measured against (default nnUNet).")
    ap.add_argument("--ymin", type=float, default=0.5,
                    help="Dice y-axis lower bound for the overall + per-class panels "
                         "(default 0.5, so the curves aren't squashed together; set "
                         "lower if any arm dips below it at coarse resolution).")
    ap.add_argument("--config", default=None,
                    help="optional YAML with 'eval_arm_labels' (arm -> legend str).")
    ap.add_argument("--bucket-edges", default=None,
                    help="comma eff_res bucket edges (mm); default = "
                         "buckets.DEFAULT_BUCKET_EDGES_MM.")
    ap.add_argument("--common-samples", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="restrict to (case, step) common to every arm (default on).")
    ap.add_argument("--tau-mm", type=float, default=DEFAULT_TAU_MM)
    return ap


if __name__ == "__main__":
    sys.exit(run(build_parser().parse_args()))
