#!/usr/bin/env python3
"""Driver: cross-resolution volume-stability figure (CoV + optic-nerve range).

Reads the metrics CSV (or builds it from a MASK_INDEX; or renders the synthetic
illustrative layout when neither is given), aggregates volume CoV across
step-sizes, and writes ``volume_stability_by_resolution.png``.

Sibling of ``simulation/comparison/method_summary.py``: one figure per driver,
reusing the shared metrics/aggregate/plots layers.

Usage:
    python simulation/evaluation/volume_stability_summary.py \
        --out comparison/viz/evaluation__thick --mode thick \
        [--metrics-csv .../metrics_long.csv | --mask-index index.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simulation.evaluation import aggregate, plots, synthetic
from simulation.evaluation.metrics import load_metrics_df, DEFAULT_TAU_MM


def run(args) -> int:
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    index = None
    if args.mask_index:
        with open(args.mask_index) as f:
            index = json.load(f)
    df = load_metrics_df(args.metrics_csv, index, args.tau_mm)
    if df is None and (args.metrics_csv or args.mask_index):
        raise SystemExit(
            "[volume_stability_summary] --metrics-csv / --mask-index was given "
            f"but no metrics could be loaded ({args.metrics_csv or args.mask_index} "
            "missing or empty). Build it first (build_mask_index.py -> "
            "build_metrics.py); refusing to draw the synthetic placeholder.")
    synth = df is None
    if not synth and args.common_samples:
        df = aggregate.restrict_to_common(df)
    if synth:
        print("[volume_stability_summary] no metrics -> synthetic layout")
        cov_mean, cov_sd, on_range = synthetic.stability()
    else:
        cov_mean, cov_sd, on_range = aggregate.stability(df, args.mode)
    p = out / "volume_stability_by_resolution.png"
    plots.stability_figure(cov_mean, cov_sd, on_range, p, synthetic=synth)
    print(f"[volume_stability_summary] wrote {p}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output dir for the figure.")
    ap.add_argument("--metrics-csv", default=None, help="prebuilt metrics_long.csv.")
    ap.add_argument("--mask-index", default=None, help="MASK_INDEX json (built on the fly).")
    ap.add_argument("--mode", default=aggregate.DEFAULT_MODE,
                    help=f"sweep mode to aggregate (default {aggregate.DEFAULT_MODE}).")
    ap.add_argument("--common-samples", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Restrict to the (case, step) common to every compared "
                         "method (default on) for a fair apples-to-apples "
                         "aggregate. --no-common-samples uses each method's full "
                         "set.")
    ap.add_argument("--tau-mm", type=float, default=DEFAULT_TAU_MM)
    return ap


if __name__ == "__main__":
    sys.exit(run(build_parser().parse_args()))
