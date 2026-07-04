#!/usr/bin/env python3
"""Driver: surface-quality figure (ASSD / HD95 / Surface-Dice boxplots).

Reads the metrics CSV (or builds it from a MASK_INDEX; or renders the synthetic
illustrative layout), aggregates per-method surface-metric distributions, and
writes ``surface_quality_metrics.png``.

Sibling of ``simulation/comparison/experiment_summary.py``.

Usage:
    python simulation/evaluation/surface_quality_summary.py \
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
            "[surface_quality_summary] --metrics-csv / --mask-index was given "
            f"but no metrics could be loaded ({args.metrics_csv or args.mask_index} "
            "missing or empty). Build it first (build_mask_index.py -> "
            "build_metrics.py); refusing to draw the synthetic placeholder.")
    synth = df is None
    if synth:
        print("[surface_quality_summary] no metrics -> synthetic layout")
        metrics = synthetic.surface()
    else:
        metrics = aggregate.surface(df, args.mode)
    p = out / "surface_quality_metrics.png"
    plots.surface_figure(metrics, p, synthetic=synth)
    print(f"[surface_quality_summary] wrote {p}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output dir for the figure.")
    ap.add_argument("--metrics-csv", default=None, help="prebuilt metrics_long.csv.")
    ap.add_argument("--mask-index", default=None, help="MASK_INDEX json (built on the fly).")
    ap.add_argument("--mode", default=aggregate.DEFAULT_MODE)
    ap.add_argument("--tau-mm", type=float, default=DEFAULT_TAU_MM)
    return ap


if __name__ == "__main__":
    sys.exit(run(build_parser().parse_args()))
