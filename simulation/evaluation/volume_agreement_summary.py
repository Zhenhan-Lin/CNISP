#!/usr/bin/env python3
"""Driver: volume-agreement figure (Bland-Altman + signed volume error).

Reads the metrics CSV (or builds it from a MASK_INDEX; or renders the synthetic
illustrative layout), aggregates per-arm volume agreement for one structure, and
writes ``volume_agreement_bland_altman.png``.

Sibling of ``simulation/comparison/paired_summary.py``.

Usage:
    python simulation/evaluation/volume_agreement_summary.py \
        --out comparison/viz/evaluation__thick --mode thick --ba-structure Globe \
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
    synth = df is None
    if synth:
        print("[volume_agreement_summary] no metrics -> synthetic layout")
        per_arm, signed = synthetic.volume_agreement()
    else:
        per_arm, signed = aggregate.volume_agreement(df, args.ba_structure)
    p = out / "volume_agreement_bland_altman.png"
    plots.volume_agreement_figure(per_arm, signed, p, synthetic=synth)
    print(f"[volume_agreement_summary] wrote {p}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output dir for the figure.")
    ap.add_argument("--metrics-csv", default=None, help="prebuilt metrics_long.csv.")
    ap.add_argument("--mask-index", default=None, help="MASK_INDEX json (built on the fly).")
    ap.add_argument("--ba-structure", default=aggregate.DEFAULT_BA_STRUCTURE,
                    help=f"structure for Bland-Altman (default {aggregate.DEFAULT_BA_STRUCTURE}).")
    ap.add_argument("--mode", default=aggregate.DEFAULT_MODE)
    ap.add_argument("--tau-mm", type=float, default=DEFAULT_TAU_MM)
    return ap


if __name__ == "__main__":
    sys.exit(run(build_parser().parse_args()))
