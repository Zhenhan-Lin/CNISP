#!/usr/bin/env python3
"""Driver: volume-veracity figure.

Reads the metrics CSV (or builds it from a MASK_INDEX; or renders the synthetic
illustrative layout), aggregates per-arm volume agreement for one structure, and
writes ``signed_volume_error.png`` by default (signed volume error across methods
only). Pass ``--bland-altman`` to instead write the full ``volume_agreement_bland_altman.png``
(Bland-Altman for nnU-Net + Proposed + the signed-error violins) plus the per-arm
BA panels. The per-arm bias CSV is always written.

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
    if df is None and (args.metrics_csv or args.mask_index):
        raise SystemExit(
            "[volume_agreement_summary] --metrics-csv / --mask-index was given "
            f"but no metrics could be loaded ({args.metrics_csv or args.mask_index} "
            "missing or empty). Build it first (build_mask_index.py -> "
            "build_metrics.py); refusing to draw the synthetic placeholder.")
    synth = df is None
    if not synth and args.common_samples:
        df = aggregate.restrict_to_common(df)
    if synth:
        print("[volume_agreement_summary] no metrics -> synthetic layout")
        per_arm, signed = synthetic.volume_agreement()
    else:
        per_arm, signed = aggregate.volume_agreement(df, args.ba_structure)
    # Default: signed volume error only (no Bland-Altman). --bland-altman restores
    # the full 3-panel BA figure + the per-arm BA panels.
    p = (out / "volume_agreement_bland_altman.png" if args.bland_altman
         else out / "signed_volume_error.png")
    plots.volume_agreement_figure(per_arm, signed, p, synthetic=synth,
                                  bland_altman=args.bland_altman)
    print(f"[volume_agreement_summary] wrote {p}")

    # ── per-arm Bland-Altman bias table (ALL 5 arms) + one panel per arm ──
    # The combined figure above only draws nnUNet + Proposed; here we quantify
    # every arm's volume bias (on the SAME restricted sample as the figure) so
    # e.g. Proposed's +bias can be read off next to the other arms, print it to
    # stdout, dump it to CSV, and (with --bland-altman) render a standalone panel
    # per arm in a subdir. The bias CSV is cheap numbers and always written; the
    # BA PNG panels are gated behind --bland-altman (they are "the BA plots").
    if not synth:
        import csv as _csv
        per_all, stats = aggregate.volume_agreement_per_arm(df, args.ba_structure)
        cols = ["arm", "structure", "n", "bias_mm3", "sd_diff_mm3",
                "loa_lo_mm3", "loa_hi_mm3", "ccc",
                "mean_vol_pred_mm3", "mean_vol_gt_mm3", "mean_signed_pct"]
        tbl = out / "bland_altman_bias_by_arm.csv"
        with open(tbl, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for row in stats:
                w.writerow({k: row[k] for k in cols})
        print(f"[volume_agreement_summary] per-arm Bland-Altman bias "
              f"({args.ba_structure}), same sample as the figure:")
        print(f"    {'arm':<14}{'n':>5}{'bias(mm³)':>12}{'±LoA(mm³)':>12}"
              f"{'CCC':>7}{'signed%':>9}")
        for row in stats:
            n = row["n"]
            if n:
                print(f"    {row['arm']:<14}{n:>5}{row['bias_mm3']:>12.1f}"
                      f"{1.96*row['sd_diff_mm3']:>12.1f}{row['ccc']:>7.2f}"
                      f"{row['mean_signed_pct']:>9.1f}")
            else:
                print(f"    {row['arm']:<14}{n:>5}{'(no rows)':>12}")
        print(f"[volume_agreement_summary] bias table -> {tbl}")
        if args.bland_altman:
            sub = out / "bland_altman_per_arm"
            sub.mkdir(parents=True, exist_ok=True)
            drawn, skipped = [], []
            for m in aggregate.METHODS:
                d = per_all[m]
                fn = sub / f"bland_altman_{m.replace(' ', '_')}.png"
                if plots.single_bland_altman_figure(
                        d["v_pred"], d["v_gt"], d["thickness"], m, args.ba_structure, fn):
                    drawn.append(m)
                else:
                    skipped.append((m, len(d["v_pred"])))
            print(f"[volume_agreement_summary] per-arm panels ({len(drawn)}) -> {sub}/")
            for m, n in skipped:
                print(f"    [skip panel] {m}: n={n} (<2 points)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output dir for the figure.")
    ap.add_argument("--metrics-csv", default=None, help="prebuilt metrics_long.csv.")
    ap.add_argument("--mask-index", default=None, help="MASK_INDEX json (built on the fly).")
    ap.add_argument("--ba-structure", default=aggregate.DEFAULT_BA_STRUCTURE,
                    help=f"structure for Bland-Altman (default {aggregate.DEFAULT_BA_STRUCTURE}).")
    ap.add_argument("--bland-altman", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="Draw the Bland-Altman figure + per-arm BA panels. Default "
                         "OFF: only 'signed_volume_error.png' is written (the bias CSV "
                         "is always written).")
    ap.add_argument("--mode", default=aggregate.DEFAULT_MODE)
    ap.add_argument("--common-samples", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Restrict to the (case, step) common to every compared "
                         "method (default on). --no-common-samples uses each "
                         "method's full set.")
    ap.add_argument("--tau-mm", type=float, default=DEFAULT_TAU_MM)
    return ap


if __name__ == "__main__":
    sys.exit(run(build_parser().parse_args()))
