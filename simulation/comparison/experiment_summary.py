#!/usr/bin/env python3
"""Cross-experiment (thin/thick/real) comparison summary.

The per-experiment ``compare`` phase writes one paired CSV per
``(run_tag, experiment)``::

    ${work_dir}/comparison/paired_per_source__<run_tag>__<exp>.csv

Each experiment is produced by a separate pipeline run (thin / thick /
real), so nothing in the per-experiment phase ever puts the three side
by side. This driver scans the comparison/ dir, auto-discovers whichever
experiments are present, and renders the paper-style cross-experiment
view (cf. Amiranashvili Table 1 = thin vs thick; Bras = real vs
simulated):

* ``comparison/experiment_summary.csv`` / ``.txt``
      experiment x method x structure -> mean+/-std Dice, n.
* ``comparison/viz/experiments/<method>_dice_vs_eff_res_by_experiment.png``
      one figure per method, overlaying the thin/thick Dice-vs-eff_res
      curves; ``real`` is drawn as a marker-only operating point because
      it is a single real low-res acquisition (step=1), not an eff_res
      sweep.
* ``comparison/viz/experiments/overview_dice_vs_eff_res.png``
      small-multiples panel: one subplot per method, all experiments
      overlaid, for a single at-a-glance figure.

Why a separate driver (not folded into build_method_summary)
------------------------------------------------------------
build_method_summary renders ONE method/experiment bundle. This one is
intrinsically cross-file: it reads several ``__<exp>`` CSVs at once. It
reuses the same paired-CSV reader, source filter and eff_res buckets so
its numbers are identical to the per-experiment plots.

Auto-discovery
--------------
``--experiments`` defaults to scanning every ``paired_per_source__*__*``
CSV in the comparison dir, so running this after thin alone gives a
1-experiment view, and re-running after thick/real grows it. nnUNet-
sparse rows are pulled from a single canonical CSV per experiment
(preferring ``nnunet_pred``) to avoid double-counting the run-tag-
independent dense sweep.

The discovery / aggregation / table writer / per-method drawing live in
``nnunet.lib.viz``; this script wires them into the figure layout.

Usage
-----
    python simulation/comparison/experiment_summary.py \\
        --config nnunet/configs.yaml \\
        --comparison-dir comparison
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Make ``nnunet.*`` importable (repo root is two levels up from
# simulation/comparison/).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nnunet.helpers.buckets import DEFAULT_BUCKET_EDGES_MM  # noqa: E402
from nnunet.helpers.config import load_yaml  # noqa: E402
from nnunet.helpers.paired_csv import (  # noqa: E402
    apply_source_filter,
    read_paired_csv,
    resolve_source_prefix_filters,
)
from nnunet.lib.viz import (  # noqa: E402
    EXP_ORDER,
    canonical_run_tag,
    discover_experiments,
    draw_method_by_experiment,
    write_experiment_summary,
)

NNUNET_METHOD = "nnUNet-sparse"


def run(args) -> int:
    cfg = load_yaml(Path(args.config))
    comparison_dir = Path(args.comparison_dir)
    out_dir = (Path(args.out_dir) if args.out_dir
               else comparison_dir / "viz" / "experiments")
    out_dir.mkdir(parents=True, exist_ok=True)
    edges = list(cfg.get("summary_bucket_edges_mm",
                         list(DEFAULT_BUCKET_EDGES_MM)))
    include_pref, exclude_pref = resolve_source_prefix_filters(
        args.include_source_prefixes, args.exclude_source_prefixes, cfg)

    # run_tag -> CNISP method label, from config.
    run_to_method: Dict[str, str] = {}
    for entry in cfg.get("cnisp_runs_to_compare", []) or []:
        rt = str(entry.get("run_tag", ""))
        ml = str(entry.get("method_label", ""))
        if rt and ml:
            run_to_method[rt] = ml
    # real_pair has its own conventional label even if absent from config.
    run_to_method.setdefault("real_pair", "CNISP-realPair")

    discovered = discover_experiments(comparison_dir)
    if not discovered:
        print(f"[build_experiment_summary] no paired_per_source__*__*.csv "
              f"under {comparison_dir}; nothing to aggregate.", file=sys.stderr)
        return 0
    wanted = (set(s.strip() for s in args.experiments.split(",") if s.strip())
              if args.experiments else None)

    # data[method][exp] = filtered rows.
    data: Dict[str, Dict[str, List[Dict]]] = defaultdict(dict)
    for exp, run_tags in discovered.items():
        if wanted is not None and exp not in wanted:
            continue
        # Per-CNISP-run methods.
        for run_tag in run_tags:
            method = run_to_method.get(run_tag)
            if not method:
                continue
            csv_path = comparison_dir / f"paired_per_source__{run_tag}__{exp}.csv"
            try:
                rows = read_paired_csv(csv_path, method)
            except SystemExit:
                continue  # method absent in this CSV; skip quietly
            data[method][exp] = apply_source_filter(rows, include_pref, exclude_pref)
        # nnUNet-sparse once per experiment from the canonical CSV.
        canon = canonical_run_tag(run_tags)
        if canon:
            csv_path = comparison_dir / f"paired_per_source__{canon}__{exp}.csv"
            try:
                rows = read_paired_csv(csv_path, NNUNET_METHOD)
                data[NNUNET_METHOD][exp] = apply_source_filter(
                    rows, include_pref, exclude_pref)
            except SystemExit:
                pass
            # nnUNet-C (control C) is also run-tag-independent: pull it once
            # per experiment from the same canonical CSV when configured.
            nnunet_c_label = cfg.get("nnunet_c_method_label")
            if nnunet_c_label:
                try:
                    rows = read_paired_csv(csv_path, nnunet_c_label)
                    data[nnunet_c_label][exp] = apply_source_filter(
                        rows, include_pref, exclude_pref)
                except SystemExit:
                    pass

    if not data:
        print("[build_experiment_summary] no method rows matched; check "
              "cnisp_runs_to_compare and the CSV method labels.",
              file=sys.stderr)
        return 0

    # ── Tables ──
    write_experiment_summary(data, comparison_dir / "experiment_summary.csv",
                             comparison_dir / "experiment_summary.txt")

    # ── Per-method overlay figures ──
    methods = list(data.keys())
    written: List[Path] = []
    for method in methods:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        if draw_method_by_experiment(ax, method, data[method], edges):
            fig.suptitle(f"{method}: Dice vs effective resolution "
                         f"by experiment", fontsize=12, fontweight="bold")
            fig.tight_layout()
            p = out_dir / f"{method}_dice_vs_eff_res_by_experiment.png"
            fig.savefig(str(p), dpi=150, bbox_inches="tight")
            written.append(p)
        plt.close(fig)

    # ── Overview small-multiples (one subplot per method) ──
    plot_methods = [m for m in methods if any(data[m].get(e) for e in EXP_ORDER)]
    if plot_methods:
        n = len(plot_methods)
        ncol = min(3, n)
        nrow = (n + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 5 * nrow),
                                 squeeze=False)
        for idx, method in enumerate(plot_methods):
            ax = axes[idx // ncol][idx % ncol]
            draw_method_by_experiment(ax, method, data[method], edges)
        for j in range(len(plot_methods), nrow * ncol):
            axes[j // ncol][j % ncol].axis("off")
        fig.suptitle("Cross-experiment Dice vs effective resolution "
                     "(thin / thick / real)", fontsize=13, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        p = out_dir / "overview_dice_vs_eff_res.png"
        fig.savefig(str(p), dpi=150, bbox_inches="tight")
        written.append(p)
        plt.close(fig)

    exps_present = sorted({e for m in data.values() for e in m},
                          key=lambda e: EXP_ORDER.index(e)
                          if e in EXP_ORDER else 99)
    print(f"[build_experiment_summary] experiments={exps_present} "
          f"methods={methods}")
    print(f"  {comparison_dir/'experiment_summary.csv'}")
    print(f"  {comparison_dir/'experiment_summary.txt'}")
    for p in written:
        print(f"  {p}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="nnunet/configs.yaml")
    ap.add_argument("--comparison-dir", required=True,
                    help="repo-level comparison/ dir (holds paired_per_source"
                         "__<run_tag>__<exp>.csv).")
    ap.add_argument("--out-dir", default=None,
                    help="Default: <comparison-dir>/viz/experiments")
    ap.add_argument("--experiments", default=None,
                    help="Comma-separated subset to include (thin,thick,"
                         "real). Default: auto-discover from the CSVs.")
    ap.add_argument("--include-source-prefixes", default=None)
    ap.add_argument("--exclude-source-prefixes", default=None)
    return ap


if __name__ == "__main__":
    sys.exit(run(build_parser().parse_args()))
