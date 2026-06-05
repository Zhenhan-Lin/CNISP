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

Usage
-----
    python nnunet/engine/build_experiment_summary.py \\
        --config nnunet/configs.yaml \\
        --comparison-dir ${work_dir}/comparison
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Make ``nnunet.*`` importable when run as ``python nnunet/engine/...``.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nnunet.helpers.buckets import (  # noqa: E402
    DEFAULT_BUCKET_EDGES_MM,
    STRUCT_ORDER,
    assign_bucket,
    bucket_sort_key,
)
from nnunet.helpers.config import load_yaml  # noqa: E402
from nnunet.helpers.paired_csv import (  # noqa: E402
    apply_source_filter,
    read_paired_csv,
    resolve_source_prefix_filters,
)

NNUNET_METHOD = "nnUNet-sparse"
# Stable experiment order + per-experiment style. thin/thick are eff_res
# sweeps (lines); real is a single operating point (markers, no line).
EXP_ORDER = ["thin", "thick", "real"]
EXP_STYLE = {
    "thin":  {"color": "#1f77b4", "marker": "o", "line": "-"},
    "thick": {"color": "#d62728", "marker": "s", "line": "--"},
    "real":  {"color": "#2ca02c", "marker": "D", "line": ""},  # point only
}


def _discover(comparison_dir: Path) -> Dict[str, List[str]]:
    """Map experiment -> sorted run_tags found on disk.

    Filenames are ``paired_per_source__<run_tag>__<exp>.csv``; run_tag
    may itself contain single underscores (atlas_gt, nnunet_pred,
    real_pair) so we split on the LAST ``__`` boundary.
    """
    found: Dict[str, List[str]] = defaultdict(list)
    for p in sorted(comparison_dir.glob("paired_per_source__*__*.csv")):
        stem = p.stem  # drop .csv
        # rsplit on the experiment delimiter: everything before the final
        # '__' is the run_tag, the tail is the experiment token.
        head, _, exp = stem.rpartition("__")
        run_tag = head[len("paired_per_source__"):]
        if not run_tag or not exp:
            continue
        found[exp].append(run_tag)
    return {e: sorted(set(v)) for e, v in found.items()}


def _canonical_run_tag(run_tags: List[str]) -> Optional[str]:
    """Pick the run_tag whose CSV holds the canonical nnUNet-sparse rows
    (prefer nnunet_pred -- a strict superset -- else the first present)."""
    if not run_tags:
        return None
    return "nnunet_pred" if "nnunet_pred" in run_tags else run_tags[0]


def _series_curve(
    rows: List[Dict], edges: List[float],
) -> Tuple[List[float], List[float], List[float], List[int]]:
    """Bucketed overall-mean (structure=='mean') Dice vs eff_res.

    Returns sorted-by-eff_res ``(xs, ys, stds, ns)``.
    """
    bucket_eff: Dict[str, List[float]] = defaultdict(list)
    bucket_mean: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        if r["structure"] != "mean":
            continue
        _, label = assign_bucket(r["eff_res_mm"], edges)
        bucket_mean[label].append(r["dice"])
        if not np.isnan(r["eff_res_mm"]):
            bucket_eff[label].append(r["eff_res_mm"])
    labels = sorted(bucket_mean.keys(), key=bucket_sort_key)
    xs: List[float] = []
    ys: List[float] = []
    es: List[float] = []
    ns: List[int] = []
    for lab in labels:
        effs = bucket_eff.get(lab, [])
        vals = bucket_mean.get(lab, [])
        if not effs or not vals:
            continue
        xs.append(float(np.mean(effs)))
        arr = np.asarray(vals)
        ys.append(float(arr.mean()))
        es.append(float(arr.std()))
        ns.append(len(arr))
    order = np.argsort(xs) if xs else []
    return ([xs[i] for i in order], [ys[i] for i in order],
            [es[i] for i in order], [ns[i] for i in order])


def _overall_stats(rows: List[Dict], structure: str) -> Tuple[float, float, int]:
    vals = [r["dice"] for r in rows if r["structure"] == structure]
    if not vals:
        return float("nan"), float("nan"), 0
    arr = np.asarray(vals)
    return float(arr.mean()), float(arr.std()), int(arr.size)


def _draw_method(ax, method: str, by_exp: Dict[str, List[Dict]],
                 edges: List[float]) -> bool:
    """Overlay each experiment's curve/point for one method. Returns True
    if anything was drawn."""
    drawn = False
    for exp in EXP_ORDER:
        rows = by_exp.get(exp)
        if not rows:
            continue
        st = EXP_STYLE.get(exp, {"color": "#444", "marker": "o", "line": "-"})
        xs, ys, es, _ = _series_curve(rows, edges)
        if not xs:
            continue
        drawn = True
        if st["line"]:  # thin/thick -> connected sweep curve
            ax.errorbar(xs, ys, yerr=es, fmt=st["marker"] + st["line"],
                        capsize=3, color=st["color"], label=exp)
        else:           # real -> operating point(s), no connecting line
            ax.errorbar(xs, ys, yerr=es, fmt=st["marker"], markersize=9,
                        capsize=3, color=st["color"], label=f"{exp} (real)")
    if drawn:
        ax.set_xlabel("effective resolution (mm, through-plane)")
        ax.set_ylabel("mean Dice (4 foreground classes)")
        ax.set_title(method)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left", fontsize=8)
    return drawn


def _write_summary(data: Dict[str, Dict[str, List[Dict]]], out_csv: Path,
                   out_txt: Path) -> None:
    methods = list(data.keys())
    structs = STRUCT_ORDER + ["mean"]
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "method", "structure",
                    "mean_dice", "std_dice", "n_observations"])
        for exp in EXP_ORDER:
            for method in methods:
                rows = data[method].get(exp)
                if not rows:
                    continue
                for s in structs:
                    m, sd, n = _overall_stats(rows, s)
                    if n == 0:
                        continue
                    w.writerow([exp, method, s, f"{m:.4f}", f"{sd:.4f}", n])

    col_w = 20
    with open(out_txt, "w") as f:
        f.write("=" * 78 + "\n")
        f.write("Cross-experiment Dice summary (thin / thick / real)\n")
        f.write("=" * 78 + "\n\n")
        f.write("Overall = mean over all (source, step) observations per "
                "(experiment, method, structure).\n")
        f.write("Source filter matches the per-experiment plots.\n\n")
        for method in methods:
            present = [e for e in EXP_ORDER if data[method].get(e)]
            if not present:
                continue
            f.write(f"### {method}\n")
            header = "structure".ljust(11) + "".join(
                e.ljust(col_w) for e in present)
            f.write(header + "\n")
            for s in structs:
                row = s.ljust(11)
                for exp in present:
                    m, sd, n = _overall_stats(data[method][exp], s)
                    cell = ("n/a" if n == 0
                            else f"{m:.3f}+/-{sd:.3f}(n={n})")
                    row += cell.ljust(col_w)
                f.write(row + "\n")
            f.write("\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="nnunet/configs.yaml")
    ap.add_argument("--comparison-dir", required=True,
                    help="${work_dir}/comparison (holds paired_per_source"
                         "__<run_tag>__<exp>.csv).")
    ap.add_argument("--out-dir", default=None,
                    help="Default: <comparison-dir>/viz/experiments")
    ap.add_argument("--experiments", default=None,
                    help="Comma-separated subset to include (thin,thick,"
                         "real). Default: auto-discover from the CSVs.")
    ap.add_argument("--include-source-prefixes", default=None)
    ap.add_argument("--exclude-source-prefixes", default=None)
    args = ap.parse_args()

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

    discovered = _discover(comparison_dir)
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
        canon = _canonical_run_tag(run_tags)
        if canon:
            csv_path = comparison_dir / f"paired_per_source__{canon}__{exp}.csv"
            try:
                rows = read_paired_csv(csv_path, NNUNET_METHOD)
                data[NNUNET_METHOD][exp] = apply_source_filter(
                    rows, include_pref, exclude_pref)
            except SystemExit:
                pass

    if not data:
        print("[build_experiment_summary] no method rows matched; check "
              "cnisp_runs_to_compare and the CSV method labels.",
              file=sys.stderr)
        return 0

    # ── Tables ──
    _write_summary(data, comparison_dir / "experiment_summary.csv",
                   comparison_dir / "experiment_summary.txt")

    # ── Per-method overlay figures ──
    methods = list(data.keys())
    written: List[Path] = []
    for method in methods:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        if _draw_method(ax, method, data[method], edges):
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
            _draw_method(ax, method, data[method], edges)
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


if __name__ == "__main__":
    sys.exit(main())
