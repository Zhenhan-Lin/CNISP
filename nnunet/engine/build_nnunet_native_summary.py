#!/usr/bin/env python3
"""nnUNet-only native-space Dice summary, indexed by sparsification step.

``compare_native.py`` writes ``comparison/paired_per_source__<suffix>.csv``
with nnUNet AND CNISP rows interleaved (long format, one row per
``(source_id, method, step_size, structure, dice)``). For analysing the
nnUNet sparse-CT curve on its own -- e.g. how its native Dice degrades
with step independently of any CNISP run -- that mixed long file is
awkward. This driver carves out just the nnUNet rows and re-shapes them
into the two tables you actually want to open:

* ``nnunet_native_per_source__<suffix>.csv``  -- WIDE, one row per
  ``(source_id, gt_source, step_size)`` with a column per structure
  (ON / Globe / Fat / Recti / mean) plus ``eff_res_mm``. This mirrors
  the shape of CNISP's own ``test_results.csv`` so the two are trivial
  to line up in pandas/Excel.
* ``nnunet_native_by_step__<suffix>.csv``     -- aggregated by
  ``step_size``: ``n_sources`` and ``mean +/- std`` for each structure
  (and the 4-class mean), plus the mean ``eff_res_mm`` in that step.

and one figure:

* ``nnunet_native_dice_vs_step__<suffix>.png`` -- overall mean Dice vs
  step (errorbar) on the left, the four per-class curves vs step on the
  right.

This is deliberately a SEPARATE artifact from
``build_method_summary.py`` (which buckets the SAME nnUNet rows by
effective-resolution rather than by step). Use this one when you want
the per-step view; use that one when you want the eff_res-bucket view.

Why feed off the paired CSV instead of recomputing Dice
-------------------------------------------------------
The paired CSV is already the single source of truth for every Dice
number in the comparison (same GT handling, same chk_*/atlas GT swap,
same eff_res values from CNISP's ``sweep_results.pkl``). Re-deriving the
nnUNet summary from it guarantees these tables can never disagree with
the head-to-head comparison; we only re-shape, never re-measure.

Usage
-----
    # default: nnUNet-sparse rows out of the nnunet_pred paired CSV
    # (a strict superset of the atlas_gt CSV's nnUNet rows), chk_*
    # filtered exactly like the other viz scripts.
    python nnunet/engine/build_nnunet_native_summary.py \\
        --config nnunet/configs.yaml \\
        --paired-csv ${work_dir}/comparison/paired_per_source__nnunet_pred__thick.csv

    # keep every source (including chk_*) for raw inspection:
    python nnunet/engine/build_nnunet_native_summary.py \\
        --config nnunet/configs.yaml \\
        --paired-csv .../paired_per_source__nnunet_pred__thick.csv \\
        --exclude-source-prefixes ""
"""

from __future__ import annotations

import argparse
import csv
import math
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
    NNUNET_METHOD_LABEL,
    STRUCT_ORDER,
)
from nnunet.helpers.config import load_yaml  # noqa: E402
from nnunet.helpers.paired_csv import (  # noqa: E402
    apply_source_filter,
    read_paired_csv,
    resolve_source_prefix_filters,
)


CLASS_COLORS = {
    "ON": "#d62728",
    "Globe": "#1f77b4",
    "Fat": "#2ca02c",
    "Recti": "#9467bd",
}
# Structure columns in display order, including the 4-class mean last.
COLS = STRUCT_ORDER + ["mean"]


def pivot_per_source(
    rows: List[Dict],
) -> List[Dict]:
    """Re-shape long ``method``-filtered rows into wide per-(source, step) rows.

    Each output dict has ``source_id``, ``gt_source``, ``step_size`` (int),
    ``eff_res_mm`` (float or NaN), and one float per entry in :data:`COLS`
    (missing structures become NaN). Rows are sorted by
    ``(source_id, step_size)`` so the CSV reads naturally.
    """
    # (source_id, gt_source, step) -> {structure: dice, "_eff": eff}
    acc: Dict[Tuple[str, str, int], Dict[str, float]] = defaultdict(dict)
    for r in rows:
        key = (r["source_id"], r.get("gt_source", ""), r["step_size"])
        acc[key][r["structure"]] = r["dice"]
        # eff_res is constant across structures of the same (source, step);
        # keep whichever non-NaN value we see.
        eff = r["eff_res_mm"]
        if not math.isnan(eff):
            acc[key]["_eff"] = eff

    out: List[Dict] = []
    for (sid, gt_source, step) in sorted(acc, key=lambda k: (k[0], k[2])):
        d = acc[(sid, gt_source, step)]
        row: Dict = {
            "source_id": sid,
            "gt_source": gt_source,
            "step_size": step,
            "eff_res_mm": d.get("_eff", float("nan")),
        }
        for c in COLS:
            row[c] = d.get(c, float("nan"))
        out.append(row)
    return out


def aggregate_by_step(
    wide_rows: List[Dict],
) -> List[Dict]:
    """Aggregate wide per-(source, step) rows by ``step_size``.

    Returns a list (sorted by step) of dicts with ``step_size``,
    ``n_sources``, ``eff_res_mm`` (mean over sources in that step), and
    ``<struct>_mean`` / ``<struct>_std`` for each entry in :data:`COLS`.
    """
    by_step: Dict[int, List[Dict]] = defaultdict(list)
    for r in wide_rows:
        by_step[r["step_size"]].append(r)

    out: List[Dict] = []
    for step in sorted(by_step):
        group = by_step[step]
        effs = [r["eff_res_mm"] for r in group if not math.isnan(r["eff_res_mm"])]
        agg: Dict = {
            "step_size": step,
            "n_sources": len(group),
            "eff_res_mm": float(np.mean(effs)) if effs else float("nan"),
        }
        for c in COLS:
            vals = [r[c] for r in group if not math.isnan(r[c])]
            if vals:
                arr = np.asarray(vals, dtype=np.float64)
                agg[f"{c}_mean"] = float(arr.mean())
                agg[f"{c}_std"] = float(arr.std())
            else:
                agg[f"{c}_mean"] = float("nan")
                agg[f"{c}_std"] = float("nan")
        out.append(agg)
    return out


def _fmt(v: float, nd: int = 6) -> str:
    return "" if (v is None or math.isnan(v)) else f"{v:.{nd}f}"


def write_per_source_csv(wide_rows: List[Dict], out_path: Path) -> None:
    fieldnames = (["source_id", "gt_source", "step_size", "eff_res_mm"]
                  + COLS)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fieldnames)
        for r in wide_rows:
            w.writerow([
                r["source_id"], r["gt_source"], r["step_size"],
                _fmt(r["eff_res_mm"], 4),
                *[_fmt(r[c]) for c in COLS],
            ])


def write_by_step_csv(step_rows: List[Dict], out_path: Path) -> None:
    fieldnames = ["step_size", "n_sources", "eff_res_mm"]
    for c in COLS:
        fieldnames += [f"{c}_mean", f"{c}_std"]
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fieldnames)
        for r in step_rows:
            row = [r["step_size"], r["n_sources"], _fmt(r["eff_res_mm"], 4)]
            for c in COLS:
                row += [_fmt(r[f"{c}_mean"], 4), _fmt(r[f"{c}_std"], 4)]
            w.writerow(row)


def plot_dice_vs_step(
    step_rows: List[Dict],
    method: str,
    out_path: Path,
) -> None:
    """Two panels: overall mean Dice vs step, and per-class Dice vs step."""
    steps = [r["step_size"] for r in step_rows]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5))

    # ── Panel 0: overall 4-class mean Dice vs step ────────────────
    ys = [r["mean_mean"] for r in step_rows]
    es = [r["mean_std"] for r in step_rows]
    ax0.errorbar(steps, ys, yerr=es, fmt="o-", capsize=4, color="#444")
    for r in step_rows:
        if not math.isnan(r["mean_mean"]):
            ax0.annotate(f"{r['mean_mean']:.3f}\nn={r['n_sources']}",
                         (r["step_size"], r["mean_mean"]),
                         textcoords="offset points", xytext=(0, 10),
                         ha="center", fontsize=8, color="#444")
    ax0.set_xlabel("sparsification step (keep every Nth slice)")
    ax0.set_ylabel("mean Dice (4 foreground classes)")
    ax0.set_title(f"{method}: overall native Dice vs step")
    ax0.set_ylim(0, 1)
    ax0.grid(True, alpha=0.3)

    # ── Panel 1: per-class Dice vs step ───────────────────────────
    for c in STRUCT_ORDER:
        ys_c = [r[f"{c}_mean"] for r in step_rows]
        es_c = [r[f"{c}_std"] for r in step_rows]
        ax1.errorbar(steps, ys_c, yerr=es_c, fmt="o-", capsize=3,
                     color=CLASS_COLORS[c], label=c)
    ax1.set_xlabel("sparsification step (keep every Nth slice)")
    ax1.set_ylabel("Dice")
    ax1.set_title(f"{method}: per-class native Dice vs step")
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower left", fontsize=8, ncol=2)

    fig.suptitle(f"{method}: native-space Dice vs sparsification step",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_nnunet_native_summary(
    paired_csv: Path,
    out_dir: Path,
    method: str,
    include_prefixes: List[str],
    exclude_prefixes: List[str],
    suffix: str,
) -> List[Path]:
    """Carve nnUNet rows out of ``paired_csv`` and write the per-step bundle.

    Returns the list of written output paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_paired_csv(paired_csv, method)
    n_before = len(rows)
    rows = apply_source_filter(rows, include_prefixes, exclude_prefixes)
    if include_prefixes or exclude_prefixes:
        print(f"[nnunet_native_summary] source filter: "
              f"include={include_prefixes!r} exclude={exclude_prefixes!r} "
              f"-> {len(rows)}/{n_before} rows kept.", file=sys.stderr)
    if not rows:
        raise SystemExit(
            f"All rows filtered out for method={method!r}; relax the "
            f"include/exclude prefixes or check the method label.")

    wide_rows = pivot_per_source(rows)
    step_rows = aggregate_by_step(wide_rows)

    per_source_csv = out_dir / f"nnunet_native_per_source{suffix}.csv"
    by_step_csv = out_dir / f"nnunet_native_by_step{suffix}.csv"
    png = out_dir / f"nnunet_native_dice_vs_step{suffix}.png"

    write_per_source_csv(wide_rows, per_source_csv)
    write_by_step_csv(step_rows, by_step_csv)
    plot_dice_vs_step(step_rows, method, png)

    n_sources = len({r["source_id"] for r in wide_rows})
    print(f"[nnunet_native_summary] {method}: {len(wide_rows)} (source,step) "
          f"row(s) across {n_sources} source(s), "
          f"{len(step_rows)} step bucket(s).")
    for p in (per_source_csv, by_step_csv, png):
        print(f"  {p}")
    return [per_source_csv, by_step_csv, png]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="nnunet/configs.yaml")
    ap.add_argument("--paired-csv", required=True,
                    help="Path to comparison/paired_per_source__<suffix>.csv "
                         "(written by compare_native.py / the compare phase).")
    ap.add_argument("--out-dir", default=None,
                    help="Output directory. Default: the paired CSV's parent "
                         "dir + '/nnunet_native/'.")
    ap.add_argument("--method", default=NNUNET_METHOD_LABEL,
                    help=f"Method label to extract (default "
                         f"{NNUNET_METHOD_LABEL!r}). Pass a CNISP label to "
                         f"get the same per-step bundle for a CNISP run.")
    ap.add_argument("--out-suffix", default=None,
                    help="Suffix appended to output filenames. Default: the "
                         "paired CSV's '__...' suffix so outputs line up with "
                         "the comparison they came from.")
    ap.add_argument("--include-source-prefixes", default=None,
                    help="Comma-separated source_id prefixes to keep. Default: "
                         "'viz_include_source_prefixes' from --config.")
    ap.add_argument("--exclude-source-prefixes", default=None,
                    help="Comma-separated source_id prefixes to drop. Default: "
                         "'viz_exclude_source_prefixes' from --config "
                         "(usually 'chk_'). Pass an empty string to keep ALL "
                         "sources for raw inspection.")
    args = ap.parse_args()

    cfg = load_yaml(Path(args.config))
    paired_csv = Path(args.paired_csv)
    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        out_dir = paired_csv.parent / "nnunet_native"

    # Default the file suffix to the paired CSV's own '__...' tail so the
    # nnUNet bundle is unambiguously tied to its source comparison.
    if args.out_suffix is not None:
        suffix = args.out_suffix
    else:
        stem = paired_csv.stem  # e.g. paired_per_source__nnunet_pred__thick
        marker = "paired_per_source"
        suffix = stem[len(marker):] if stem.startswith(marker) else f"__{stem}"

    include_prefixes, exclude_prefixes = resolve_source_prefix_filters(
        args.include_source_prefixes, args.exclude_source_prefixes, cfg,
    )

    build_nnunet_native_summary(
        paired_csv=paired_csv,
        out_dir=out_dir,
        method=args.method,
        include_prefixes=include_prefixes,
        exclude_prefixes=exclude_prefixes,
        suffix=suffix,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
