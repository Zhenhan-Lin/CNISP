#!/usr/bin/env python3
"""Per-method Dice summary by effective-resolution bucket.

Reads ``{work_dir}/comparison/paired_per_source__<run_tag>.csv`` (one
row per ``(source_id, method, step_size, structure, dice)``, written
by ``nnunet/compare_native.py``) and produces, for the requested method
label (e.g. ``nnUNet-sparse`` / ``CNISP-atlasGT`` / ``CNISP-nnUNetPred``),
a matched set of artifacts:

* ``{out_dir}/{method}_per_source.csv``          - long, filtered to method
* ``{out_dir}/{method}_summary_by_eff_res.csv``  - aggregated by
                                                    ``(eff_res_bucket, structure)``
* ``{out_dir}/{method}_summary_by_eff_res.txt``  - human-readable wide table
* ``{out_dir}/{method}_recon_summary.png``       - 3-subplot figure:
    1. overall mean Dice vs eff_res     (errorbar over sources in bucket)
    2. per-class Dice vs eff_res        (4 lines: ON / Globe / Fat / Recti)
    3. per-case Dice distribution       (boxplot + scatter per bucket)

Why every method shares one driver
----------------------------------
``compare_native.py`` already emits
``comparison/paired_per_source__<run_tag>.csv`` with both methods'
rows interleaved -- same source set, same eff_res values, same bucket
edges. Driving the per-method viz off that file guarantees the CNISP
and nnUNet summaries never drift out of sync (same n_sources, same
axis), and the same plotting code renders any method just by changing
``--method``.

Notes
-----
* ``paired_per_source.csv`` only carries dense Dice -- ``compare_native``
  never computes "observed-only" Dice. That's why this viz drops the
  observed line CNISP's old ``recon_summary.png`` used to plot.
* eff_res values are read straight from the CSV (which inherited them
  from CNISP's ``sweep_results.pkl`` via ``compare_native``), so all
  methods share the same x-axis sample per source/step.

Usage
-----
    # nnUNet (deployment-mode shared chk_* GT)
    python nnunet/engine/build_method_summary.py \\
        --config nnunet/configs.yaml \\
        --method nnUNet-sparse \\
        --paired-csv work_dir/comparison/paired_per_source__nnunet_pred.csv \\
        --out-dir    work_dir/comparison/viz/nnUNet-sparse__nnunet_pred

    # CNISP-atlasGT
    python nnunet/engine/build_method_summary.py \\
        --config nnunet/configs.yaml \\
        --method CNISP-atlasGT \\
        --paired-csv work_dir/comparison/paired_per_source__atlas_gt.csv \\
        --out-dir    cnisp_output_basedir/<model>/viz/atlas_gt
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402


# Kept in sync with compare_native.STRUCT_ORDER on purpose. If a new
# foreground structure is added in one place, both files have to grow.
STRUCT_ORDER = ["ON", "Globe", "Fat", "Recti"]
CLASS_COLORS = {
    "ON": "#d62728",
    "Globe": "#1f77b4",
    "Fat": "#2ca02c",
    "Recti": "#9467bd",
}


def _load_yaml(p: Path) -> Dict:
    with open(p) as f:
        return yaml.safe_load(f) or {}


def _read_paired_csv(p: Path, method: str) -> List[Dict]:
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Run `nnunet/compare_native.py` first "
            f"(or the `compare` phase of run_pipeline.sh)."
        )
    rows: List[Dict] = []
    with open(p) as f:
        for r in csv.DictReader(f):
            if r.get("method") != method:
                continue
            try:
                step = int(float(r["step_size"]))
                dice = float(r["dice"])
            except (KeyError, ValueError):
                continue
            eff_str = r.get("eff_res_mm", "")
            try:
                eff = float(eff_str) if eff_str else float("nan")
            except ValueError:
                eff = float("nan")
            rows.append({
                "source_id": r.get("source_id", ""),
                "gt_source": r.get("gt_source", ""),
                "method": method,
                "step_size": step,
                "eff_res_mm": eff,
                "structure": r.get("structure", ""),
                "dice": dice,
            })
    if not rows:
        raise SystemExit(
            f"{p}: no rows with method=={method!r}. "
            f"Did `compare_native.py` write this method's rows?"
        )
    return rows


def _assign_bucket(eff_res: float,
                   edges: List[float]) -> Tuple[int, str]:
    if math.isnan(eff_res):
        return -1, "unknown"
    for i, ub in enumerate(edges):
        if eff_res <= ub + 1e-6:
            lo = 0.0 if i == 0 else edges[i - 1]
            return i, f"({lo:.1f}, {ub:.1f}]"
    return len(edges), f"({edges[-1]:.1f}, inf]"


def _bucket_sort_key(label: str) -> float:
    if label == "unknown":
        return 1e9
    try:
        return float(label.split(",")[0].lstrip("("))
    except ValueError:
        return 1e9


def _aggregate(
    rows: List[Dict],
    edges: List[float],
) -> Tuple[
    List[str],
    Dict[str, Dict[str, List[float]]],
    Dict[str, List[float]],
    Dict[str, Dict[int, List[float]]],
]:
    """Group rows by eff_res bucket.

    Returns
    -------
    bucket_order        : ordered list of bucket labels (low eff_res first,
                          'unknown' last)
    bucket_struct       : bucket -> structure -> [dice]
    bucket_eff          : bucket -> [eff_res_mm samples], one per (source, step)
    bucket_step_perCase : bucket -> step_size -> [per-case mean Dice]
                          (taken from the ``structure=='mean'`` rows so
                          each entry is one (source, step) observation)
    """
    bucket_struct: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    bucket_eff: Dict[str, List[float]] = defaultdict(list)
    bucket_step_perCase: Dict[str, Dict[int, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for r in rows:
        _, label = _assign_bucket(r["eff_res_mm"], edges)
        bucket_struct[label][r["structure"]].append(r["dice"])
        if r["structure"] == "mean":
            bucket_eff[label].append(r["eff_res_mm"])
            bucket_step_perCase[label][r["step_size"]].append(r["dice"])

    bucket_order = list(bucket_struct.keys())
    bucket_order.sort(key=_bucket_sort_key)
    return bucket_order, bucket_struct, bucket_eff, bucket_step_perCase


def _write_per_source_csv(rows: List[Dict], out_path: Path) -> None:
    fieldnames = ["source_id", "gt_source", "method", "step_size",
                  "eff_res_mm", "structure", "dice"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({
                "source_id": r["source_id"],
                "gt_source": r.get("gt_source", ""),
                "method": r["method"],
                "step_size": r["step_size"],
                "eff_res_mm": (
                    "" if math.isnan(r["eff_res_mm"])
                    else f"{r['eff_res_mm']:.4f}"
                ),
                "structure": r["structure"],
                "dice": f"{r['dice']:.6f}",
            })


def _write_summary_csv(
    bucket_order: List[str],
    bucket_struct: Dict[str, Dict[str, List[float]]],
    bucket_eff: Dict[str, List[float]],
    out_path: Path,
) -> None:
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["eff_res_bucket", "structure", "mean_dice", "std_dice",
                    "n_observations", "eff_res_mean_mm", "eff_res_std_mm"])
        for label in bucket_order:
            effs = bucket_eff.get(label, [])
            if effs:
                eff_mean: float = float(np.mean(effs))
                eff_std: float = float(np.std(effs))
            else:
                eff_mean = float("nan")
                eff_std = float("nan")
            for s in STRUCT_ORDER + ["mean"]:
                vals = bucket_struct[label].get(s, [])
                eff_mean_s = ("" if math.isnan(eff_mean)
                              else f"{eff_mean:.3f}")
                eff_std_s = ("" if math.isnan(eff_std)
                             else f"{eff_std:.3f}")
                if not vals:
                    w.writerow([label, s, "", "", 0, eff_mean_s, eff_std_s])
                    continue
                arr = np.asarray(vals)
                w.writerow([label, s,
                            f"{arr.mean():.4f}", f"{arr.std():.4f}",
                            len(arr), eff_mean_s, eff_std_s])


def _write_summary_txt(
    method: str,
    bucket_order: List[str],
    bucket_struct: Dict[str, Dict[str, List[float]]],
    bucket_eff: Dict[str, List[float]],
    out_path: Path,
) -> None:
    col_w = 22
    with open(out_path, "w") as f:
        f.write("=" * 78 + "\n")
        f.write(f"{method} per-source Dice by eff_res bucket (native space)\n")
        f.write("=" * 78 + "\n\n")
        f.write(f"Source: paired_per_source.csv "
                f"(rows where method=={method!r}).\n")
        f.write("Dice computed against native-head GT; no GT is ever "
                "resampled.\n\n")

        f.write("eff_res mean per bucket (mm):\n")
        for label in bucket_order:
            effs = bucket_eff.get(label, [])
            if effs:
                f.write(f"  {label:<25} {np.mean(effs):.3f} "
                        f"+/- {np.std(effs):.3f} (n={len(effs)})\n")
        f.write("\n")

        f.write("Mean Dice by eff_res bucket "
                "(n_observations in parentheses)\n")
        f.write("-" * 78 + "\n")
        header = "structure".ljust(11) + "".join(
            c.ljust(col_w) for c in bucket_order
        )
        f.write(header + "\n")
        for s in STRUCT_ORDER + ["mean"]:
            row = s.ljust(11)
            for label in bucket_order:
                vals = bucket_struct[label].get(s, [])
                if not vals:
                    row += "n/a".ljust(col_w)
                else:
                    arr = np.asarray(vals)
                    cell = f"{arr.mean():.3f}+/-{arr.std():.3f}(n={len(arr)})"
                    row += cell.ljust(col_w)
            f.write(row + "\n")
        f.write("\n")


def _plot_summary_png(
    method: str,
    bucket_order: List[str],
    bucket_struct: Dict[str, Dict[str, List[float]]],
    bucket_eff: Dict[str, List[float]],
    bucket_step_perCase: Dict[str, Dict[int, List[float]]],
    out_path: Path,
) -> None:
    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(3, 1, hspace=0.5)

    # ── (1) Overall mean Dice vs eff_res ─────────────────────────
    ax0 = fig.add_subplot(gs[0])
    x_eff: List[float] = []
    y_mean: List[float] = []
    y_std: List[float] = []
    n_labels: List[str] = []
    for label in bucket_order:
        effs = bucket_eff.get(label, [])
        vals = bucket_struct[label].get("mean", [])
        if not effs or not vals:
            continue
        x_eff.append(float(np.mean(effs)))
        arr = np.asarray(vals)
        y_mean.append(float(arr.mean()))
        y_std.append(float(arr.std()))
        n_labels.append(f"n={len(arr)}")
    if x_eff:
        order = np.argsort(x_eff)
        xs = [x_eff[i] for i in order]
        ys = [y_mean[i] for i in order]
        es = [y_std[i] for i in order]
        ns = [n_labels[i] for i in order]
        ax0.errorbar(xs, ys, yerr=es, fmt="o-", capsize=4, color="#444")
        for x, y, lab in zip(xs, ys, ns):
            ax0.annotate(f"{y:.3f}\n{lab}", (x, y),
                         textcoords="offset points", xytext=(0, 10),
                         ha="center", fontsize=8, color="#444")
    ax0.set_xlabel("effective resolution (mm, through-plane)")
    ax0.set_ylabel("mean Dice (4 foreground classes)")
    ax0.set_title(f"{method}: overall Dice vs effective resolution")
    ax0.set_ylim(0, 1)
    ax0.grid(True, alpha=0.3)

    # ── (2) Per-class Dice vs eff_res ─────────────────────────────
    ax1 = fig.add_subplot(gs[1])
    for c in STRUCT_ORDER:
        xs_c: List[float] = []
        ys_c: List[float] = []
        es_c: List[float] = []
        for label in bucket_order:
            effs = bucket_eff.get(label, [])
            vals = bucket_struct[label].get(c, [])
            if not effs or not vals:
                continue
            arr = np.asarray(vals)
            xs_c.append(float(np.mean(effs)))
            ys_c.append(float(arr.mean()))
            es_c.append(float(arr.std()))
        if not xs_c:
            continue
        order = np.argsort(xs_c)
        xs_c = [xs_c[i] for i in order]
        ys_c = [ys_c[i] for i in order]
        es_c = [es_c[i] for i in order]
        ax1.errorbar(xs_c, ys_c, yerr=es_c, fmt="o-", capsize=3,
                     color=CLASS_COLORS[c], label=c)
    ax1.set_xlabel("effective resolution (mm, through-plane)")
    ax1.set_ylabel("Dice")
    ax1.set_title(f"{method}: per-class Dice vs effective resolution")
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower left", fontsize=8, ncol=2)

    # ── (3) Per-case Dice distribution per bucket ─────────────────
    ax2 = fig.add_subplot(gs[2])
    box_data: List[List[float]] = []
    box_pos: List[int] = []
    box_labels: List[str] = []
    for i, label in enumerate(bucket_order):
        all_step_vals: List[float] = []
        for s_vals in bucket_step_perCase.get(label, {}).values():
            all_step_vals.extend(s_vals)
        if not all_step_vals:
            continue
        box_data.append(all_step_vals)
        box_pos.append(i)
        box_labels.append(label)
    if box_data:
        bp = ax2.boxplot(box_data, positions=box_pos, widths=0.6,
                         patch_artist=True, showfliers=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#a6cee3")
            patch.set_alpha(0.7)
        for pos, vals in zip(box_pos, box_data):
            ax2.scatter([pos] * len(vals), vals, s=8, color="#1f3a5f",
                        alpha=0.35, zorder=3)
            ax2.annotate(f"n={len(vals)}", (pos, max(vals)),
                         textcoords="offset points", xytext=(0, 6),
                         ha="center", fontsize=8, color="gray")
        ax2.set_xticks(box_pos)
        ax2.set_xticklabels(box_labels, rotation=20, fontsize=8)
    ax2.set_ylabel("per-case Dice (mean over 4 foreground classes)")
    ax2.set_title(f"{method}: per-case Dice distribution by eff_res bucket")
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"{method} reconstruction summary  "
        f"(driven by paired_per_source.csv)",
        fontsize=13, fontweight="bold", y=0.995,
    )
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="nnunet/configs.yaml")
    ap.add_argument("--method", required=True,
                    help="Method label as written into paired_per_source"
                         "__*.csv (e.g. nnUNet-sparse, CNISP-atlasGT, "
                         "CNISP-nnUNetPred).")
    ap.add_argument("--paired-csv", required=True,
                    help="Path to the paired CSV for this CNISP run "
                         "(e.g. ${work_dir}/comparison/paired_per_source"
                         "__atlas_gt.csv).")
    ap.add_argument(
        "--out-dir", required=True,
        help="Where to write outputs. Pipeline conventions: "
             "${work_dir}/comparison/viz/<method>__<run_tag>/ for nnUNet "
             "rows, ${cnisp_output_basedir}/<model>/viz/<run_tag>/ for CNISP rows.",
    )
    args = ap.parse_args()

    cfg = _load_yaml(Path(args.config))
    paired_csv = Path(args.paired_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bucket_edges = list(cfg.get(
        "summary_bucket_edges_mm",
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.5, 8.5, 11.0, 13.0],
    ))

    rows = _read_paired_csv(paired_csv, args.method)
    bucket_order, bucket_struct, bucket_eff, bucket_step = _aggregate(
        rows, bucket_edges,
    )

    per_src = out_dir / f"{args.method}_per_source.csv"
    summary_csv = out_dir / f"{args.method}_summary_by_eff_res.csv"
    summary_txt = out_dir / f"{args.method}_summary_by_eff_res.txt"
    summary_png = out_dir / f"{args.method}_recon_summary.png"

    _write_per_source_csv(rows, per_src)
    _write_summary_csv(bucket_order, bucket_struct, bucket_eff, summary_csv)
    _write_summary_txt(args.method, bucket_order, bucket_struct,
                       bucket_eff, summary_txt)
    _plot_summary_png(args.method, bucket_order, bucket_struct,
                      bucket_eff, bucket_step, summary_png)

    print(f"[build_method_summary] {args.method}: {len(rows)} long rows -> "
          f"{out_dir}/")
    for p in (per_src, summary_csv, summary_txt, summary_png):
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
