#!/usr/bin/env python3
"""Plotting + table-writing primitives for the native-space Dice summaries.

The *rendering/aggregation* layer shared by the four summary drivers under
``engine/`` -- ``build_method_summary`` (per-method bundle),
``build_paired_summary`` (head-to-head overlay), ``build_nnunet_native_summary``
(nnUNet-only by step/eff_res), and ``build_experiment_summary`` (cross
thin/thick/real). Those drivers now only read their CSV, call the matching
``aggregate_*`` here, and hand the result to the ``write_*`` / ``draw_*`` /
``plot_*`` functions; all matplotlib + CSV/TXT formatting lives here.

Function names are prefixed by their consumer family (``method_*`` / ``paired_*``
/ ``native_*`` / ``experiment_*``) so the four sets coexist without collision.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from nnunet.helpers.buckets import (  # noqa: E402
    NNUNET_METHOD_LABEL,
    STRUCT_ORDER,
    assign_bucket,
    bucket_sort_key,
)

# Structure columns in display order, including the 4-class mean last.
COLS: List[str] = STRUCT_ORDER + ["mean"]

# Per-class colours shared by the per-method and nnUNet-only plots.
CLASS_COLORS = {
    "ON": "#d62728",
    "Globe": "#1f77b4",
    "Fat": "#2ca02c",
    "Recti": "#9467bd",
}

# Per-method colours kept consistent across all paired panels so the legend
# never has to be re-keyed when scanning between subplots.
METHOD_COLORS = {
    "nnUNet-sparse":   "#d62728",   # red    - image-conditioned baseline
    "CNISP-atlasGT":   "#1f77b4",   # blue   - GT-conditioned ceiling curve
    "CNISP-nnUNetPred": "#2ca02c",  # green  - deployment-mode CNISP
    "CNISP-v6.5-gt-atlasGT": "#1f77b4",  # blue  - v6.5-gt ceiling curve
    "CNISP-v6.5-gt":   "#2ca02c",   # green  - v6.5-gt corrector_gt run
    "nnUNet-C":        "#ff7f0e",   # orange - CNISP-prelabel corrector
    "nnUNet-C (C)":    "#ff7f0e",   # orange - control C: CNISP-prelabel corrector
    "nnUNet-C (B)":    "#9467bd",   # purple - control B: nnUNet-prelabel corrector (stacked)
}
DEFAULT_CNISP_COLOR = "#1f77b4"
DEFAULT_NNUNET_COLOR = "#d62728"

# Deterministic fallback palette for method labels not in METHOD_COLORS, so a
# multi-method overlay (e.g. two CNISP run_tags on one figure) still gets
# distinct colors instead of all collapsing onto one default.
_FALLBACK_PALETTE = [
    "#1f77b4", "#2ca02c", "#9467bd", "#8c564b", "#17becf", "#bcbd22",
]


def color_for(method: str, fallback: str) -> str:
    return METHOD_COLORS.get(method, fallback)


def method_color(method: str, idx: int) -> str:
    """Stable per-method color: the registered one, else a palette slot."""
    if method in METHOD_COLORS:
        return METHOD_COLORS[method]
    return _FALLBACK_PALETTE[idx % len(_FALLBACK_PALETTE)]


def fmt(v: float, nd: int = 6) -> str:
    return "" if (v is None or math.isnan(v)) else f"{v:.{nd}f}"


def save_standalone(out_path: Path, figsize, draw_fn) -> None:
    """Render a single panel into its own PNG via ``draw_fn(ax)``."""
    fig, ax = plt.subplots(figsize=figsize)
    draw_fn(ax)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════
# build_method_summary: per-method by-eff_res bundle
# ════════════════════════════════════════════════════════════════


def aggregate_by_bucket(
    rows: List[Dict],
    edges: List[float],
) -> Tuple[
    List[str],
    Dict[str, Dict[str, List[float]]],
    Dict[str, List[float]],
    Dict[str, Dict[int, List[float]]],
]:
    """Group rows by eff_res bucket (single method).

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
        _, label = assign_bucket(r["eff_res_mm"], edges)
        bucket_struct[label][r["structure"]].append(r["dice"])
        if r["structure"] == "mean":
            bucket_eff[label].append(r["eff_res_mm"])
            bucket_step_perCase[label][r["step_size"]].append(r["dice"])

    bucket_order = list(bucket_struct.keys())
    bucket_order.sort(key=bucket_sort_key)
    return bucket_order, bucket_struct, bucket_eff, bucket_step_perCase


def write_method_per_source_csv(rows: List[Dict], out_path: Path) -> None:
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


def write_method_summary_csv(
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


def write_method_summary_txt(
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


def draw_overall_dice(
    ax,
    method: str,
    bucket_order: List[str],
    bucket_struct: Dict[str, Dict[str, List[float]]],
    bucket_eff: Dict[str, List[float]],
) -> None:
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
        ax.errorbar(xs, ys, yerr=es, fmt="o-", capsize=4, color="#444")
        for x, y, lab in zip(xs, ys, ns):
            ax.annotate(f"{y:.3f}\n{lab}", (x, y),
                        textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=8, color="#444")
    ax.set_xlabel("effective resolution (mm, through-plane)")
    ax.set_ylabel("mean Dice (4 foreground classes)")
    ax.set_title(f"{method}: overall Dice vs effective resolution")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)


def draw_per_class_dice(
    ax,
    method: str,
    bucket_order: List[str],
    bucket_struct: Dict[str, Dict[str, List[float]]],
    bucket_eff: Dict[str, List[float]],
) -> None:
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
        ax.errorbar(xs_c, ys_c, yerr=es_c, fmt="o-", capsize=3,
                    color=CLASS_COLORS[c], label=c)
    ax.set_xlabel("effective resolution (mm, through-plane)")
    ax.set_ylabel("Dice")
    ax.set_title(f"{method}: per-class Dice vs effective resolution")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=8, ncol=2)


def draw_per_case_distribution(
    ax,
    method: str,
    bucket_order: List[str],
    bucket_step_perCase: Dict[str, Dict[int, List[float]]],
) -> None:
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
        bp = ax.boxplot(box_data, positions=box_pos, widths=0.6,
                        patch_artist=True, showfliers=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#a6cee3")
            patch.set_alpha(0.7)
        for pos, vals in zip(box_pos, box_data):
            ax.scatter([pos] * len(vals), vals, s=8, color="#1f3a5f",
                       alpha=0.35, zorder=3)
            ax.annotate(f"n={len(vals)}", (pos, max(vals)),
                        textcoords="offset points", xytext=(0, 6),
                        ha="center", fontsize=8, color="gray")
        ax.set_xticks(box_pos)
        ax.set_xticklabels(box_labels, rotation=20, fontsize=8)
    ax.set_ylabel("per-case Dice (mean over 4 foreground classes)")
    ax.set_title(f"{method}: per-case Dice distribution by eff_res bucket")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)


def plot_method_summary(
    method: str,
    bucket_order: List[str],
    bucket_struct: Dict[str, Dict[str, List[float]]],
    bucket_eff: Dict[str, List[float]],
    bucket_step_perCase: Dict[str, Dict[int, List[float]]],
    out_path: Path,
    standalone_paths: Optional[Dict[str, Path]] = None,
) -> None:
    """Render the combined 3-subplot PNG plus optional stand-alone copies.

    ``standalone_paths`` (when given) is a mapping from panel name
    (``"overall" | "per_class" | "per_case"``) to the file path each
    individual panel should be written to. The combined PNG is always
    written to ``out_path`` regardless.
    """
    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(3, 1, hspace=0.5)

    ax0 = fig.add_subplot(gs[0])
    draw_overall_dice(ax0, method, bucket_order, bucket_struct, bucket_eff)

    ax1 = fig.add_subplot(gs[1])
    draw_per_class_dice(ax1, method, bucket_order, bucket_struct, bucket_eff)

    ax2 = fig.add_subplot(gs[2])
    draw_per_case_distribution(ax2, method, bucket_order, bucket_step_perCase)

    # Neutral wording on purpose. CNISP IS a reconstruction model, but
    # this same driver also renders nnUNet-sparse panels (which are
    # image-conditioned segmentation, NOT reconstruction). Saying
    # "Dice vs effective resolution" is true for both and keeps the
    # per-method title symmetric with the paired comparison plot.
    fig.suptitle(
        f"{method}: Dice vs effective resolution  "
        f"(driven by paired_per_source.csv)",
        fontsize=13, fontweight="bold", y=0.995,
    )
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    if standalone_paths:
        if "overall" in standalone_paths:
            save_standalone(
                standalone_paths["overall"], (10, 5),
                lambda ax: draw_overall_dice(
                    ax, method, bucket_order, bucket_struct, bucket_eff,
                ),
            )
        if "per_class" in standalone_paths:
            save_standalone(
                standalone_paths["per_class"], (10, 5),
                lambda ax: draw_per_class_dice(
                    ax, method, bucket_order, bucket_struct, bucket_eff,
                ),
            )
        if "per_case" in standalone_paths:
            save_standalone(
                standalone_paths["per_case"], (12, 5),
                lambda ax: draw_per_case_distribution(
                    ax, method, bucket_order, bucket_step_perCase,
                ),
            )


# ════════════════════════════════════════════════════════════════
# build_paired_summary: head-to-head overlay (nnUNet vs one CNISP run)
# ════════════════════════════════════════════════════════════════


def aggregate_paired(
    rows: List[Dict],
    edges: List[float],
) -> Tuple[
    List[str],
    Dict[Tuple[str, str], Dict[str, List[float]]],
    Dict[Tuple[str, str], List[float]],
]:
    """Group rows by (method, eff_res bucket).

    Returns
    -------
    bucket_order   : ordered bucket labels (lowest eff_res first, 'unknown' last)
    by_method_bucket : (method, bucket) -> structure -> [dice]
    eff_by_bucket  : (method, bucket) -> [eff_res_mm samples]
    """
    by_method_bucket: Dict[Tuple[str, str], Dict[str, List[float]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    eff_by_bucket: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    seen_buckets: List[str] = []
    seen_set = set()
    for r in rows:
        _, label = assign_bucket(r["eff_res_mm"], edges)
        if label not in seen_set:
            seen_set.add(label)
            seen_buckets.append(label)
        by_method_bucket[(r["method"], label)][r["structure"]].append(r["dice"])
        if r["structure"] == "mean":
            eff_by_bucket[(r["method"], label)].append(r["eff_res_mm"])
    seen_buckets.sort(key=bucket_sort_key)
    return seen_buckets, by_method_bucket, eff_by_bucket


def series_for(
    method: str,
    structure: str,
    bucket_order: List[str],
    by_method_bucket: Dict[Tuple[str, str], Dict[str, List[float]]],
    eff_by_bucket: Dict[Tuple[str, str], List[float]],
) -> Tuple[List[float], List[float], List[float], List[int]]:
    """Per-bucket (x=mean eff_res, y=mean dice, e=std, n) for one method+struct."""
    xs: List[float] = []
    ys: List[float] = []
    es: List[float] = []
    ns: List[int] = []
    for label in bucket_order:
        vals = by_method_bucket.get((method, label), {}).get(structure, [])
        effs = eff_by_bucket.get((method, label), [])
        if not vals or not effs:
            continue
        arr = np.asarray(vals)
        xs.append(float(np.mean(effs)))
        ys.append(float(arr.mean()))
        es.append(float(arr.std()))
        ns.append(int(len(arr)))
    if xs:
        order = np.argsort(xs)
        xs = [xs[i] for i in order]
        ys = [ys[i] for i in order]
        es = [es[i] for i in order]
        ns = [ns[i] for i in order]
    return xs, ys, es, ns


def draw_paired_overall(
    ax,
    methods: List[str],
    bucket_order: List[str],
    by_method_bucket,
    eff_by_bucket,
    label_map=None,
) -> None:
    # label_map: DISPLAY-ONLY method -> legend label remap (e.g. internal
    # "nnUNet-C (C)" -> "D (Proposed)"). Data keys and colors stay on the
    # internal method string; only the legend text changes. Default = identity.
    lm = label_map or {}
    for idx, m in enumerate(methods):
        xs, ys, es, ns = series_for(
            m, "mean", bucket_order, by_method_bucket, eff_by_bucket,
        )
        if not xs:
            continue
        c = method_color(m, idx)
        ax.errorbar(xs, ys, yerr=es, fmt="o-", capsize=4, color=c, label=lm.get(m, m))
        # Stagger the per-point labels by method index so the (up to 4)
        # near-coincident markers at each eff_res bucket don't pile their
        # annotations on top of each other: alternate above/below the marker
        # with growing offset; the per-method color keeps them readable.
        sign = 1 if (idx % 2 == 0) else -1
        dy = sign * (11 + (idx // 2) * 22)
        va = "bottom" if sign > 0 else "top"
        for x, y, n in zip(xs, ys, ns):
            ax.annotate(f"{y:.3f}\nn={n}", (x, y),
                        textcoords="offset points", xytext=(0, dy),
                        ha="center", va=va, fontsize=7, color=c)
    ax.set_xlabel("effective resolution (mm, through-plane)")
    ax.set_ylabel("mean Dice (4 foreground classes)")
    ax.set_title("Overall mean Dice vs effective resolution")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)


def draw_paired_per_class(
    axes,
    methods: List[str],
    bucket_order: List[str],
    by_method_bucket,
    eff_by_bucket,
    label_map=None,
) -> None:
    """Fill a 2x2 grid of axes (one per foreground class) with paired curves.

    ``label_map`` is a DISPLAY-ONLY legend remap (data keys/colors unchanged).
    """
    lm = label_map or {}
    for i, c in enumerate(STRUCT_ORDER):
        ax = axes[i // 2][i % 2]
        for idx, m in enumerate(methods):
            xs, ys, es, _ = series_for(
                m, c, bucket_order, by_method_bucket, eff_by_bucket,
            )
            if not xs:
                continue
            ax.errorbar(
                xs, ys, yerr=es, fmt="o-", capsize=3,
                color=method_color(m, idx), label=lm.get(m, m),
            )
        ax.set_title(c)
        ax.set_xlabel("effective resolution (mm)")
        ax.set_ylabel("Dice")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left", fontsize=8)


def bucket_means(
    method: str,
    bucket_order: List[str],
    by_method_bucket,
    eff_by_bucket,
) -> Dict[str, Tuple[float, float, int]]:
    """bucket_label -> (mean Dice, mean eff_res, n_observations) for the mean row."""
    out: Dict[str, Tuple[float, float, int]] = {}
    for label in bucket_order:
        vals = by_method_bucket.get((method, label), {}).get("mean", [])
        effs = eff_by_bucket.get((method, label), [])
        if not vals or not effs:
            continue
        out[label] = (
            float(np.mean(vals)), float(np.mean(effs)), int(len(vals)),
        )
    return out


def draw_delta(
    ax,
    cnisp_method: str,
    bucket_order: List[str],
    by_method_bucket,
    eff_by_bucket,
    label_map=None,
) -> None:
    """Bar chart of (CNISP - nnUNet) on the mean Dice row, per shared bucket.

    Only buckets where BOTH methods have at least one observation are
    plotted; otherwise the bar would be meaningless. eff_res mean of the
    CNISP rows is used for the x-axis (CNISP is the bucket reference
    because all eff_res values come from CNISP's sweep_results.pkl).

    ``label_map`` remaps ONLY the title/axis text (data keys unchanged), so the
    baseline lookup below still keys on the internal ``NNUNET_METHOD_LABEL``.
    """
    lm = label_map or {}
    cn_disp = lm.get(cnisp_method, cnisp_method)
    nn_disp = lm.get(NNUNET_METHOD_LABEL, NNUNET_METHOD_LABEL)
    nn = bucket_means(NNUNET_METHOD_LABEL, bucket_order,
                      by_method_bucket, eff_by_bucket)
    cn = bucket_means(cnisp_method, bucket_order,
                      by_method_bucket, eff_by_bucket)
    shared = [b for b in bucket_order if b in nn and b in cn]
    if not shared:
        ax.set_title(f"{cn_disp} - {nn_disp}  (no shared buckets)")
        ax.axis("off")
        return

    xs: List[float] = []
    deltas: List[float] = []
    labels: List[str] = []
    for b in shared:
        cn_mean, _, _ = cn[b]
        nn_mean, _, _ = nn[b]
        _, eff, _ = cn[b]
        xs.append(eff)
        deltas.append(cn_mean - nn_mean)
        labels.append(b)
    order = np.argsort(xs)
    xs = [xs[i] for i in order]
    deltas = [deltas[i] for i in order]
    labels = [labels[i] for i in order]

    colors = [
        color_for(cnisp_method, DEFAULT_CNISP_COLOR) if d >= 0
        else color_for(NNUNET_METHOD_LABEL, DEFAULT_NNUNET_COLOR)
        for d in deltas
    ]
    ax.bar(xs, deltas, width=0.55, color=colors, alpha=0.85,
           edgecolor="#444", linewidth=0.6)
    ax.axhline(0, color="#444", linewidth=0.8)
    for x, d, lab in zip(xs, deltas, labels):
        ax.annotate(
            f"{d:+.3f}\n{lab}",
            (x, d), textcoords="offset points",
            xytext=(0, 4 if d >= 0 else -14),
            ha="center", fontsize=7, color="#222",
        )
    ax.set_xlabel("effective resolution (mm, CNISP bucket mean)")
    ax.set_ylabel(f"Dice delta  ({cn_disp} - {nn_disp})")
    ax.set_title(
        f"Head-to-head: {cn_disp} - {nn_disp}  "
        f"(positive => {cn_disp} wins)"
    )
    ax.grid(True, axis="y", alpha=0.3)


def plot_paired(
    cnisp_method: str,
    bucket_order: List[str],
    by_method_bucket,
    eff_by_bucket,
    out_dir: Path,
    extra_methods: Optional[List[str]] = None,
) -> Dict[str, Path]:
    # nnUNet-sparse + the CNISP run + any extra methods (e.g. nnUNet-C) are
    # overlaid on the overall + per-class panels. The delta panel stays the
    # head-to-head (CNISP - nnUNet-sparse) so its semantics are unchanged.
    methods = [NNUNET_METHOD_LABEL, cnisp_method] + list(extra_methods or [])

    # Stand-alone: overall
    overall_path = out_dir / "paired_overall_dice_vs_eff_res.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    draw_paired_overall(ax, methods, bucket_order, by_method_bucket, eff_by_bucket)
    fig.tight_layout()
    fig.savefig(str(overall_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Stand-alone: per-class 2x2 grid
    per_class_path = out_dir / "paired_per_class_dice_vs_eff_res.png"
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    draw_paired_per_class(axes, methods, bucket_order, by_method_bucket, eff_by_bucket)
    fig.suptitle("Per-class Dice vs effective resolution", fontsize=12,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(str(per_class_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Stand-alone: delta
    delta_path = out_dir / "paired_delta_dice_vs_eff_res.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    draw_delta(ax, cnisp_method, bucket_order, by_method_bucket, eff_by_bucket)
    fig.tight_layout()
    fig.savefig(str(delta_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Combined three-row figure
    #   row 0          -> overall mean Dice (full width)
    #   row 1 (2x2 sub) -> per-class panels
    #   row 2          -> delta bar chart (full width)
    # Sizing keeps each panel close to the aspect of its standalone sibling;
    # the suptitle y is tuned so bbox_inches="tight" doesn't leave a tall band.
    combined_path = out_dir / "paired_dice_vs_eff_res.png"
    fig = plt.figure(figsize=(11, 18))
    gs = fig.add_gridspec(3, 1, hspace=0.35, height_ratios=[1, 1.9, 1])

    ax0 = fig.add_subplot(gs[0])
    draw_paired_overall(ax0, methods, bucket_order, by_method_bucket, eff_by_bucket)

    inner_gs = gs[1].subgridspec(2, 2, hspace=0.5, wspace=0.25)
    inner_axes = [
        [fig.add_subplot(inner_gs[0, 0]), fig.add_subplot(inner_gs[0, 1])],
        [fig.add_subplot(inner_gs[1, 0]), fig.add_subplot(inner_gs[1, 1])],
    ]
    draw_paired_per_class(inner_axes, methods, bucket_order,
                          by_method_bucket, eff_by_bucket)

    ax2 = fig.add_subplot(gs[2])
    draw_delta(ax2, cnisp_method, bucket_order,
               by_method_bucket, eff_by_bucket)

    fig.suptitle(
        f"{NNUNET_METHOD_LABEL} vs {cnisp_method}: Dice vs effective resolution  "
        f"(driven by paired_per_source.csv)",
        fontsize=13, fontweight="bold", y=0.92,
    )
    fig.savefig(str(combined_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "overall": overall_path,
        "per_class": per_class_path,
        "delta": delta_path,
        "combined": combined_path,
    }


def write_paired_csv(
    cnisp_method: str,
    bucket_order: List[str],
    by_method_bucket,
    eff_by_bucket,
    out_path: Path,
) -> None:
    """Bucket-by-structure table with both methods' mean/std/n + delta on the mean row."""
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "eff_res_bucket", "structure",
            f"{NNUNET_METHOD_LABEL}_mean_dice",
            f"{NNUNET_METHOD_LABEL}_std_dice",
            f"{NNUNET_METHOD_LABEL}_n",
            f"{cnisp_method}_mean_dice",
            f"{cnisp_method}_std_dice",
            f"{cnisp_method}_n",
            "delta_mean_dice",  # CNISP - nnUNet on this (bucket, structure)
            "eff_res_mean_mm",  # CNISP bucket-mean eff_res (mirrors plots)
        ])
        for label in bucket_order:
            cn_eff = eff_by_bucket.get((cnisp_method, label), [])
            eff_mean_s = (f"{float(np.mean(cn_eff)):.3f}"
                          if cn_eff else "")
            for s in STRUCT_ORDER + ["mean"]:
                nn_vals = by_method_bucket.get(
                    (NNUNET_METHOD_LABEL, label), {}).get(s, [])
                cn_vals = by_method_bucket.get(
                    (cnisp_method, label), {}).get(s, [])
                nn_arr = np.asarray(nn_vals) if nn_vals else None
                cn_arr = np.asarray(cn_vals) if cn_vals else None
                delta = ""
                if nn_arr is not None and cn_arr is not None:
                    delta = f"{cn_arr.mean() - nn_arr.mean():+.4f}"
                w.writerow([
                    label, s,
                    "" if nn_arr is None else f"{nn_arr.mean():.4f}",
                    "" if nn_arr is None else f"{nn_arr.std():.4f}",
                    0 if nn_arr is None else int(len(nn_arr)),
                    "" if cn_arr is None else f"{cn_arr.mean():.4f}",
                    "" if cn_arr is None else f"{cn_arr.std():.4f}",
                    0 if cn_arr is None else int(len(cn_arr)),
                    delta,
                    eff_mean_s,
                ])


# ════════════════════════════════════════════════════════════════
# build_nnunet_native_summary: nnUNet-only by step / eff_res
# ════════════════════════════════════════════════════════════════


def aggregate_native_by_step(wide_rows: List[Dict]) -> List[Dict]:
    """Aggregate wide per-(source, step) rows by ``step_size``."""
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


def aggregate_native_by_eff_res(
    wide_rows: List[Dict], edges: List[float],
) -> List[Dict]:
    """Aggregate wide per-(source, step) rows into eff_res buckets.

    Same bucket edges as ``build_method_summary`` / the CNISP plots, so the
    nnUNet eff_res figure lines up point-for-point with CNISP's.
    """
    by_bucket: Dict[str, List[Dict]] = defaultdict(list)
    bucket_eff: Dict[str, List[float]] = defaultdict(list)
    for r in wide_rows:
        eff = r["eff_res_mm"]
        if math.isnan(eff):
            label = "unknown"
        else:
            _, label = assign_bucket(eff, edges)
            bucket_eff[label].append(eff)
        by_bucket[label].append(r)

    labels = sorted(by_bucket.keys(), key=bucket_sort_key)
    out: List[Dict] = []
    for label in labels:
        group = by_bucket[label]
        effs = bucket_eff.get(label, [])
        agg: Dict = {
            "eff_res_bucket": label,
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


def write_native_per_source_csv(wide_rows: List[Dict], out_path: Path) -> None:
    fieldnames = ["source_id", "gt_source", "step_size", "eff_res_mm"] + COLS
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fieldnames)
        for r in wide_rows:
            w.writerow([
                r["source_id"], r["gt_source"], r["step_size"],
                fmt(r["eff_res_mm"], 4),
                *[fmt(r[c]) for c in COLS],
            ])


def write_native_by_step_csv(step_rows: List[Dict], out_path: Path) -> None:
    fieldnames = ["step_size", "n_sources", "eff_res_mm"]
    for c in COLS:
        fieldnames += [f"{c}_mean", f"{c}_std"]
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fieldnames)
        for r in step_rows:
            row = [r["step_size"], r["n_sources"], fmt(r["eff_res_mm"], 4)]
            for c in COLS:
                row += [fmt(r[f"{c}_mean"], 4), fmt(r[f"{c}_std"], 4)]
            w.writerow(row)


def write_native_by_eff_res_csv(bucket_rows: List[Dict], out_path: Path) -> None:
    fieldnames = ["eff_res_bucket", "n_sources", "eff_res_mm"]
    for c in COLS:
        fieldnames += [f"{c}_mean", f"{c}_std"]
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fieldnames)
        for r in bucket_rows:
            row = [r["eff_res_bucket"], r["n_sources"], fmt(r["eff_res_mm"], 4)]
            for c in COLS:
                row += [fmt(r[f"{c}_mean"], 4), fmt(r[f"{c}_std"], 4)]
            w.writerow(row)


def plot_native_dice_vs_eff_res(
    bucket_rows: List[Dict], method: str, out_path: Path,
) -> None:
    """Overall + per-class Dice vs effective resolution (matches CNISP's axis)."""
    pts = [r for r in bucket_rows if not math.isnan(r["eff_res_mm"])]
    pts.sort(key=lambda r: r["eff_res_mm"])
    xs = [r["eff_res_mm"] for r in pts]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5))

    ys = [r["mean_mean"] for r in pts]
    es = [r["mean_std"] for r in pts]
    ax0.errorbar(xs, ys, yerr=es, fmt="o-", capsize=4, color="#444")
    for r in pts:
        if not math.isnan(r["mean_mean"]):
            ax0.annotate(f"{r['mean_mean']:.3f}\nn={r['n_sources']}",
                         (r["eff_res_mm"], r["mean_mean"]),
                         textcoords="offset points", xytext=(0, 10),
                         ha="center", fontsize=8, color="#444")
    ax0.set_xlabel("effective resolution (mm, through-plane)")
    ax0.set_ylabel("mean Dice (4 foreground classes)")
    ax0.set_title(f"{method}: overall native Dice vs eff_res")
    ax0.set_ylim(0, 1)
    ax0.grid(True, alpha=0.3)

    for c in STRUCT_ORDER:
        ys_c = [r[f"{c}_mean"] for r in pts]
        es_c = [r[f"{c}_std"] for r in pts]
        ax1.errorbar(xs, ys_c, yerr=es_c, fmt="o-", capsize=3,
                     color=CLASS_COLORS[c], label=c)
    ax1.set_xlabel("effective resolution (mm, through-plane)")
    ax1.set_ylabel("Dice")
    ax1.set_title(f"{method}: per-class native Dice vs eff_res")
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower left", fontsize=8, ncol=2)

    fig.suptitle(f"{method}: native-space Dice vs effective resolution",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_native_dice_vs_step(step_rows: List[Dict], method: str, out_path: Path) -> None:
    """Two panels: overall mean Dice vs step, and per-class Dice vs step."""
    steps = [r["step_size"] for r in step_rows]
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5))

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


# ════════════════════════════════════════════════════════════════
# build_experiment_summary: cross thin/thick/real
# ════════════════════════════════════════════════════════════════

# Stable experiment order + per-experiment style. thin/thick are eff_res
# sweeps (lines); real is a single operating point (markers, no line).
EXP_ORDER = ["thin", "thick", "real"]
EXP_STYLE = {
    "thin":  {"color": "#1f77b4", "marker": "o", "line": "-"},
    "thick": {"color": "#d62728", "marker": "s", "line": "--"},
    "real":  {"color": "#2ca02c", "marker": "D", "line": ""},  # point only
}


def discover_experiments(comparison_dir: Path) -> Dict[str, List[str]]:
    """Map experiment -> sorted run_tags found on disk.

    Filenames are ``paired_per_source__<run_tag>__<exp>.csv``; run_tag
    may itself contain single underscores (atlas_gt, nnunet_pred,
    real_pair) so we split on the LAST ``__`` boundary.
    """
    found: Dict[str, List[str]] = defaultdict(list)
    for p in sorted(comparison_dir.glob("paired_per_source__*__*.csv")):
        stem = p.stem  # drop .csv
        head, _, exp = stem.rpartition("__")
        run_tag = head[len("paired_per_source__"):]
        if not run_tag or not exp:
            continue
        found[exp].append(run_tag)
    return {e: sorted(set(v)) for e, v in found.items()}


def canonical_run_tag(run_tags: List[str]) -> Optional[str]:
    """Pick the run_tag whose CSV holds the canonical nnUNet-sparse rows
    (prefer nnunet_pred -- a strict superset -- else the first present)."""
    if not run_tags:
        return None
    return "nnunet_pred" if "nnunet_pred" in run_tags else run_tags[0]


def series_curve(
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


def overall_stats(rows: List[Dict], structure: str) -> Tuple[float, float, int]:
    vals = [r["dice"] for r in rows if r["structure"] == structure]
    if not vals:
        return float("nan"), float("nan"), 0
    arr = np.asarray(vals)
    return float(arr.mean()), float(arr.std()), int(arr.size)


def draw_method_by_experiment(ax, method: str, by_exp: Dict[str, List[Dict]],
                              edges: List[float]) -> bool:
    """Overlay each experiment's curve/point for one method. Returns True
    if anything was drawn."""
    drawn = False
    for exp in EXP_ORDER:
        rows = by_exp.get(exp)
        if not rows:
            continue
        st = EXP_STYLE.get(exp, {"color": "#444", "marker": "o", "line": "-"})
        xs, ys, es, _ = series_curve(rows, edges)
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


def write_experiment_summary(data: Dict[str, Dict[str, List[Dict]]], out_csv: Path,
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
                    m, sd, n = overall_stats(rows, s)
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
                    m, sd, n = overall_stats(data[method][exp], s)
                    cell = ("n/a" if n == 0
                            else f"{m:.3f}+/-{sd:.3f}(n={n})")
                    row += cell.ljust(col_w)
                f.write(row + "\n")
            f.write("\n")
