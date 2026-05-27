#!/usr/bin/env python3
"""Paired (head-to-head) Dice comparison plots for one CNISP run.

Sibling of ``nnunet/engine/build_method_summary.py``. Where that script
emits a *per-method* by-eff_res bundle (one method's curves alone), this
one consumes the SAME ``paired_per_source__<run_tag>.csv`` and overlays
both methods on every panel so the comparison is visible at a glance.

Subplots (delta-focused layout, matches the layout the user picked in
the design checkpoint):

  1. Overall mean Dice vs effective resolution
       two lines on shared axes: nnUNet-sparse vs CNISP-<run>
  2. Per-class Dice vs effective resolution
       2x2 grid (ON / Globe / Fat / Recti); each panel overlays both
       methods so you can see whether ON drives the gap, etc.
  3. (CNISP - nnUNet) Dice delta vs eff_res bucket  (mean row)
       bar chart of the head-to-head difference within each shared
       bucket; positive bars => CNISP wins for that bucket.

Outputs (under ``--out-dir``):

* ``paired_overall_dice_vs_eff_res.png``       stand-alone panel 1
* ``paired_per_class_dice_vs_eff_res.png``     stand-alone panel 2 (2x2 grid)
* ``paired_delta_dice_vs_eff_res.png``         stand-alone panel 3
* ``paired_dice_vs_eff_res.png``               combined 3-row figure
* ``paired_summary_by_eff_res.csv``            machine-readable delta + per-
                                               method mean/std/n in each
                                               bucket (one row per
                                               (bucket, structure)).

The titles deliberately avoid the word "reconstruction": CNISP IS a
reconstruction model, but nnUNet-sparse is image-conditioned
segmentation, so we say "Dice vs effective resolution" -- a description
that's accurate for both.

Usage
-----
    python nnunet/engine/build_paired_summary.py \\
        --config nnunet/configs.yaml \\
        --paired-csv work_dir/comparison/paired_per_source__atlas_gt.csv \\
        --cnisp-method CNISP-atlasGT \\
        --out-dir    work_dir/comparison/viz/paired__atlas_gt
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


STRUCT_ORDER = ["ON", "Globe", "Fat", "Recti"]
NNUNET_METHOD_LABEL = "nnUNet-sparse"

# Per-method colours kept consistent across all three panels so the
# legend never has to be re-keyed when scanning between subplots.
METHOD_COLORS = {
    "nnUNet-sparse":   "#d62728",   # red  – image-conditioned baseline
    "CNISP-atlasGT":   "#1f77b4",   # blue – GT-conditioned ceiling curve
    "CNISP-nnUNetPred": "#2ca02c",  # green – deployment-mode CNISP
}
DEFAULT_CNISP_COLOR = "#1f77b4"
DEFAULT_NNUNET_COLOR = "#d62728"


def _color_for(method: str, fallback: str) -> str:
    return METHOD_COLORS.get(method, fallback)


# ── Config / CSV plumbing (mirrors build_method_summary.py) ──────

def _load_yaml(p: Path) -> Dict:
    with open(p) as f:
        return yaml.safe_load(f) or {}


def _read_paired_csv(p: Path, methods: List[str]) -> List[Dict]:
    """Read rows whose ``method`` field matches one of the given methods.

    Numeric coercion mirrors ``build_method_summary._read_paired_csv``
    so the per-method plots and the paired plots stay perfectly aligned
    on the same observations.
    """
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Run `nnunet/compare_native.py` first "
            f"(or the `compare` phase of run_pipeline.sh)."
        )
    keep = set(methods)
    rows: List[Dict] = []
    with open(p) as f:
        for r in csv.DictReader(f):
            m = r.get("method")
            if m not in keep:
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
                "method": m,
                "step_size": step,
                "eff_res_mm": eff,
                "structure": r.get("structure", ""),
                "dice": dice,
            })
    if not rows:
        raise SystemExit(
            f"{p}: no rows matched methods={methods!r}. "
            f"Did `compare_native.py` write these methods' rows?"
        )
    # Sanity: warn if one of the requested methods is entirely missing.
    seen = {r["method"] for r in rows}
    missing = [m for m in methods if m not in seen]
    if missing:
        print(f"[build_paired_summary] WARN: no rows for method(s) "
              f"{missing!r} in {p}", file=sys.stderr)
    return rows


def _apply_source_filter(
    rows: List[Dict],
    include_prefixes: List[str],
    exclude_prefixes: List[str],
) -> List[Dict]:
    """Restrict rows by ``source_id`` prefix.

    ``include_prefixes`` (if non-empty) keeps only sources whose id
    starts with one of the listed prefixes. ``exclude_prefixes`` drops
    any matching sources -- it's evaluated AFTER include, so an explicit
    deny can carve a hole out of an include set if both are passed.

    Used by run_pipeline.sh's compare phase to keep paired plots focused
    on the cohort whose ground truth is a real human-labelled mask
    (``atlas_*``), excluding ``chk_*`` deployment cases whose chk_pseudo
    GT in ``test_label_source=nnunet_pred`` mode is the same Dataset835
    dense prediction that nnUNet-sparse at step=1 IS, producing a
    structural identity-1.0 row that inflates the deployment curve.
    """
    inc = tuple(p for p in include_prefixes if p)
    exc = tuple(p for p in exclude_prefixes if p)
    if not inc and not exc:
        return rows
    out: List[Dict] = []
    for r in rows:
        sid = r.get("source_id", "")
        if inc and not sid.startswith(inc):
            continue
        if exc and sid.startswith(exc):
            continue
        out.append(r)
    return out


def _csv_list(s: str) -> List[str]:
    """Parse a comma-separated CLI value into a clean list of prefixes."""
    if not s:
        return []
    return [t.strip() for t in s.split(",") if t.strip()]


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


# ── Aggregation ─────────────────────────────────────────────────

def _aggregate(
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
        _, label = _assign_bucket(r["eff_res_mm"], edges)
        if label not in seen_set:
            seen_set.add(label)
            seen_buckets.append(label)
        by_method_bucket[(r["method"], label)][r["structure"]].append(r["dice"])
        if r["structure"] == "mean":
            eff_by_bucket[(r["method"], label)].append(r["eff_res_mm"])
    seen_buckets.sort(key=_bucket_sort_key)
    return seen_buckets, by_method_bucket, eff_by_bucket


def _series_for(
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


# ── Drawing helpers ─────────────────────────────────────────────

def _draw_overall(
    ax,
    methods: List[str],
    bucket_order: List[str],
    by_method_bucket,
    eff_by_bucket,
) -> None:
    for m in methods:
        xs, ys, es, ns = _series_for(
            m, "mean", bucket_order, by_method_bucket, eff_by_bucket,
        )
        if not xs:
            continue
        ax.errorbar(
            xs, ys, yerr=es, fmt="o-", capsize=4,
            color=_color_for(m, DEFAULT_NNUNET_COLOR
                             if m == NNUNET_METHOD_LABEL else DEFAULT_CNISP_COLOR),
            label=m,
        )
        for x, y, n in zip(xs, ys, ns):
            ax.annotate(f"{y:.3f}\nn={n}", (x, y),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=7, color=_color_for(
                            m, DEFAULT_NNUNET_COLOR if m == NNUNET_METHOD_LABEL
                            else DEFAULT_CNISP_COLOR))
    ax.set_xlabel("effective resolution (mm, through-plane)")
    ax.set_ylabel("mean Dice (4 foreground classes)")
    ax.set_title("Overall mean Dice vs effective resolution")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)


def _draw_per_class(
    axes,
    methods: List[str],
    bucket_order: List[str],
    by_method_bucket,
    eff_by_bucket,
) -> None:
    """Fill a 2x2 grid of axes (one per foreground class) with paired curves."""
    for i, c in enumerate(STRUCT_ORDER):
        ax = axes[i // 2][i % 2]
        for m in methods:
            xs, ys, es, _ = _series_for(
                m, c, bucket_order, by_method_bucket, eff_by_bucket,
            )
            if not xs:
                continue
            ax.errorbar(
                xs, ys, yerr=es, fmt="o-", capsize=3,
                color=_color_for(m, DEFAULT_NNUNET_COLOR
                                 if m == NNUNET_METHOD_LABEL
                                 else DEFAULT_CNISP_COLOR),
                label=m,
            )
        ax.set_title(c)
        ax.set_xlabel("effective resolution (mm)")
        ax.set_ylabel("Dice")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left", fontsize=8)


def _bucket_means(
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


def _draw_delta(
    ax,
    cnisp_method: str,
    bucket_order: List[str],
    by_method_bucket,
    eff_by_bucket,
) -> None:
    """Bar chart of (CNISP - nnUNet) on the mean Dice row, per shared bucket.

    Only buckets where BOTH methods have at least one observation are
    plotted; otherwise the bar would be meaningless. eff_res mean of the
    CNISP rows is used for the x-axis (CNISP is the bucket reference
    because all eff_res values come from CNISP's sweep_results.pkl).
    """
    nn = _bucket_means(NNUNET_METHOD_LABEL, bucket_order,
                       by_method_bucket, eff_by_bucket)
    cn = _bucket_means(cnisp_method, bucket_order,
                       by_method_bucket, eff_by_bucket)
    shared = [b for b in bucket_order if b in nn and b in cn]
    if not shared:
        ax.set_title(f"{cnisp_method} - {NNUNET_METHOD_LABEL}  (no shared buckets)")
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
        _color_for(cnisp_method, DEFAULT_CNISP_COLOR) if d >= 0
        else _color_for(NNUNET_METHOD_LABEL, DEFAULT_NNUNET_COLOR)
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
    ax.set_ylabel(f"Dice delta  ({cnisp_method} - {NNUNET_METHOD_LABEL})")
    ax.set_title(
        f"Head-to-head: {cnisp_method} - {NNUNET_METHOD_LABEL}  "
        f"(positive => {cnisp_method} wins)"
    )
    ax.grid(True, axis="y", alpha=0.3)


# ── Combined + stand-alone plotting ─────────────────────────────

def _plot_paired(
    cnisp_method: str,
    bucket_order: List[str],
    by_method_bucket,
    eff_by_bucket,
    out_dir: Path,
) -> Dict[str, Path]:
    methods = [NNUNET_METHOD_LABEL, cnisp_method]

    # Stand-alone: overall ────────────────────────────────────────
    overall_path = out_dir / "paired_overall_dice_vs_eff_res.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    _draw_overall(ax, methods, bucket_order, by_method_bucket, eff_by_bucket)
    fig.tight_layout()
    fig.savefig(str(overall_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Stand-alone: per-class 2x2 grid ────────────────────────────
    per_class_path = out_dir / "paired_per_class_dice_vs_eff_res.png"
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    _draw_per_class(axes, methods, bucket_order, by_method_bucket, eff_by_bucket)
    fig.suptitle("Per-class Dice vs effective resolution", fontsize=12,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(str(per_class_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Stand-alone: delta ──────────────────────────────────────────
    delta_path = out_dir / "paired_delta_dice_vs_eff_res.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    _draw_delta(ax, cnisp_method, bucket_order, by_method_bucket, eff_by_bucket)
    fig.tight_layout()
    fig.savefig(str(delta_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Combined three-row figure ───────────────────────────────────
    # GridSpec layout:
    #   row 0          -> overall mean Dice (full width)
    #   row 1 (2x2 sub) -> per-class panels
    #   row 2          -> delta bar chart (full width)
    combined_path = out_dir / "paired_dice_vs_eff_res.png"
    fig = plt.figure(figsize=(14, 16))
    gs = fig.add_gridspec(3, 1, hspace=0.45, height_ratios=[1, 1.6, 1])

    ax0 = fig.add_subplot(gs[0])
    _draw_overall(ax0, methods, bucket_order, by_method_bucket, eff_by_bucket)

    inner_gs = gs[1].subgridspec(2, 2, hspace=0.55, wspace=0.3)
    inner_axes = [
        [fig.add_subplot(inner_gs[0, 0]), fig.add_subplot(inner_gs[0, 1])],
        [fig.add_subplot(inner_gs[1, 0]), fig.add_subplot(inner_gs[1, 1])],
    ]
    _draw_per_class(inner_axes, methods, bucket_order,
                    by_method_bucket, eff_by_bucket)

    ax2 = fig.add_subplot(gs[2])
    _draw_delta(ax2, cnisp_method, bucket_order,
                by_method_bucket, eff_by_bucket)

    fig.suptitle(
        f"{NNUNET_METHOD_LABEL} vs {cnisp_method}: Dice vs effective resolution  "
        f"(driven by paired_per_source.csv)",
        fontsize=13, fontweight="bold", y=0.995,
    )
    fig.savefig(str(combined_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "overall": overall_path,
        "per_class": per_class_path,
        "delta": delta_path,
        "combined": combined_path,
    }


# ── CSV writer (per-bucket paired view) ─────────────────────────

def _write_paired_csv(
    cnisp_method: str,
    bucket_order: List[str],
    by_method_bucket,
    eff_by_bucket,
    out_path: Path,
) -> None:
    """Bucket-by-structure table with both methods' mean/std/n + delta on the mean row.

    Mirrors ``paired_summary__<run_tag>.csv`` schema but is keyed on
    eff_res bucket rather than aggregated columns so downstream notebooks
    can plot it without re-parsing the wide table.
    """
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


# ── Main ────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="nnunet/configs.yaml")
    ap.add_argument(
        "--paired-csv", required=True,
        help="Path to paired_per_source__<run_tag>.csv (written by "
             "nnunet/compare_native.py).",
    )
    ap.add_argument(
        "--cnisp-method", required=True,
        help="Method label of the CNISP rows in the paired CSV "
             "(e.g. CNISP-atlasGT, CNISP-nnUNetPred). The nnUNet method "
             "label is always 'nnUNet-sparse'.",
    )
    ap.add_argument(
        "--nnunet-method", default=NNUNET_METHOD_LABEL,
        help=f"Override the nnUNet method label (default: {NNUNET_METHOD_LABEL}).",
    )
    ap.add_argument(
        "--out-dir", required=True,
        help="Where to write the paired plots and CSV. Pipeline "
             "convention: ${work_dir}/comparison/viz/paired__<run_tag>/.",
    )
    ap.add_argument(
        "--include-source-prefixes", default=None,
        help="Comma-separated source_id prefixes to keep (e.g. 'atlas_'). "
             "Default: read 'viz_include_source_prefixes' from --config "
             "(if absent, no include-side filtering -- keep everything).",
    )
    ap.add_argument(
        "--exclude-source-prefixes", default=None,
        help="Comma-separated source_id prefixes to drop (e.g. 'chk_'). "
             "Default: read 'viz_exclude_source_prefixes' from --config "
             "(default 'chk_' there, so the paired plots stay focused on "
             "human-labelled cases and avoid the chk_ deployment-mode "
             "identity-1.0 row at step=1).",
    )
    args = ap.parse_args()

    cfg = _load_yaml(Path(args.config))
    bucket_edges = list(cfg.get(
        "summary_bucket_edges_mm",
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.5, 8.5, 11.0, 13.0],
    ))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.include_source_prefixes is None:
        include_prefixes = list(cfg.get("viz_include_source_prefixes", []))
    else:
        include_prefixes = _csv_list(args.include_source_prefixes)
    if args.exclude_source_prefixes is None:
        exclude_prefixes = list(cfg.get("viz_exclude_source_prefixes", []))
    else:
        exclude_prefixes = _csv_list(args.exclude_source_prefixes)

    methods = [args.nnunet_method, args.cnisp_method]
    rows = _read_paired_csv(Path(args.paired_csv), methods)
    n_before = len(rows)
    rows = _apply_source_filter(rows, include_prefixes, exclude_prefixes)
    if include_prefixes or exclude_prefixes:
        print(
            f"[build_paired_summary] source filter: "
            f"include={include_prefixes!r} exclude={exclude_prefixes!r} "
            f"-> {len(rows)}/{n_before} rows kept.",
            file=sys.stderr,
        )
    if not rows:
        raise SystemExit(
            f"All rows filtered out (include={include_prefixes!r}, "
            f"exclude={exclude_prefixes!r}). Relax the filter or check "
            f"source_id prefixes in {args.paired_csv}."
        )
    bucket_order, by_method_bucket, eff_by_bucket = _aggregate(
        rows, bucket_edges,
    )

    paths = _plot_paired(
        args.cnisp_method, bucket_order, by_method_bucket,
        eff_by_bucket, out_dir,
    )
    csv_path = out_dir / "paired_summary_by_eff_res.csv"
    _write_paired_csv(
        args.cnisp_method, bucket_order, by_method_bucket,
        eff_by_bucket, csv_path,
    )

    print(f"[build_paired_summary] {args.cnisp_method} vs {args.nnunet_method}: "
          f"{len(rows)} rows -> {out_dir}/")
    for k, p in paths.items():
        print(f"  [{k}] {p}")
    print(f"  [csv]      {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
