"""Matplotlib rendering primitives for the evaluation figures (rendering layer).

Top layer of ``simulation.evaluation`` (analogous to the ``draw_*`` / ``plot_*``
functions in ``nnunet.lib.viz``): each ``*_figure`` takes an aggregated result
(from ``aggregate`` or ``synthetic``) plus an output path and writes one PNG. No
data loading or aggregation happens here, so a new figure = a new function here +
a new thin driver, reusing the same metrics/aggregate layers.

Method display styling (the 5 pipelines) lives here since it is presentation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec

from simulation.evaluation.metrics import METHODS, STRUCTURES

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 11,
    "savefig.dpi": 200, "savefig.bbox": "tight",
})

LEGEND = {"nnUNet": "nnUNet (baseline)",
          "Cascade UNet": "Cascade UNet (nnU\u2192nnU self-correction)",
          "CNISP": "CNISP (shape prior only)",
          "Proposed": "Proposed (nnU\u2192CNISP\u2192nnU)",
          "Oracle": "Oracle (CNISP+GT ceiling)",
          "GT": "Ground truth (reference)"}
COLOR = {"nnUNet": "#d62728", "Cascade UNet": "#9467bd", "CNISP": "#1f77b4",
         "Proposed": "#2ca02c", "Oracle": "#7f7f7f", "GT": "#000000"}


def _violin(ax, series_by_method: Dict, widths: float, rotation: int = 30) -> None:
    """Per-METHODS violin that tolerates empty/degenerate arms.

    matplotlib's ``violinplot`` raises on a zero-size array (e.g. an arm with no
    rows in the metrics table, or the GT reference before the table is rebuilt),
    which would abort the whole figure. We therefore plot only the non-empty arms
    at their fixed METHODS x-slot, colour each by its own method, and still label
    every slot. A constant arm (e.g. GT range == 0) is jittered infinitesimally so
    its KDE renders instead of collapsing.
    """
    positions, data, kept = [], [], []
    for i, m in enumerate(METHODS):
        arr = np.asarray(series_by_method.get(m, []), dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        if np.ptp(arr) == 0:
            arr = arr + np.linspace(-1e-6, 1e-6, arr.size)
        positions.append(i + 1); data.append(arr); kept.append(m)
    if data:
        parts = ax.violinplot(data, positions=positions, showmedians=True, widths=widths)
        for b, m in zip(parts["bodies"], kept):
            b.set_facecolor(COLOR[m]); b.set_alpha(0.6); b.set_edgecolor("0.3")
        for k in ("cmedians", "cbars", "cmins", "cmaxes"):
            if k in parts:
                parts[k].set_color("0.3")
    ax.set_xticks(range(1, len(METHODS) + 1))
    ax.set_xticklabels(METHODS, rotation=rotation, ha="right", fontsize=8.5)


def _foot(fig, synthetic: bool) -> None:
    if synthetic:
        fig.text(0.995, 0.004, "Illustrative layout \u00b7 synthetic placeholder data",
                 ha="right", fontsize=7, style="italic", color="0.55")


def stability_figure(cov_mean: Dict, cov_sd: Dict, on_range: Dict,
                     out_path: Path, synthetic: bool = False) -> None:
    """Cross-resolution volume stability: CoV bars + optic-nerve per-scan range."""
    fig = plt.figure(figsize=(11, 4.4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.1, 1], wspace=0.28)
    ax = fig.add_subplot(gs[0]); x = np.arange(len(STRUCTURES))
    n_m = len(METHODS); w = 0.8 / n_m; c = (n_m - 1) / 2.0
    for i, m in enumerate(METHODS):
        vals = [cov_mean[m][s] for s in STRUCTURES]; err = [cov_sd[m][s] for s in STRUCTURES]
        ax.bar(x + (i - c) * w, vals, w, yerr=err, capsize=2.5, color=COLOR[m],
               label=LEGEND[m], ec="white", lw=0.5, error_kw=dict(lw=0.8))
    ax.axhline(10, ls=":", color="0.4")
    ax.text(len(STRUCTURES) - 0.55, 10.4, "10% (radiomics stability threshold)",
            fontsize=7.5, color="0.4", ha="right")
    ax.set_xticks(x); ax.set_xticklabels(STRUCTURES)
    ax.set_ylabel("Volume CoV across resolutions (%)  \u2193")
    ax.set_title("(a)  Lower cross-resolution variability = better harmonization", loc="left")
    ax.legend(fontsize=8, loc="upper left")
    axb = fig.add_subplot(gs[1])
    _violin(axb, on_range, widths=0.8, rotation=30)
    axb.set_ylabel("Per-scan volume range\nacross resolutions (% of mean)  \u2193")
    axb.set_title("(b)  Optic nerve: per-scan wander", loc="left", fontsize=10.5)
    _foot(fig, synthetic)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path)); plt.close(fig)


def _bland_altman(ax, fig, v_pred, v_gt, thickness, title, col, colorbar=False):
    v_pred, v_gt, thickness = map(np.asarray, (v_pred, v_gt, thickness))
    mean, diff = (v_pred + v_gt) / 2, v_pred - v_gt
    b, s = diff.mean(), diff.std()
    ccc = 2 * np.cov(v_pred, v_gt, bias=True)[0, 1] / (
        v_pred.var() + v_gt.var() + (v_pred.mean() - v_gt.mean()) ** 2)
    sc = ax.scatter(mean, diff, c=thickness, cmap="viridis", s=34, ec="0.25", lw=0.4, zorder=3)
    ax.axhline(b, color=col, lw=1.9); ax.axhline(b + 1.96 * s, color=col, ls="--", lw=1.2)
    ax.axhline(b - 1.96 * s, color=col, ls="--", lw=1.2); ax.axhline(0, color="0.55", ls=":", lw=0.9)
    ax.text(0.975, 0.965, f"bias  {b:+.0f} mm\u00b3\nLoA  \u00b1{1.96*s:.0f} mm\u00b3\nLin's CCC  {ccc:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", alpha=0.9))
    ax.set_title(title, loc="left"); ax.set_xlabel("mean volume  (V$_{pred}$+V$_{GT}$)/2  (mm\u00b3)")
    if colorbar:
        cb = fig.colorbar(sc, ax=ax, pad=0.02); cb.set_label("slice thickness / eff. res (mm)", fontsize=8.5)


def signed_volume_error_figure(signed: Dict, out_path: Path,
                               synthetic: bool = False) -> None:
    """Signed volume error across methods (violins only; no Bland-Altman).

    This is the default volume-veracity figure: the same panel (c) that
    ``volume_agreement_figure`` draws, rendered standalone.
    """
    fig = plt.figure(figsize=(6.2, 4.6))
    ax = fig.add_subplot(1, 1, 1)
    _violin(ax, signed, widths=0.85, rotation=25)
    ax.axhline(0, color="0.55", ls=":", lw=0.9)
    ax.set_ylabel("signed volume error (%)")
    ax.set_title("Signed volume error across methods", loc="left")
    ax.text(-0.09, 0.98, "over\nestimate", transform=ax.transAxes, fontsize=7.5,
            color="0.4", va="top")
    ax.text(-0.09, 0.02, "under\nestimate", transform=ax.transAxes, fontsize=7.5,
            color="0.4", va="bottom")
    _foot(fig, synthetic)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path)); plt.close(fig)


def volume_agreement_figure(per_arm: Dict, signed: Dict,
                            out_path: Path, synthetic: bool = False,
                            bland_altman: bool = False) -> None:
    """Volume veracity. Default (``bland_altman=False``): signed volume error across
    methods only. ``bland_altman=True``: the full 3-panel figure (Bland-Altman for
    nnU-Net + Proposed, then signed-error violins)."""
    if not bland_altman:
        signed_volume_error_figure(signed, out_path, synthetic=synthetic)
        return
    fig = plt.figure(figsize=(13, 4.4))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1.12, 1.05], wspace=0.33)
    ax_a = fig.add_subplot(gs[0]); ax_a.set_ylabel("V$_{pred}$ \u2212 V$_{GT}$  (mm\u00b3)")
    _bland_altman(ax_a, fig, **{k: per_arm["nnUNet"][k] for k in ("v_pred", "v_gt", "thickness")},
                  title="(a)  Bland\u2013Altman \u2014 nnUNet", col=COLOR["nnUNet"])
    ax_b = fig.add_subplot(gs[1], sharey=ax_a)
    _bland_altman(ax_b, fig, **{k: per_arm["Proposed"][k] for k in ("v_pred", "v_gt", "thickness")},
                  title="(b)  Bland\u2013Altman \u2014 Proposed", col=COLOR["Proposed"], colorbar=True)
    all_diff = np.concatenate([per_arm["nnUNet"]["v_pred"] - per_arm["nnUNet"]["v_gt"],
                               per_arm["Proposed"]["v_pred"] - per_arm["Proposed"]["v_gt"]])
    pad = 0.15 * (all_diff.max() - all_diff.min() + 1)
    ax_a.set_ylim(all_diff.min() - pad, all_diff.max() + pad)
    ax_a.text(-0.17, 0.98, "over-\nestimate", transform=ax_a.transAxes, fontsize=7.5, color="0.4", va="top")
    ax_a.text(-0.17, 0.02, "under-\nestimate", transform=ax_a.transAxes, fontsize=7.5, color="0.4", va="bottom")
    ax_c = fig.add_subplot(gs[2])
    _violin(ax_c, signed, widths=0.85, rotation=25)
    ax_c.axhline(0, color="0.55", ls=":", lw=0.9)
    ax_c.set_ylabel("signed volume error (%)"); ax_c.set_title("(c)  Signed volume error across methods", loc="left")
    fig.suptitle("Volume accuracy on the GT set: near-zero bias, tight limits of agreement, "
                 "no thickness-dependent drift for the proposed method", fontsize=10.6, y=1.02)
    _foot(fig, synthetic)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path)); plt.close(fig)


def single_bland_altman_figure(v_pred, v_gt, thickness, arm: str, structure: str,
                               out_path: Path, synthetic: bool = False) -> bool:
    """One standalone Bland-Altman panel for a single arm (per-arm subfolder).

    Reuses ``_bland_altman`` (same bias/LoA/CCC as the combined figure). Returns
    True if it drew (>=2 paired points), False if skipped (too few points for a
    meaningful cloud / for ``np.cov``).
    """
    v_pred = np.asarray(v_pred, dtype=float)
    v_gt = np.asarray(v_gt, dtype=float)
    if v_pred.size < 2:
        return False
    fig = plt.figure(figsize=(5.4, 4.6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_ylabel("V$_{pred}$ − V$_{GT}$  (mm³)")
    _bland_altman(ax, fig, v_pred=v_pred, v_gt=v_gt, thickness=thickness,
                  title=f"Bland–Altman — {arm} ({structure})",
                  col=COLOR.get(arm, "#333333"), colorbar=True)
    ax.axhline(0, color="0.55", ls=":", lw=0.9)
    _foot(fig, synthetic)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path)); plt.close(fig)
    return True


def surface_figure(metrics: Dict[str, Dict[str, np.ndarray]],
                   out_path: Path, synthetic: bool = False) -> None:
    """Surface quality: ASSD / HD95 / Surface-Dice boxplots per method."""
    names = ["ASSD (mm)", "HD95 (mm)", "Surface Dice @1mm"]
    arrows = {"ASSD (mm)": "\u2193", "HD95 (mm)": "\u2193", "Surface Dice @1mm": "\u2191"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, name in zip(axes, names):
        bp = ax.boxplot([metrics[name][m] for m in METHODS], patch_artist=True, widths=0.6,
                        showfliers=False, medianprops=dict(color="0.15", lw=1.4))
        for i, b in enumerate(bp["boxes"]):
            b.set_facecolor(COLOR[METHODS[i]]); b.set_alpha(0.75); b.set_edgecolor("0.3")
        ax.set_xticks(range(1, len(METHODS) + 1)); ax.set_xticklabels(METHODS, rotation=30, ha="right", fontsize=8.5)
        ax.set_title(f"{name}  {arrows[name]}", loc="left")
    axes[0].set_ylabel("value")
    fig.suptitle("Surface quality: the proposed masks are closer to the reference boundary and smoother",
                 fontsize=11.5, y=1.02)
    _foot(fig, synthetic)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path)); plt.close(fig)


def _dice_series(arm, structure, bucket_order, by_arm_bucket, eff_by_bucket):
    """(xs, ys, es, ns) sorted by eff_res for one (arm, structure) across buckets."""
    xs, ys, es, ns = [], [], [], []
    for bkt in bucket_order:
        vals = by_arm_bucket.get((arm, bkt), {}).get(structure, [])
        effs = eff_by_bucket.get((arm, bkt), [])
        if not vals or not effs:
            continue
        arr = np.asarray(vals, dtype=float)
        xs.append(float(np.mean(effs)))
        ys.append(float(arr.mean()))
        es.append(float(arr.std()))
        ns.append(int(arr.size))
    if xs:
        order = np.argsort(xs)
        xs = [xs[i] for i in order]; ys = [ys[i] for i in order]
        es = [es[i] for i in order]; ns = [ns[i] for i in order]
    return xs, ys, es, ns


def _annotate_nodes(ax, xs, ys, color, idx, fs=6.0):
    """Label each (x, y) node with its Dice, staggered above/below by arm idx so
    the (up to 5) near-coincident markers at each eff_res don't overlap."""
    sign = 1 if (idx % 2 == 0) else -1
    dy = sign * (6 + (idx // 2) * 11)
    va = "bottom" if sign > 0 else "top"
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, dy), ha="center", va=va, fontsize=fs, color=color)


def dice_vs_eff_res_figure(bucket_order, by_arm_bucket, eff_by_bucket, out_path,
                           delta_arm: str = "Proposed", baseline: str = "nnUNet",
                           legend_map=None, ymin: float = 0.5,
                           synthetic: bool = False) -> None:
    """5-arm Dice-vs-effective-resolution: overall + per-class 2x2 + delta panel.

    Single-source-of-truth replacement for the old comparison-track combined__thick
    figure, driven purely by metrics_long (native-mask Dice for every arm). Colors
    + display names come from this module's COLOR/LEGEND (already A-E); pass
    ``legend_map`` (arm -> str) to override the legend from a config.
    """
    lg = dict(LEGEND)
    if legend_map:
        lg.update(legend_map)
    fig = plt.figure(figsize=(12, 16))
    gs = gridspec.GridSpec(3, 1, hspace=0.32, height_ratios=[1, 1.9, 1])

    ax0 = fig.add_subplot(gs[0])
    for j, m in enumerate(METHODS):
        xs, ys, es, _ = _dice_series(m, "mean", bucket_order, by_arm_bucket, eff_by_bucket)
        if not xs:
            continue
        ax0.errorbar(xs, ys, yerr=es, fmt="o-", capsize=4, color=COLOR[m], label=lg.get(m, m))
        _annotate_nodes(ax0, xs, ys, COLOR[m], j, fs=6.5)
    ax0.set_xlabel("effective resolution (mm, through-plane)")
    ax0.set_ylabel("mean Dice (4 fg classes)")
    ax0.set_title("(a)  Overall mean Dice vs effective resolution", loc="left")
    ax0.set_ylim(ymin, 1.0); ax0.grid(True, alpha=0.3); ax0.legend(fontsize=8, loc="lower left")

    inner = gs[1].subgridspec(2, 2, hspace=0.4, wspace=0.22)
    for i, struct in enumerate(STRUCTURES):
        ax = fig.add_subplot(inner[i // 2, i % 2])
        for j, m in enumerate(METHODS):
            xs, ys, es, _ = _dice_series(m, struct, bucket_order, by_arm_bucket, eff_by_bucket)
            if not xs:
                continue
            ax.errorbar(xs, ys, yerr=es, fmt="o-", capsize=3, color=COLOR[m], label=lg.get(m, m))
            _annotate_nodes(ax, xs, ys, COLOR[m], j, fs=5.5)
        ax.set_title(struct, loc="left"); ax.set_xlabel("eff. res (mm)")
        ax.set_ylabel("Dice"); ax.set_ylim(ymin, 1.0); ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7, loc="lower left")

    ax2 = fig.add_subplot(gs[2])
    xs, ds = [], []
    for b in bucket_order:
        base = by_arm_bucket.get((baseline, b), {}).get("mean", [])
        dl = by_arm_bucket.get((delta_arm, b), {}).get("mean", [])
        effs = eff_by_bucket.get((delta_arm, b), [])
        if not base or not dl or not effs:
            continue
        xs.append(float(np.mean(effs)))
        ds.append(float(np.mean(dl) - np.mean(base)))
    if xs:
        order = np.argsort(xs); xs = [xs[i] for i in order]; ds = [ds[i] for i in order]
        colors = [COLOR[delta_arm] if d >= 0 else COLOR[baseline] for d in ds]
        ax2.bar(xs, ds, width=0.4, color=colors, alpha=0.85, edgecolor="#444", lw=0.6)
        for x, d in zip(xs, ds):
            ax2.annotate(f"{d:+.3f}", (x, d), textcoords="offset points",
                         xytext=(0, 3 if d >= 0 else -10), ha="center",
                         fontsize=6, color="#222")
    ax2.axhline(0, color="#444", lw=0.8)
    ax2.set_xlabel("effective resolution (mm)")
    ax2.set_ylabel(f"Dice Δ ({lg.get(delta_arm, delta_arm)} − {lg.get(baseline, baseline)})")
    ax2.set_title(f"(c)  Head-to-head: {lg.get(delta_arm, delta_arm)} − "
                  f"{lg.get(baseline, baseline)}  (positive → {delta_arm} wins)", loc="left")
    ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Dice vs effective resolution — 5-arm (from metrics_long)",
                 fontsize=12, fontweight="bold", y=0.905)
    _foot(fig, synthetic)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path)); plt.close(fig)


__all__ = ["stability_figure", "volume_agreement_figure",
           "signed_volume_error_figure", "single_bland_altman_figure",
           "surface_figure", "dice_vs_eff_res_figure", "LEGEND", "COLOR"]
