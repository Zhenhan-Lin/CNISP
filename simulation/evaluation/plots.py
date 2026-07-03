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

LEGEND = {"nnU-Net": "nnU-Net (baseline)", "CNISP": "CNISP (shape prior only)",
          "nnU\u2192nnU": "nnU\u2192nnU (self-correction)",
          "Proposed": "Proposed (nnU\u2192CNISP\u2192nnU)", "Oracle": "CNISP+GT (oracle)"}
COLOR = {"nnU-Net": "#d62728", "CNISP": "#1f77b4", "nnU\u2192nnU": "#9467bd",
         "Proposed": "#2ca02c", "Oracle": "#7f7f7f"}


def _foot(fig, synthetic: bool) -> None:
    if synthetic:
        fig.text(0.995, 0.004, "Illustrative layout \u00b7 synthetic placeholder data",
                 ha="right", fontsize=7, style="italic", color="0.55")


def stability_figure(cov_mean: Dict, cov_sd: Dict, on_range: Dict,
                     out_path: Path, synthetic: bool = False) -> None:
    """Cross-resolution volume stability: CoV bars + optic-nerve per-scan range."""
    fig = plt.figure(figsize=(11, 4.4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.1, 1], wspace=0.28)
    ax = fig.add_subplot(gs[0]); x = np.arange(len(STRUCTURES)); w = 0.16
    for i, m in enumerate(METHODS):
        vals = [cov_mean[m][s] for s in STRUCTURES]; err = [cov_sd[m][s] for s in STRUCTURES]
        ax.bar(x + (i - 2) * w, vals, w, yerr=err, capsize=2.5, color=COLOR[m],
               label=LEGEND[m], ec="white", lw=0.5, error_kw=dict(lw=0.8))
    ax.axhline(10, ls=":", color="0.4")
    ax.text(len(STRUCTURES) - 0.55, 10.4, "10% (radiomics stability threshold)",
            fontsize=7.5, color="0.4", ha="right")
    ax.set_xticks(x); ax.set_xticklabels(STRUCTURES)
    ax.set_ylabel("Volume CoV across resolutions (%)  \u2193")
    ax.set_title("(a)  Lower cross-resolution variability = better harmonization", loc="left")
    ax.legend(fontsize=8, loc="upper left")
    axb = fig.add_subplot(gs[1])
    parts = axb.violinplot([on_range[m] for m in METHODS], showmedians=True, widths=0.8)
    for i, b in enumerate(parts["bodies"]):
        b.set_facecolor(COLOR[METHODS[i]]); b.set_alpha(0.6); b.set_edgecolor("0.3")
    for k in ("cmedians", "cbars", "cmins", "cmaxes"):
        if k in parts: parts[k].set_color("0.3")
    axb.set_xticks(range(1, 6)); axb.set_xticklabels(METHODS, rotation=30, ha="right", fontsize=8.5)
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


def volume_agreement_figure(per_arm: Dict, signed: Dict,
                            out_path: Path, synthetic: bool = False) -> None:
    """Volume veracity: Bland-Altman (nnU-Net vs Proposed) + signed-error violins."""
    fig = plt.figure(figsize=(13, 4.4))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1.12, 1.05], wspace=0.33)
    ax_a = fig.add_subplot(gs[0]); ax_a.set_ylabel("V$_{pred}$ \u2212 V$_{GT}$  (mm\u00b3)")
    _bland_altman(ax_a, fig, **{k: per_arm["nnU-Net"][k] for k in ("v_pred", "v_gt", "thickness")},
                  title="(a)  Bland\u2013Altman \u2014 nnU-Net", col=COLOR["nnU-Net"])
    ax_b = fig.add_subplot(gs[1], sharey=ax_a)
    _bland_altman(ax_b, fig, **{k: per_arm["Proposed"][k] for k in ("v_pred", "v_gt", "thickness")},
                  title="(b)  Bland\u2013Altman \u2014 Proposed", col=COLOR["Proposed"], colorbar=True)
    all_diff = np.concatenate([per_arm["nnU-Net"]["v_pred"] - per_arm["nnU-Net"]["v_gt"],
                               per_arm["Proposed"]["v_pred"] - per_arm["Proposed"]["v_gt"]])
    pad = 0.15 * (all_diff.max() - all_diff.min() + 1)
    ax_a.set_ylim(all_diff.min() - pad, all_diff.max() + pad)
    ax_a.text(-0.17, 0.98, "over-\nestimate", transform=ax_a.transAxes, fontsize=7.5, color="0.4", va="top")
    ax_a.text(-0.17, 0.02, "under-\nestimate", transform=ax_a.transAxes, fontsize=7.5, color="0.4", va="bottom")
    ax_c = fig.add_subplot(gs[2])
    parts = ax_c.violinplot([signed[m] for m in METHODS], showmedians=True, widths=0.85)
    for i, b in enumerate(parts["bodies"]):
        b.set_facecolor(COLOR[METHODS[i]]); b.set_alpha(0.62); b.set_edgecolor("0.3")
    for k in ("cmedians", "cbars", "cmins", "cmaxes"):
        if k in parts: parts[k].set_color("0.3")
    ax_c.axhline(0, color="0.55", ls=":", lw=0.9)
    ax_c.set_xticks(range(1, 6)); ax_c.set_xticklabels(METHODS, rotation=25, ha="right", fontsize=8.5)
    ax_c.set_ylabel("signed volume error (%)"); ax_c.set_title("(c)  Signed volume error across methods", loc="left")
    fig.suptitle("Volume accuracy on the GT set: near-zero bias, tight limits of agreement, "
                 "no thickness-dependent drift for the proposed method", fontsize=10.6, y=1.02)
    _foot(fig, synthetic)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path)); plt.close(fig)


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
        ax.set_xticks(range(1, 6)); ax.set_xticklabels(METHODS, rotation=30, ha="right", fontsize=8.5)
        ax.set_title(f"{name}  {arrows[name]}", loc="left")
    axes[0].set_ylabel("value")
    fig.suptitle("Surface quality: the proposed masks are closer to the reference boundary and smoother",
                 fontsize=11.5, y=1.02)
    _foot(fig, synthetic)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path)); plt.close(fig)


__all__ = ["stability_figure", "volume_agreement_figure", "surface_figure",
           "LEGEND", "COLOR"]
