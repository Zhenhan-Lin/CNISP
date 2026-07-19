"""Matplotlib rendering for anatomical plausibility figures (rendering layer).

Figures A-E for the two-layer anatomical plausibility evaluation.
Reuses COLOR/LEGEND from the existing plots module for visual consistency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec

from simulation.evaluation.metrics import STRUCTURES
from simulation.evaluation.plots import COLOR, LEGEND

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 11,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})

# Layer pair display info
_LAYER_PAIRS = [
    ("nnUNet", "CNISP", "Layer 1 (prior channel)"),
    ("Cascade UNet", "Proposed", "Layer 2 (cascade output)"),
]


def _significance_star(p: float) -> str:
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "n.s."


def _sort_buckets(labels: List[str]) -> List[str]:
    """Sort bucket labels by lower bound; 'unknown' last."""
    from nnunet.helpers.buckets import bucket_sort_key
    return sorted(labels, key=bucket_sort_key)


# ============================================================
# Figure A: Topology violation rate vs eff_res bucket
# ============================================================

def topology_violation_figure(
    violation_df,
    tests_df,
    out_path: Path,
    metric: str = "has_multi_cc",
) -> None:
    """2x2 per structure: grouped bars (Layer 1 + Layer 2) per bucket.

    violation_df: output of topology_violation_rate() with columns
        [arm, structure, bucket_label, rate, ci_lo, ci_hi, n].
    tests_df: output of paired_tests() for p-value annotations.
    """
    import pandas as pd

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    axes_flat = axes.flatten()

    for idx, struct in enumerate(STRUCTURES):
        ax = axes_flat[idx]
        sub = violation_df[violation_df["structure"] == struct]
        buckets = _sort_buckets(sub["bucket_label"].unique().tolist())
        if not buckets:
            ax.set_title(struct)
            continue

        x = np.arange(len(buckets))
        n_bars = 4  # nnUNet, CNISP, Cascade UNet, Proposed
        w = 0.8 / n_bars

        arms_ordered = ["nnUNet", "CNISP", "Cascade UNet", "Proposed"]
        for i, arm in enumerate(arms_ordered):
            rates, ci_lo, ci_hi = [], [], []
            for bkt in buckets:
                row = sub[(sub["arm"] == arm) & (sub["bucket_label"] == bkt)]
                if row.empty:
                    rates.append(0.0)
                    ci_lo.append(0.0)
                    ci_hi.append(0.0)
                else:
                    r = row.iloc[0]
                    rates.append(r["rate"])
                    # Error-bar HALF-LENGTHS (distance from the plotted rate to
                    # each CI bound). A Wilson-type CI is not centered on the raw
                    # rate, so near 0/1 a bound can cross the rate and make a
                    # half-length negative -- matplotlib rejects yerr < 0. Clamp
                    # to 0 (the whisker just doesn't extend past the bar).
                    ci_lo.append(max(0.0, r["rate"] - r["ci_lo"]))
                    ci_hi.append(max(0.0, r["ci_hi"] - r["rate"]))

            offset = (i - (n_bars - 1) / 2.0) * w
            yerr = [ci_lo, ci_hi]
            ax.bar(x + offset, rates, w, yerr=yerr, capsize=2,
                   color=COLOR.get(arm, "#888"), label=LEGEND.get(arm, arm),
                   ec="white", lw=0.5, error_kw=dict(lw=0.7))

        # Significance annotations from tests_df
        if tests_df is not None and not tests_df.empty:
            for layer_a, layer_b, layer_label in _LAYER_PAIRS:
                t_rows = tests_df[
                    (tests_df["structure"] == struct) &
                    (tests_df["metric"] == metric) &
                    (tests_df["layer"].str.contains(layer_label.split(" ")[0].replace("Layer", "Layer")))
                ]
                # Overall significance
                overall = tests_df[
                    (tests_df["structure"] == struct) &
                    (tests_df["metric"] == metric) &
                    (tests_df["bucket"] == "overall")
                ]
                # (annotations omitted from crowded per-bucket view for clarity)

        ax.set_xticks(x)
        ax.set_xticklabels(buckets, rotation=35, ha="right", fontsize=8)
        ax.set_title(struct)
        ax.set_ylim(0, 1.05)
        if idx % 2 == 0:
            ax.set_ylabel("Violation rate")
        if idx < 2:
            ax.set_xlabel("")

    axes_flat[0].legend(fontsize=8, loc="upper left", ncol=2)
    fig.suptitle(
        f"Topology violation rate ({metric.replace('has_', '').replace('_', ' ')}) "
        f"vs effective resolution",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path))
    plt.close(fig)


# ============================================================
# Figure B: Cross-slice continuity
# ============================================================

def continuity_figure(
    df,
    out_path: Path,
    metric: str = "mean_centroid_jump_mm",
    ylabel: str = "Mean centroid jump (mm)",
    title: str = "Cross-slice continuity vs effective resolution",
) -> None:
    """2x2 per structure: boxplot per arm within each bucket."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes_flat = axes.flatten()

    arms_ordered = ["nnUNet", "CNISP", "Cascade UNet", "Proposed"]
    buckets = _sort_buckets(df["bucket_label"].unique().tolist())

    for idx, struct in enumerate(STRUCTURES):
        ax = axes_flat[idx]
        sub = df[df["structure"] == struct]

        positions = []
        data_all = []
        colors_all = []
        tick_positions = []
        tick_labels_list = []

        n_arms = len(arms_ordered)
        group_width = n_arms + 1

        for bi, bkt in enumerate(buckets):
            bkt_data = sub[sub["bucket_label"] == bkt]
            base_pos = bi * group_width
            tick_positions.append(base_pos + (n_arms - 1) / 2.0)
            tick_labels_list.append(bkt)

            for ai, arm in enumerate(arms_ordered):
                vals = bkt_data[bkt_data["arm"] == arm][metric].dropna().values
                if len(vals) == 0:
                    vals = np.array([np.nan])
                positions.append(base_pos + ai)
                data_all.append(vals)
                colors_all.append(COLOR.get(arm, "#888"))

        if data_all:
            bp = ax.boxplot(
                data_all, positions=positions, widths=0.7,
                patch_artist=True, showfliers=False,
                medianprops=dict(color="0.15", lw=1.2),
            )
            for patch, c in zip(bp["boxes"], colors_all):
                patch.set_facecolor(c)
                patch.set_alpha(0.7)
                patch.set_edgecolor("0.3")

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels_list, rotation=35, ha="right", fontsize=8)
        ax.set_title(struct)
        if idx % 2 == 0:
            ax.set_ylabel(ylabel)

    # Legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=COLOR.get(a, "#888"), label=LEGEND.get(a, a),
                     alpha=0.7, ec="0.3") for a in arms_ordered]
    axes_flat[0].legend(handles=handles, fontsize=8, loc="upper left")

    fig.suptitle(title, fontsize=12, y=1.01)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path))
    plt.close(fig)


# ============================================================
# Figure D: Compactness distribution (optional)
# ============================================================

def compactness_figure(
    df,
    out_path: Path,
    eff_res_threshold: float = 5.0,
) -> None:
    """Split violin for thick-slice cases: one per structure, arms overlaid."""
    thick = df[df["eff_res"].notna() & (df["eff_res"] >= eff_res_threshold)]
    if thick.empty:
        return

    arms_ordered = ["nnUNet", "CNISP", "Cascade UNet", "Proposed"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)

    for idx, struct in enumerate(STRUCTURES):
        ax = axes[idx]
        sub = thick[thick["structure"] == struct]
        data, positions, kept = [], [], []
        for i, arm in enumerate(arms_ordered):
            vals = sub[sub["arm"] == arm]["compactness"].dropna().values
            if len(vals) == 0:
                continue
            if np.ptp(vals) == 0:
                vals = vals + np.linspace(-1e-6, 1e-6, len(vals))
            data.append(vals)
            positions.append(i + 1)
            kept.append(arm)

        if data:
            parts = ax.violinplot(data, positions=positions, showmedians=True, widths=0.7)
            for b, arm in zip(parts["bodies"], kept):
                b.set_facecolor(COLOR.get(arm, "#888"))
                b.set_alpha(0.6)
                b.set_edgecolor("0.3")
            for k in ("cmedians", "cbars", "cmins", "cmaxes"):
                if k in parts:
                    parts[k].set_color("0.3")

        ax.set_xticks(range(1, len(arms_ordered) + 1))
        ax.set_xticklabels(arms_ordered, rotation=30, ha="right", fontsize=8)
        ax.set_title(struct)

    axes[0].set_ylabel("Compactness (isoperimetric ratio)")
    fig.suptitle(
        f"Shape compactness for thick-slice cases (eff_res >= {eff_res_threshold} mm)",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path))
    plt.close(fig)


# ============================================================
# Figure E: Qualitative comparison overlay
# ============================================================

def qualitative_figure(
    ct_path: Optional[str],
    pred_paths: Dict[str, str],
    pred_schemes: Dict[str, str],
    pred_offsets: Dict[str, int],
    gt_path: Optional[str],
    gt_scheme: Optional[str],
    gt_offset: int,
    out_path: Path,
    slice_idx: Optional[int] = None,
    view: str = "coronal",
) -> None:
    """Qualitative overlay: CT background + boundary contours from each arm.

    Columns: nnUNet raw pred | CNISP pred | arm B output | Proposed output | GT.
    """
    import nibabel as nib
    from simulation.evaluation.metrics import binary_structures

    column_order = ["nnUNet", "CNISP", "Cascade UNet", "Proposed", "GT"]
    struct_colors = {"Globe": "#e41a1c", "Optic nerve": "#377eb8",
                     "Recti": "#4daf4a", "Fat": "#ff7f00"}

    # Load CT
    ct_vol = None
    if ct_path and Path(ct_path).exists():
        ct_img = nib.load(str(ct_path))
        ct_vol = np.asarray(ct_img.dataobj).astype(float)

    # Load all prediction volumes
    volumes: Dict[str, np.ndarray] = {}
    for arm in column_order:
        if arm == "GT":
            p = gt_path
            scheme = gt_scheme
            offset = gt_offset
        else:
            p = pred_paths.get(arm)
            scheme = pred_schemes.get(arm)
            offset = pred_offsets.get(arm, 0)
        if p is None or not Path(p).exists():
            continue
        img = nib.load(str(p))
        data = np.asarray(img.dataobj)
        if offset:
            data = np.clip(data + offset, 0, None)
        volumes[arm] = data.astype(np.int32)

    if not volumes:
        return

    # Determine slice for display
    ref_vol = next(iter(volumes.values()))
    if view == "coronal":
        disp_axis = 1
    elif view == "sagittal":
        disp_axis = 0
    else:
        disp_axis = 2

    if slice_idx is None:
        # Pick a slice near the center of foreground
        fg = np.argwhere(ref_vol > 0)
        if fg.size > 0:
            slice_idx = int(np.median(fg[:, disp_axis]))
        else:
            slice_idx = ref_vol.shape[disp_axis] // 2

    n_cols = len([a for a in column_order if a in volumes])
    fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]

    col_i = 0
    for arm in column_order:
        if arm not in volumes:
            continue
        ax = axes[col_i]
        vol = volumes[arm]
        scheme = gt_scheme if arm == "GT" else pred_schemes.get(arm, "nnunet")

        # Extract slice
        slc = [slice(None)] * 3
        slc[disp_axis] = slice_idx
        seg_slice = vol[tuple(slc)]

        # CT background
        if ct_vol is not None:
            ct_slice = ct_vol[tuple(slc)]
            ax.imshow(ct_slice.T, cmap="gray", origin="lower",
                      vmin=-200, vmax=400, aspect="auto")
        else:
            ax.imshow(np.zeros_like(seg_slice.T), cmap="gray", origin="lower",
                      aspect="auto")

        # Boundary contours per structure
        masks = binary_structures(vol, scheme)
        for struct_name, struct_mask in masks.items():
            s_slice = struct_mask[tuple(slc)]
            if not s_slice.any():
                continue
            from scipy.ndimage import binary_erosion
            boundary = s_slice & ~binary_erosion(s_slice, iterations=1, border_value=0)
            if boundary.any():
                ax.contour(boundary.T, levels=[0.5], colors=[struct_colors.get(struct_name, "white")],
                           linewidths=1.2, origin="lower")

        ax.set_title(LEGEND.get(arm, arm), fontsize=9)
        ax.axis("off")
        col_i += 1

    fig.suptitle("Qualitative comparison (boundary contours on CT)", fontsize=11, y=1.02)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=600)
    plt.close(fig)


__all__ = [
    "topology_violation_figure", "continuity_figure",
    "compactness_figure", "qualitative_figure",
]
