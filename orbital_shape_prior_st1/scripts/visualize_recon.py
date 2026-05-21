#!/usr/bin/env python3
"""
Visualize the contents of a reconstruction output folder
(output_basedir/<model_name>/) produced by 03_infer.py.

Generates:
    recon_layout.txt   — file tree, per-step counts, key file sizes
    recon_summary.png  — composite figure with:
                            * test_summary.csv as a table
                            * mean-Dice trend by step
                            * cross-resolution heatmaps (if present)

Usage (run on the machine that holds the reconstruction folder):
    PYTHONPATH=. python3 scripts/visualize_recon.py \
        -p configs/paths.yaml -m orbital_ad_v2

    # or point directly at the folder:
    PYTHONPATH=. python3 scripts/visualize_recon.py \
        -d /home-local/linz18/CNISP/orbital_shape_prior_st1/reconstructions/orbital_ad_v2
"""

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import yaml


# ── tree dump ─────────────────────────────────────────────────────

_INTERESTING_SUFFIXES = {".pkl", ".csv", ".json", ".npy", ".nii.gz", ".png"}


def _human(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:6.1f} {unit}"
        size /= 1024
    return f"{size:6.1f} TB"


def _scan_step_dir(step_dir: Path) -> Dict[str, int]:
    """Count files in each step_XX/ subdir."""
    counts = {}
    for sub in ("pred", "latents", "obs_vs_recon", "iso_space"):
        d = step_dir / sub
        if d.exists():
            counts[sub] = sum(1 for _ in d.glob("*"))
    return counts


def build_layout(recon_dir: Path) -> str:
    """Return a textual tree of recon_dir."""
    lines: List[str] = []
    if not recon_dir.exists():
        return f"(recon dir does not exist: {recon_dir})"

    top_entries = sorted(recon_dir.iterdir(),
                         key=lambda p: (not p.is_file(), p.name))

    lines.append(f"{recon_dir}/")
    for entry in top_entries:
        if entry.is_file():
            size = entry.stat().st_size
            lines.append(f"├── {entry.name:<30s} {_human(size)}")
        else:
            lines.append(f"├── {entry.name}/")
            if entry.name.startswith("step_"):
                counts = _scan_step_dir(entry)
                meta = entry / "metadata.json"
                if meta.exists():
                    try:
                        with open(meta) as f:
                            md = json.load(f)
                        n_cases = len(md.get("cases", []))
                        lines.append(f"│    ({n_cases} cases in metadata.json)")
                    except (json.JSONDecodeError, KeyError):
                        pass
                for sub, n in counts.items():
                    lines.append(f"│   ├── {sub:<14s} {n} files")
            elif entry.name == "cross_resolution_analysis":
                pngs = sorted(entry.glob("*.png"))
                csvs = sorted(entry.glob("*.csv"))
                for p in pngs:
                    lines.append(f"│   ├── {p.name}")
                for c in csvs:
                    lines.append(f"│   ├── {c.name}")
            elif entry.name == "native_space":
                n = sum(1 for _ in entry.glob("*.nii.gz"))
                lines.append(f"│   ({n} *.nii.gz)")
    return "\n".join(lines)


# ── per-step stats ────────────────────────────────────────────────

def load_test_results_csv(path: Path) -> Optional[List[Dict]]:
    if not path.exists():
        return None
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def load_test_summary_csv(path: Path) -> Optional[List[Dict]]:
    if not path.exists():
        return None
    with open(path) as f:
        reader = csv.DictReader(f)
        return list(reader)


# Pickles larger than this skip full load and only report the size.
# inference_results.pkl can be multiple GB because it embeds dense volumes.
_PKL_LOAD_LIMIT_BYTES = 200 * 1024 * 1024  # 200 MB


def summarize_pkl(pkl_path: Path) -> Optional[Dict]:
    if not pkl_path.exists():
        return None
    size = pkl_path.stat().st_size
    if size > _PKL_LOAD_LIMIT_BYTES:
        return {"n_rows": None, "keys": None, "size": size, "skipped": True}
    try:
        with open(pkl_path, "rb") as f:
            rows = pickle.load(f)
    except Exception as e:  # pragma: no cover - diagnostic helper
        return {"n_rows": None, "keys": None, "size": size, "error": repr(e)}
    if not rows:
        return {"n_rows": 0, "keys": [], "size": size}
    keys = list(rows[0].keys()) if hasattr(rows[0], "keys") else None
    return {"n_rows": len(rows), "keys": keys, "size": size}


# ── plotting ──────────────────────────────────────────────────────

_CLASSES = ("ON", "Globe", "Fat", "Recti")
_CLASS_COLORS = {
    "ON": "#d62728",
    "Globe": "#1f77b4",
    "Fat": "#2ca02c",
    "Recti": "#9467bd",
}


def _parse_summary(summary_rows: List[Dict]):
    """Parse test_summary.csv rows into structured arrays keyed by step."""
    steps: List[int] = []
    eff_res: List[float] = []
    out: Dict[str, Dict[str, List[float]]] = {
        "dense": {c: [] for c in _CLASSES},
        "obs": {c: [] for c in _CLASSES},
        "dense_std": {c: [] for c in _CLASSES},
        "obs_std": {c: [] for c in _CLASSES},
    }
    mean_dense_m, mean_dense_s, mean_obs_m, mean_obs_s = [], [], [], []
    for r in summary_rows:
        try:
            s = int(float(r["step_size"]))
        except (KeyError, TypeError, ValueError):
            continue
        steps.append(s)
        eff_res.append(float(r.get("effective_resolution_mm", "nan")))
        mean_dense_m.append(float(r.get("dice_dense_mean", "nan")))
        mean_dense_s.append(float(r.get("dice_dense_std", "nan")))
        mean_obs_m.append(float(r.get("dice_observed_mean", "nan")))
        mean_obs_s.append(float(r.get("dice_observed_std", "nan")))
        for c in _CLASSES:
            out["dense"][c].append(float(r.get(f"{c}_dense_mean", "nan")))
            out["obs"][c].append(float(r.get(f"{c}_obs_mean", "nan")))
            out["dense_std"][c].append(float(r.get(f"{c}_dense_std", "nan")))
            out["obs_std"][c].append(float(r.get(f"{c}_obs_std", "nan")))
    return {
        "steps": steps,
        "eff_res": eff_res,
        "mean_dense_m": mean_dense_m,
        "mean_dense_s": mean_dense_s,
        "mean_obs_m": mean_obs_m,
        "mean_obs_s": mean_obs_s,
        **out,
    }


def _plot_mean_trend(ax, S):
    if not S["steps"]:
        ax.set_title("Mean Dice vs step  (missing)")
        ax.axis("off")
        return
    ax.errorbar(S["steps"], S["mean_dense_m"], yerr=S["mean_dense_s"],
                fmt="o-", capsize=4, color="#d62728", label="dense (all voxels)")
    ax.errorbar(S["steps"], S["mean_obs_m"], yerr=S["mean_obs_s"],
                fmt="s--", capsize=4, color="#1f77b4", label="observed (slice voxels)")
    # dense labels go BELOW their line; observed labels go ABOVE their line.
    # Where both numbers coincide (e.g. step=1 has identical dense and obs),
    # show a single combined annotation.
    coincide_eps = 1e-3
    for s, d, o in zip(S["steps"], S["mean_dense_m"], S["mean_obs_m"]):
        if abs(d - o) < coincide_eps:
            ax.annotate(f"{d:.3f}", (s, d), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=8,
                        color="dimgray")
        else:
            ax.annotate(f"{d:.3f}", (s, d), textcoords="offset points",
                        xytext=(0, -14), ha="center", fontsize=8,
                        color="#d62728")
            ax.annotate(f"{o:.3f}", (s, o), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=8,
                        color="#1f77b4")
    ax.set_xlabel("step_size")
    ax.set_ylabel("Dice")
    ax.set_title("Overall Dice — dense vs observed-only")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)


def _plot_per_class(ax, S, kind: str, title: str):
    """kind is 'dense' or 'obs'."""
    if not S["steps"]:
        ax.set_title(f"{title}  (missing)")
        ax.axis("off")
        return
    for c in _CLASSES:
        ax.errorbar(S["steps"], S[kind][c], yerr=S[f"{kind}_std"][c],
                    fmt="o-", capsize=3, color=_CLASS_COLORS[c], label=c)
    ax.set_xlabel("step_size")
    ax.set_ylabel("Dice")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=8, ncol=2)


def _plot_dice_distribution(ax, results_rows: List[Dict]):
    """Box-plot of per-case dense Dice vs step_size from test_results.csv."""
    if not results_rows:
        ax.set_title("Per-case Dice distribution  (missing)")
        ax.axis("off")
        return
    by_step: Dict[int, List[float]] = {}
    for r in results_rows:
        try:
            s = int(float(r["step_size"]))
            d = float(r["dice_dense_mean"])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isnan(d):
            continue
        by_step.setdefault(s, []).append(d)
    if not by_step:
        ax.set_title("Per-case Dice distribution  (no rows)")
        ax.axis("off")
        return
    steps = sorted(by_step.keys())
    data = [by_step[s] for s in steps]
    bp = ax.boxplot(data, positions=steps, widths=0.6,
                    patch_artist=True, showfliers=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#a6cee3")
        patch.set_alpha(0.7)
    for s, vals in zip(steps, data):
        ax.scatter([s] * len(vals), vals, s=8, color="#1f3a5f",
                   alpha=0.35, zorder=3)
        ax.annotate(f"n={len(vals)}", (s, max(vals)),
                    textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=8, color="gray")
    ax.set_xlabel("step_size")
    ax.set_ylabel("Dice (dense, per-case)")
    ax.set_title("Per-case dense Dice distribution")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)


def _embed_existing_png(ax, png_path: Path, title: str):
    ax.axis("off")
    if not png_path.exists():
        ax.set_title(f"{title}  (missing)")
        return
    img = mpimg.imread(str(png_path))
    ax.imshow(img)
    ax.set_title(title, fontsize=10)


def build_summary_png(recon_dir: Path, out_png: Path) -> None:
    summary_rows = load_test_summary_csv(recon_dir / "test_summary.csv") or []
    results_rows = load_test_results_csv(recon_dir / "test_results.csv") or []
    S = _parse_summary(summary_rows)

    cra_dir = recon_dir / "cross_resolution_analysis"
    heatmaps: List[Tuple[Path, str]] = []
    for name in ("cross_res_dice_mean", "cross_res_dice_ON",
                 "cross_res_dice_Globe", "cross_res_dice_Fat",
                 "cross_res_dice_Recti"):
        png = cra_dir / f"{name}.png"
        if png.exists():
            heatmaps.append((png, name.replace("cross_res_dice_", "")))

    # Row 0: overall mean trend  (full width)
    # Row 1: per-class dense  |  per-class observed
    # Row 2: per-case box-plot          (full width)
    # Row 3+: heatmaps (if any)
    n_heat = len(heatmaps)
    extra_rows = (n_heat + 2) // 3 if n_heat else 0
    nrows = 3 + extra_rows

    fig = plt.figure(figsize=(15, 3.8 * nrows))
    gs = fig.add_gridspec(nrows, 6, hspace=0.55, wspace=0.35)

    ax0 = fig.add_subplot(gs[0, :])
    _plot_mean_trend(ax0, S)

    ax1 = fig.add_subplot(gs[1, :3])
    _plot_per_class(ax1, S, "dense", "Per-class dense Dice vs step_size")

    ax2 = fig.add_subplot(gs[1, 3:])
    _plot_per_class(ax2, S, "obs",
                    "Per-class observed-only Dice vs step_size")

    ax3 = fig.add_subplot(gs[2, :])
    _plot_dice_distribution(ax3, results_rows)

    for i, (png, title) in enumerate(heatmaps):
        r = 3 + i // 3
        c0 = (i % 3) * 2
        ax = fig.add_subplot(gs[r, c0:c0 + 2])
        _embed_existing_png(ax, png, title)

    subtitle_bits: List[str] = []
    if S["steps"]:
        subtitle_bits.append(
            "step_size: " + ", ".join(str(s) for s in S["steps"])
        )
        subtitle_bits.append(
            "eff-res (mm): "
            + ", ".join(f"{r:.2f}" for r in S["eff_res"])
        )
    subtitle = "    |    ".join(subtitle_bits)
    fig.suptitle(
        f"Reconstruction summary — {recon_dir.name}"
        + (f"\n{subtitle}" if subtitle else ""),
        fontsize=13, fontweight="bold", y=0.995,
    )
    fig.savefig(str(out_png), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--paths", default=None,
                        help="paths.yaml; used with -m to resolve output_basedir/<model>")
    parser.add_argument("-m", "--model_name", default=None)
    parser.add_argument("-d", "--recon_dir", default=None,
                        help="Direct path to the reconstruction folder.")
    args = parser.parse_args()

    if args.recon_dir:
        recon_dir = Path(args.recon_dir).resolve()
    else:
        if not (args.paths and args.model_name):
            parser.error("Provide either -d or both -p and -m.")
        with open(args.paths) as f:
            paths = yaml.safe_load(f)
        recon_dir = Path(paths["output_basedir"]) / args.model_name

    if not recon_dir.exists():
        raise SystemExit(f"recon_dir does not exist: {recon_dir}")

    # ── tree ─────────────────────────────────────────────────────
    tree = build_layout(recon_dir)
    tree_path = recon_dir / "recon_layout.txt"
    with open(tree_path, "w") as f:
        f.write(tree + "\n")
    print(tree)
    print(f"\n[tree saved] {tree_path}")

    # ── pickle key check ─────────────────────────────────────────
    for pkl_name in ("inference_results.pkl", "sweep_results.pkl"):
        info = summarize_pkl(recon_dir / pkl_name)
        if info is None:
            print(f"  {pkl_name}: missing")
            continue
        size_str = _human(info["size"]) if "size" in info else "?"
        if info.get("skipped"):
            print(f"  {pkl_name}: {size_str} (load skipped — exceeds "
                  f"{_PKL_LOAD_LIMIT_BYTES // (1024*1024)} MB cap)")
        elif "error" in info:
            print(f"  {pkl_name}: {size_str}  load failed: {info['error']}")
        else:
            print(f"  {pkl_name}: {info['n_rows']} rows, "
                  f"size={size_str}, keys={info['keys']}")

    # ── summary PNG ──────────────────────────────────────────────
    out_png = recon_dir / "recon_summary.png"
    build_summary_png(recon_dir, out_png)
    print(f"[summary saved] {out_png}")


if __name__ == "__main__":
    main()
