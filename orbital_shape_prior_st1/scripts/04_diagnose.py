#!/usr/bin/env python3
"""
Step 4: Post-inference diagnostics.

Measures three aspects of model quality:

  1. RECONSTRUCTION QC — Per-case, per-structure:
     - Is the reconstruction in the right PLACE? (centroid shift in mm)
     - Is the reconstruction the right SIZE? (volume ratio pred/gt)
     - How much error is position vs shape? (aligned vs unaligned Dice gap)
     - Which structures fail? (per-structure breakdown)

  2. LATENT SPACE ANALYSIS — Population-level:
     - Latent norm distribution (detect collapse or explosion)
     - Inter-case similarity (t-SNE, pairwise distances)
     - Correlation with anatomical metadata (volume, side)

  3. CROSS-RESOLUTION CONSISTENCY — Resolution robustness:
     - Pairwise Dice between iso-space predictions at different step sizes
     - Heatmap: how much do predictions agree across resolutions?
     - Per-structure breakdown of cross-resolution agreement
     - Uses iso_space/ outputs from step 3 resolution sweep

Usage:
    python scripts/04_diagnose.py \
        -p configs/paths.yaml \
        -c configs/train.yaml \
        -t configs/test.yaml \
        -m orbital_ad_v2
"""

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from diagnostics.reconstruction_qc import (
    run_diagnostics, print_diagnostic_report,
)
from diagnostics.latent_analysis import (
    analyze_latent_space, print_latent_report,
)


# ── Cross-resolution utilities (self-contained, no engine imports) ─

def _dice(a: np.ndarray, b: np.ndarray, num_classes: int) -> Dict:
    per_class = []
    for c in range(1, num_classes):
        pa, pb = (a == c), (b == c)
        inter = np.sum(pa & pb)
        total = np.sum(pa) + np.sum(pb)
        per_class.append(2.0 * inter / (total + 1e-5))
    return {"mean": float(np.mean(per_class)), "per_class": per_class}


def _load_iso_predictions(output_dir: Path, step_sizes: List[int],
                          casenames: List[str]) -> Dict:
    import nibabel as nib
    preds = {}
    for step in step_sizes:
        iso_dir = output_dir / f"step_{step:02d}" / "iso_space"
        preds[step] = {}
        for cn in casenames:
            path = iso_dir / f"{cn}_pred_iso.nii.gz"
            if path.exists():
                preds[step][cn] = np.asarray(
                    nib.load(str(path)).dataobj
                ).astype(np.int32)
    return preds


def _get_casenames_from_metadata(output_dir: Path, step_sizes: List[int]) -> List[str]:
    for step in step_sizes:
        meta_path = output_dir / f"step_{step:02d}" / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            return [c["casename"] for c in meta["cases"]]
    return []


def _compute_pairwise_dice(preds, step_sizes, casenames, num_classes):
    n = len(step_sizes)
    per_case = np.full((len(casenames), n, n), np.nan)
    for ci, cn in enumerate(casenames):
        for si, s1 in enumerate(step_sizes):
            for sj, s2 in enumerate(step_sizes):
                if si == sj:
                    per_case[ci, si, sj] = 1.0
                    continue
                if cn not in preds[s1] or cn not in preds[s2]:
                    continue
                p1, p2 = preds[s1][cn], preds[s2][cn]
                if p1.shape != p2.shape:
                    ms = tuple(min(a, b) for a, b in zip(p1.shape, p2.shape))
                    p1, p2 = p1[:ms[0], :ms[1], :ms[2]], p2[:ms[0], :ms[1], :ms[2]]
                per_case[ci, si, sj] = _dice(p1, p2, num_classes)["mean"]
    return np.nanmean(per_case, axis=0), per_case


def _plot_heatmap(matrix, step_sizes, spacings_per_step, save_path, title,
                  struct_names=None, preds=None, casenames=None, num_classes=None):
    n = len(step_sizes)
    labels = [f"step={s}\n({spacings_per_step.get(s, 0):.1f}mm)" for s in step_sizes]

    fig, ax = plt.subplots(figsize=(2.0 + 0.9 * n, 1.6 + 0.9 * n))
    vmin = max(0.6, np.nanmin(matrix) - 0.02)
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=vmin, vmax=1.0,
                   interpolation="nearest", aspect="equal")
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            if np.isnan(val):
                continue
            color = "white" if val < (vmin + 1.0) / 2 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=color)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Resolution B", fontsize=11)
    ax.set_ylabel("Resolution A", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Dice", fontsize=10)
    plt.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_cross_resolution_analysis(output_dir: Path, step_sizes: List[int],
                                  num_classes: int):
    """
    Section 3: Cross-resolution consistency analysis.

    Reads iso_space predictions from step 3, computes pairwise Dice,
    saves heatmaps and CSV.
    """
    print(f"\n{'='*60}")
    print("SECTION 3: Cross-Resolution Consistency")
    print(f"{'='*60}")

    casenames = _get_casenames_from_metadata(output_dir, step_sizes)
    if not casenames:
        print("  No metadata found — skipping cross-resolution analysis.")
        return None

    # Check which steps actually have iso predictions
    available_steps = []
    for step in step_sizes:
        iso_dir = output_dir / f"step_{step:02d}" / "iso_space"
        if iso_dir.exists() and any(iso_dir.glob("*.nii.gz")):
            available_steps.append(step)
    if len(available_steps) < 2:
        print(f"  Need ≥2 steps with iso predictions, found {len(available_steps)} — skipping.")
        return None

    print(f"  Cases: {len(casenames)}")
    print(f"  Steps with iso predictions: {available_steps}")

    # Median effective resolution per step
    spacings_per_step = {}
    for step in available_steps:
        meta_path = output_dir / f"step_{step:02d}" / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            resolutions = [c["effective_through_plane_mm"] for c in meta["cases"]]
            spacings_per_step[step] = float(np.median(resolutions))
        else:
            spacings_per_step[step] = float("nan")

    # Load iso predictions
    print("  Loading iso predictions...")
    preds = _load_iso_predictions(output_dir, available_steps, casenames)
    for step in available_steps:
        print(f"    step={step}: {len(preds[step])}/{len(casenames)} loaded")

    # Pairwise Dice
    print("  Computing pairwise Dice...")
    matrix, per_case = _compute_pairwise_dice(
        preds, available_steps, casenames, num_classes,
    )

    # Print matrix
    print(f"\n  {'':>10s}", end="")
    for s in available_steps:
        print(f" step={s:>2d}", end="")
    print()
    for si, s1 in enumerate(available_steps):
        print(f"  step={s1:>2d}  ", end="")
        for sj in range(len(available_steps)):
            print(f"  {matrix[si, sj]:.3f}", end="")
        print()

    # Save plots and CSV
    analysis_dir = output_dir / "cross_resolution_analysis"
    analysis_dir.mkdir(exist_ok=True)

    # Mean heatmap
    _plot_heatmap(
        matrix, available_steps, spacings_per_step,
        analysis_dir / "cross_res_dice_mean.png",
        "Cross-Resolution Dice (mean)",
    )
    print(f"\n  Heatmap: {analysis_dir / 'cross_res_dice_mean.png'}")

    # Per-structure heatmaps
    struct_names = {1: "ON", 2: "Globe", 3: "Fat", 4: "Recti"}
    n = len(available_steps)
    for cls_id in range(1, num_classes):
        cls_matrix = np.full((len(casenames), n, n), np.nan)
        for ci, cn in enumerate(casenames):
            for si, s1 in enumerate(available_steps):
                for sj, s2 in enumerate(available_steps):
                    if si == sj:
                        cls_matrix[ci, si, sj] = 1.0
                        continue
                    if cn not in preds[s1] or cn not in preds[s2]:
                        continue
                    p1, p2 = preds[s1][cn], preds[s2][cn]
                    if p1.shape != p2.shape:
                        ms = tuple(min(a, b) for a, b in zip(p1.shape, p2.shape))
                        p1, p2 = p1[:ms[0], :ms[1], :ms[2]], p2[:ms[0], :ms[1], :ms[2]]
                    pa, pb = (p1 == cls_id), (p2 == cls_id)
                    inter = np.sum(pa & pb)
                    total = np.sum(pa) + np.sum(pb)
                    cls_matrix[ci, si, sj] = 2.0 * inter / (total + 1e-5)

        cls_mean = np.nanmean(cls_matrix, axis=0)
        name = struct_names.get(cls_id, f"class_{cls_id}")
        _plot_heatmap(
            cls_mean, available_steps, spacings_per_step,
            analysis_dir / f"cross_res_dice_{name}.png",
            f"Cross-Resolution Dice — {name}",
        )
    print(f"  Per-structure heatmaps: {analysis_dir}/")

    # CSV
    csv_path = analysis_dir / "pairwise_dice_matrix.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + [f"step_{s}" for s in available_steps])
        for si, s1 in enumerate(available_steps):
            w.writerow([f"step_{s1}"] + [f"{matrix[si, sj]:.4f}"
                                          for sj in range(n)])
    print(f"  Matrix CSV: {csv_path}")

    return {"matrix": matrix, "step_sizes": available_steps,
            "spacings_per_step": spacings_per_step}


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--paths", required=True)
    parser.add_argument("-m", "--model_name", required=True)
    parser.add_argument("-c", "--config", default=None, help="train config yaml")
    parser.add_argument("-t", "--test_config", default=None, help="test config yaml")
    args = parser.parse_args()

    with open(args.paths) as f:
        paths = yaml.safe_load(f)

    # Merge train + test configs if provided
    params = dict(paths)
    if args.config:
        with open(args.config) as f:
            params.update(yaml.safe_load(f) or {})
    if args.test_config:
        with open(args.test_config) as f:
            params.update(yaml.safe_load(f) or {})

    output_dir = Path(paths["output_basedir"]) / args.model_name

    # ── Section 1: Reconstruction diagnostics ─────────────────────
    results_path = output_dir / "inference_results.pkl"
    results = None
    if results_path.exists():
        with open(results_path, "rb") as f:
            results = pickle.load(f)

        print("\n" + "=" * 60)
        print("SECTION 1: Reconstruction QC")
        print("=" * 60)
        diags = run_diagnostics(results)
        print_diagnostic_report(diags)
    else:
        print(f"\nSKIP Section 1: {results_path} not found.")
        diags = None

    # ── Section 2: Latent space analysis ──────────────────────────
    if results:
        latents = np.array([r["latent"] for r in results if "latent" in r])
        if latents.shape[0] > 0:
            meta_dir = Path(paths["aligned_dir"]) / "metadata"
            metadata = []
            for r in results:
                meta_path = meta_dir / f"{r['casename']}.json"
                if meta_path.exists():
                    with open(meta_path) as f:
                        metadata.append(json.load(f))
                else:
                    metadata.append({})

            print("\n" + "=" * 60)
            print("SECTION 2: Latent Space Analysis")
            print("=" * 60)
            analysis = analyze_latent_space(latents, metadata)
            print_latent_report(analysis)
    else:
        print(f"\nSKIP Section 2: no inference results.")

    # ── Section 3: Cross-resolution consistency ───────────────────
    # With per-case adaptive sweep the union of step values is not in the
    # config. Discover step directories from disk and fall back to the
    # legacy ``test_step_sizes`` / ``slice_step_size`` keys only if no
    # ``step_XX/`` subdirectories exist yet.
    step_sizes = sorted(
        int(p.name.split("_")[1])
        for p in output_dir.glob("step_*")
        if p.is_dir() and p.name.split("_")[1].isdigit()
    )
    if not step_sizes:
        step_sizes = params.get("test_step_sizes",
                                [params.get("slice_step_size", 4)])
        if isinstance(step_sizes, int):
            step_sizes = [step_sizes]
    num_classes = params.get("num_classes", 5)

    cross_res = run_cross_resolution_analysis(output_dir, step_sizes, num_classes)

    # ── Save combined report ──────────────────────────────────────
    report = {"model_name": args.model_name}

    if diags:
        report["reconstruction_qc"] = {
            "n_cases": len(diags),
            "per_case": [{
                "casename": d.casename,
                "mean_dice_unaligned": d.mean_dice_unaligned,
                "mean_dice_aligned": d.mean_dice_aligned,
                "position_contribution": d.mean_position_contribution,
                "per_structure": d.per_structure,
            } for d in diags],
        }

    if cross_res:
        report["cross_resolution"] = {
            "step_sizes": cross_res["step_sizes"],
            "spacings_per_step": cross_res["spacings_per_step"],
            "pairwise_dice_matrix": cross_res["matrix"].tolist(),
        }

    report_path = output_dir / "diagnostic_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\nFull diagnostic report: {report_path}")


if __name__ == "__main__":
    main()