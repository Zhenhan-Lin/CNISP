"""
Result visualization & summary for orbital shape prior inference.

This module is a *result viewer* — it does not interpret model behavior
(no reconstruction QC, no latent-space analysis). It reads the folder
produced by `engine.infer.infer_test_set` and emits:

    recon_layout.txt            file-tree summary of the reconstruction dir
    recon_summary.png           per-step Dice trends, per-class breakdowns
                                and per-case Dice distribution; any
                                cross-resolution heatmaps that exist are
                                embedded alongside.
    cross_resolution_analysis/  pairwise hard-Dice between iso-space
                                predictions at every step pair (mean +
                                per-structure heatmaps + CSV).
    native_sweep_summary.json   per-step audit of native_space_step_XX/
                                outputs (source coverage, file presence).

Usage from a script:
    from engine.visualize import visualize_results
    visualize_results(params)   # params merges paths + train + test yaml

Where `params` provides:
    output_basedir : str       parent of <model_name>/
    model_name     : str
    num_classes    : int       (optional, defaults to 5)
"""

import csv
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402  (must come after backend)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


# ── Constants ────────────────────────────────────────────────────

_CLASSES = ("ON", "Globe", "Fat", "Recti")
_CLASS_COLORS = {
    "ON": "#d62728",
    "Globe": "#1f77b4",
    "Fat": "#2ca02c",
    "Recti": "#9467bd",
}
_STRUCT_NAMES = {1: "ON", 2: "Globe", 3: "Fat", 4: "Recti"}

# Pickles larger than this skip full load and only report their size.
# inference_results.pkl can be multiple GB because it embeds dense volumes.
_PKL_LOAD_LIMIT_BYTES = 200 * 1024 * 1024  # 200 MB


# ── Filesystem layout dump ───────────────────────────────────────

def _human(size: int) -> str:
    s = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if s < 1024:
            return f"{s:6.1f} {unit}"
        s /= 1024
    return f"{s:6.1f} TB"


def _scan_step_dir(step_dir: Path) -> Dict[str, int]:
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
            lines.append(f"├── {entry.name:<32s} {_human(size)}")
            continue

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
        elif entry.name.startswith("native_space_step_"):
            n_nii = sum(1 for _ in entry.glob("*.nii.gz"))
            mani = entry / "manifest.json"
            extra = ""
            if mani.exists():
                try:
                    with open(mani) as f:
                        info = json.load(f)
                    extra = f", manifest n_sources={info.get('n_sources', '?')}"
                except (json.JSONDecodeError, KeyError):
                    extra = ", manifest unreadable"
            lines.append(f"│   ({n_nii} *.nii.gz{extra})")
        elif entry.name == "native_space":
            n = sum(1 for _ in entry.glob("*.nii.gz"))
            lines.append(f"│   ({n} *.nii.gz)")
        elif entry.name == "cross_resolution_analysis":
            for p in sorted(entry.glob("*.png")):
                lines.append(f"│   ├── {p.name}")
            for c in sorted(entry.glob("*.csv")):
                lines.append(f"│   ├── {c.name}")
    return "\n".join(lines)


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


# ── CSV readers ──────────────────────────────────────────────────

def _read_csv(path: Path) -> Optional[List[Dict]]:
    if not path.exists():
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


# ── Summary parsing & plots ──────────────────────────────────────

def _parse_summary(summary_rows: List[Dict]) -> Dict:
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
                fmt="s--", capsize=4, color="#1f77b4",
                label="observed (slice voxels)")
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
    summary_rows = _read_csv(recon_dir / "test_summary.csv") or []
    results_rows = _read_csv(recon_dir / "test_results.csv") or []
    S = _parse_summary(summary_rows)

    cra_dir = recon_dir / "cross_resolution_analysis"
    heatmaps: List[Tuple[Path, str]] = []
    for name in ("cross_res_dice_mean", "cross_res_dice_ON",
                 "cross_res_dice_Globe", "cross_res_dice_Fat",
                 "cross_res_dice_Recti"):
        png = cra_dir / f"{name}.png"
        if png.exists():
            heatmaps.append((png, name.replace("cross_res_dice_", "")))

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


# ── Cross-resolution heatmaps (iso-space consistency) ────────────

def _dice_per_class(a: np.ndarray, b: np.ndarray,
                    num_classes: int) -> np.ndarray:
    per_class = np.empty(num_classes - 1, dtype=np.float64)
    for c in range(1, num_classes):
        pa, pb = (a == c), (b == c)
        inter = np.sum(pa & pb)
        total = np.sum(pa) + np.sum(pb)
        per_class[c - 1] = 2.0 * inter / (total + 1e-5)
    return per_class


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


def _get_casenames_from_metadata(output_dir: Path,
                                 step_sizes: List[int]) -> List[str]:
    for step in step_sizes:
        meta_path = output_dir / f"step_{step:02d}" / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            return [c["casename"] for c in meta["cases"]]
    return []


def _compute_pairwise_dice(preds, step_sizes, casenames, num_classes):
    n = len(casenames)
    m = len(step_sizes)
    k = num_classes - 1
    per_case = np.full((n, k, m, m), np.nan, dtype=np.float64)

    for ci, cn in enumerate(casenames):
        for si, s1 in enumerate(step_sizes):
            if cn not in preds[s1]:
                continue
            p1 = preds[s1][cn]
            for sj in range(si, m):
                s2 = step_sizes[sj]
                if si == sj:
                    per_case[ci, :, si, sj] = 1.0
                    continue
                if cn not in preds[s2]:
                    continue
                p2 = preds[s2][cn]
                # Same case + same native spacing -> same iso_target,
                # regardless of step (see engine/infer.py iso_space block).
                # If they don't match, refuse to silently truncate.
                assert p1.shape == p2.shape, (
                    f"iso shape mismatch for {cn} step={s1} vs step={s2}: "
                    f"{p1.shape} vs {p2.shape}. Re-run inference for this "
                    f"case so iso_space outputs share a common grid."
                )
                dice_k = _dice_per_class(p1, p2, num_classes)
                per_case[ci, :, si, sj] = dice_k
                per_case[ci, :, sj, si] = dice_k  # symmetric

    mean_per_class = np.nanmean(per_case, axis=0)  # [K, M, M]
    return mean_per_class, per_case


def _plot_heatmap(matrix, step_sizes, spacings_per_step, save_path, title):
    n = len(step_sizes)
    labels = [f"step={s}\n({spacings_per_step.get(s, 0):.1f}mm)"
             for s in step_sizes]

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
                                  num_classes: int) -> Optional[Dict]:
    """
    Pairwise agreement between iso-space predictions at every step pair.
    Symmetric, model-vs-model (no GT involved) so this is a measure of the
    prior's resolution invariance, not absolute accuracy.
    """
    print(f"\n{'='*60}")
    print("Cross-resolution consistency (iso-space pairwise Dice)")
    print(f"{'='*60}")

    casenames = _get_casenames_from_metadata(output_dir, step_sizes)
    if not casenames:
        print("  No metadata.json — skipping cross-resolution analysis.")
        return None

    available_steps = []
    for step in step_sizes:
        iso_dir = output_dir / f"step_{step:02d}" / "iso_space"
        if iso_dir.exists() and any(iso_dir.glob("*.nii.gz")):
            available_steps.append(step)
    if len(available_steps) < 2:
        print(f"  Need ≥2 steps with iso predictions, found "
              f"{len(available_steps)} — skipping.")
        return None

    print(f"  Cases: {len(casenames)}")
    print(f"  Steps with iso predictions: {available_steps}")

    spacings_per_step: Dict[int, float] = {}
    for step in available_steps:
        meta_path = output_dir / f"step_{step:02d}" / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            resolutions = [c["effective_through_plane_mm"]
                           for c in meta["cases"]]
            spacings_per_step[step] = float(np.median(resolutions))
        else:
            spacings_per_step[step] = float("nan")

    print("  Loading iso predictions...")
    preds = _load_iso_predictions(output_dir, available_steps, casenames)
    for step in available_steps:
        print(f"    step={step}: {len(preds[step])}/{len(casenames)} loaded")

    print("  Computing pairwise Dice...")
    mean_per_class, _ = _compute_pairwise_dice(
        preds, available_steps, casenames, num_classes,
    )
    matrix = mean_per_class.mean(axis=0)  # mean over foreground classes

    print(f"\n  {'':>10s}", end="")
    for s in available_steps:
        print(f" step={s:>2d}", end="")
    print()
    for si, s1 in enumerate(available_steps):
        print(f"  step={s1:>2d}  ", end="")
        for sj in range(len(available_steps)):
            print(f"  {matrix[si, sj]:.3f}", end="")
        print()

    analysis_dir = output_dir / "cross_resolution_analysis"
    analysis_dir.mkdir(exist_ok=True)

    _plot_heatmap(
        matrix, available_steps, spacings_per_step,
        analysis_dir / "cross_res_dice_mean.png",
        "Cross-Resolution Dice (mean)",
    )
    print(f"\n  Mean heatmap: {analysis_dir / 'cross_res_dice_mean.png'}")

    for cls_id in range(1, num_classes):
        name = _STRUCT_NAMES.get(cls_id, f"class_{cls_id}")
        _plot_heatmap(
            mean_per_class[cls_id - 1], available_steps, spacings_per_step,
            analysis_dir / f"cross_res_dice_{name}.png",
            f"Cross-Resolution Dice — {name}",
        )
    print(f"  Per-structure heatmaps: {analysis_dir}/")

    csv_path = analysis_dir / "pairwise_dice_matrix.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + [f"step_{s}" for s in available_steps])
        for si, s1 in enumerate(available_steps):
            w.writerow([f"step_{s1}"]
                       + [f"{matrix[si, sj]:.4f}"
                          for sj in range(len(available_steps))])
    print(f"  Matrix CSV: {csv_path}")

    return {
        "matrix": matrix,
        "per_class_matrix": mean_per_class,
        "step_sizes": available_steps,
        "spacings_per_step": spacings_per_step,
    }


# ── Native sweep audit ──────────────────────────────────────────

def audit_native_sweep(recon_dir: Path) -> Optional[Dict]:
    """
    Walk every native_space_step_XX/ subdirectory and verify the manifest
    matches the .nii.gz files actually on disk. Returns a per-step summary
    and writes native_sweep_summary.json next to it.
    """
    step_dirs = sorted(p for p in recon_dir.glob("native_space_step_*")
                       if p.is_dir())
    if not step_dirs:
        return None

    summary: Dict[str, Dict] = {}
    for sd in step_dirs:
        step_token = sd.name.replace("native_space_step_", "")
        try:
            step_int = int(step_token)
        except ValueError:
            continue
        on_disk = sorted(p.name for p in sd.glob("*.nii.gz"))
        manifest_path = sd / "manifest.json"
        manifest: Dict = {}
        missing: List[str] = []
        extra: List[str] = []
        if manifest_path.exists():
            try:
                with open(manifest_path) as f:
                    manifest = json.load(f)
            except json.JSONDecodeError:
                manifest = {}
            declared = {Path(p).name
                        for p in manifest.get("by_source_id", {}).values()}
            on_disk_set = set(on_disk)
            missing = sorted(declared - on_disk_set)
            extra = sorted(on_disk_set - declared)
        summary[f"step_{step_int:02d}"] = {
            "dir": str(sd),
            "n_files_on_disk": len(on_disk),
            "n_sources_in_manifest":
                len(manifest.get("by_source_id", {})),
            "missing_vs_manifest": missing,
            "extra_vs_manifest": extra,
        }
    out_path = recon_dir / "native_sweep_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Native-sweep audit: {out_path}")
    for step, info in summary.items():
        ok = (not info["missing_vs_manifest"]
              and info["n_files_on_disk"] == info["n_sources_in_manifest"])
        flag = "OK" if ok else "WARN"
        print(f"    {step}: {info['n_files_on_disk']} files, "
              f"{info['n_sources_in_manifest']} manifest entries  [{flag}]")
    return summary


# ── Top-level entry point ────────────────────────────────────────

def resolve_recon_dir(params: dict) -> Path:
    if "recon_dir" in params:
        return Path(params["recon_dir"]).resolve()
    base = params.get("output_basedir")
    name = params.get("model_name")
    if not base or not name:
        raise ValueError(
            "visualize: provide either params['recon_dir'] or both "
            "params['output_basedir'] and params['model_name']."
        )
    return Path(base) / name


def visualize_results(params: dict) -> Dict:
    """
    Generate the full result-summary bundle for one reconstruction folder.

    Returns a dict of artifact paths/metadata for use by the calling script.
    """
    recon_dir = resolve_recon_dir(params)
    if not recon_dir.exists():
        raise SystemExit(f"recon_dir does not exist: {recon_dir}")

    print("=" * 60)
    print(f"Result visualization — {recon_dir}")
    print("=" * 60)

    # ── Tree dump ────────────────────────────────────────────────
    tree = build_layout(recon_dir)
    tree_path = recon_dir / "recon_layout.txt"
    with open(tree_path, "w") as f:
        f.write(tree + "\n")
    print(tree)
    print(f"\n[tree saved] {tree_path}")

    # ── Pickle inspection ────────────────────────────────────────
    pkl_info: Dict[str, Optional[Dict]] = {}
    for pkl_name in ("inference_results.pkl", "sweep_results.pkl"):
        info = summarize_pkl(recon_dir / pkl_name)
        pkl_info[pkl_name] = info
        if info is None:
            print(f"  {pkl_name}: missing")
            continue
        size_str = _human(info["size"]) if "size" in info else "?"
        if info.get("skipped"):
            print(f"  {pkl_name}: {size_str} (load skipped — exceeds "
                  f"{_PKL_LOAD_LIMIT_BYTES // (1024 * 1024)} MB cap)")
        elif "error" in info:
            print(f"  {pkl_name}: {size_str}  load failed: {info['error']}")
        else:
            print(f"  {pkl_name}: {info['n_rows']} rows, "
                  f"size={size_str}, keys={info['keys']}")

    # ── Cross-resolution heatmaps (must come before summary PNG so
    #     the heatmap thumbnails get embedded into recon_summary.png) ─
    step_sizes = sorted(
        int(p.name.split("_")[1])
        for p in recon_dir.glob("step_*")
        if p.is_dir() and p.name.split("_")[1].isdigit()
    )
    num_classes = int(params.get("num_classes", 5))
    cross_res = None
    if step_sizes:
        cross_res = run_cross_resolution_analysis(
            recon_dir, step_sizes, num_classes,
        )
    else:
        print("\n  Skip cross-resolution: no step_XX/ directories found.")

    # ── Summary PNG ──────────────────────────────────────────────
    out_png = recon_dir / "recon_summary.png"
    build_summary_png(recon_dir, out_png)
    print(f"[summary saved] {out_png}")

    # ── Native sweep audit ───────────────────────────────────────
    native_sweep = audit_native_sweep(recon_dir)
    if native_sweep is None:
        print("  Skip native sweep audit: no native_space_step_*/ "
              "directories found.")

    return {
        "recon_dir": str(recon_dir),
        "tree": str(tree_path),
        "summary_png": str(out_png),
        "pkl_info": pkl_info,
        "cross_resolution": (
            {
                "step_sizes": cross_res["step_sizes"],
                "spacings_per_step": cross_res["spacings_per_step"],
                "pairwise_dice_matrix": cross_res["matrix"].tolist(),
            }
            if cross_res is not None else None
        ),
        "native_sweep": native_sweep,
    }
