"""
CNISP-only result viewer for orbital shape prior inference.

This module owns the artifacts that are **specific to CNISP** and that
``nnunet/engine/build_method_summary.py`` cannot reproduce:

    recon_layout.txt            file-tree summary of the reconstruction dir
                                (folder structure + file counts + sizes)
    cross_resolution_analysis/  pairwise hard-Dice between iso-space
                                predictions at every step pair (mean +
                                per-structure heatmaps + CSV). This is
                                a *prior self-consistency* metric --
                                Dice between CNISP's own predictions at
                                different sparsities, on its canonical
                                iso patch grid. nnUNet has no analogue.
    native_sweep_summary.json   per-step audit of native_space_step_XX/
                                outputs (source coverage, file presence).
    stdout pickle summary       inference_results.pkl / sweep_results.pkl
                                size + row-count introspection.

What this module deliberately does NOT do anymore
-------------------------------------------------
Per-step Dice trend lines, per-class Dice curves and per-case Dice
boxplots used to be packed into a single ``recon_summary.png`` here.
That bundle is now produced by ``nnunet/engine/build_method_summary.py``
(driven from ``paired_per_source.csv``) so that CNISP and nnUNet always
share one source set + eff_res axis. The output for CNISP lands at
``${output_basedir}/<model_name>/viz/CNISP_recon_summary.png`` and
sibling files.

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
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


# ── Constants ────────────────────────────────────────────────────

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
    """Find the reconstruction directory for this visualization invocation.

    Precedence:
      1. Explicit ``params['recon_dir']`` wins (the legacy -d flag).
      2. ``output_basedir/<model_name>/runs/<run_tag>/`` (Option C
         layout). ``run_tag`` defaults to ``atlas_gt`` so a stock test
         config keeps pointing at the ceiling-curve run.
      3. Backwards compatibility: if the runs/<run_tag>/ directory does
         not exist on disk (e.g. legacy runs that pre-date the run-tag
         layout), fall back to ``output_basedir/<model_name>/``.
    """
    if "recon_dir" in params:
        return Path(params["recon_dir"]).resolve()
    base = params.get("output_basedir")
    name = params.get("model_name")
    if not base or not name:
        raise ValueError(
            "visualize: provide either params['recon_dir'] or both "
            "params['output_basedir'] and params['model_name']."
        )
    run_tag = str(params.get("run_tag", "atlas_gt"))
    tagged = Path(base) / name / "runs" / run_tag
    if tagged.exists():
        return tagged
    legacy = Path(base) / name
    if legacy.exists():
        print(f"  [visualize] runs/{run_tag}/ not found; falling back to "
              f"legacy layout at {legacy}")
        return legacy
    # Neither layout exists yet -- caller will get the "recon_dir does
    # not exist" message they expect.
    return tagged


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

    # ── Cross-resolution iso-space pairwise heatmaps ─────────────
    # CNISP-only analytic: pairwise hard Dice between predictions at
    # every (step_A, step_B) pair, on the canonical iso patch grid.
    # Measures the prior's resolution invariance (no GT involved).
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

    # ── Native sweep audit ───────────────────────────────────────
    native_sweep = audit_native_sweep(recon_dir)
    if native_sweep is None:
        print("  Skip native sweep audit: no native_space_step_*/ "
              "directories found.")

    print("\nDice trend / per-class / per-case figures are produced by the\n"
          "`compare` phase (nnunet/engine/build_method_summary.py); they\n"
          "land at <output_basedir>/<model>/viz/<run_tag>/"
          "<method_label>_recon_summary.png and siblings.")

    return {
        "recon_dir": str(recon_dir),
        "tree": str(tree_path),
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
