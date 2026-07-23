"""Per-method cross-resolution Dice heatmaps for the comparison figures.

Reuses the EXACT heatmap + pairwise-Dice functions from the CNISP
``engine/visualize.py`` (``_plot_heatmap`` / ``_compute_pairwise_dice`` /
``_dice_per_class``), pulled in by file path so this does not trigger the
``orbital_shape_prior_st1.engine`` package __init__ (tensorboard) chain -- those
functions only need numpy + matplotlib.

Where ``engine/visualize.run_cross_resolution_analysis`` works on a single CNISP
recon folder (iso_space/ per step), this module drives the SAME core from the
evaluation MASK_INDEX so every arm (nnUNet / Cascade / CNISP / Proposed / Oracle)
gets its own cross-resolution heatmaps -- i.e. "how self-consistent is each
method's segmentation as the input resolution changes." Masks are loaded per
(arm, source, step), resampled onto the source's reference grid (order 0), and
remapped to a canonical {Globe,Optic nerve,Recti,Fat}={1,2,3,4} label scheme so
the pairwise Dice is comparable across arms with different native schemes.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from simulation.evaluation.metrics import binary_structures, STRUCTURES, METHODS

_REPO = Path(__file__).resolve().parents[2]
_VIS_PY = _REPO / "orbital_shape_prior_st1" / "engine" / "visualize.py"

_NUM_CLASSES = len(STRUCTURES) + 1          # background + 4 structures
_CANON_SCHEME = {s: i for i, s in enumerate(STRUCTURES, start=1)}


def _load_vis():
    """Import engine/visualize.py by path (numpy+matplotlib only; no engine __init__)."""
    spec = importlib.util.spec_from_file_location("cnisp_visualize", _VIS_PY)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_VIS = _load_vis()
_plot_heatmap = _VIS._plot_heatmap                 # (matrix, steps, spacings, path, title)
_compute_pairwise_dice = _VIS._compute_pairwise_dice   # (preds, steps, cases, num_classes)


# ── mask loading (per arm/source/step) ───────────────────────────────────────
def _canon_labels(data: np.ndarray, scheme: str) -> np.ndarray:
    """Remap a label array (in ``scheme``) to the canonical 1..4 STRUCTURES ints."""
    bs = binary_structures(data, scheme)
    out = np.zeros(data.shape, dtype=np.int32)
    for s in STRUCTURES:
        out[bs[s]] = _CANON_SCHEME[s]
    return out


def _load_canon(entry: Dict, ref):
    """Load one MASK_INDEX entry -> canonical-label int array on the reference grid.

    ``ref`` is ``None`` (this entry defines the grid) or ``(shape, affine)`` to
    resample onto (nearest / order 0, world-coordinate aligned).
    """
    import nibabel as nib
    from nibabel.processing import resample_from_to

    img = nib.load(str(entry["pred_path"]))
    data = np.asarray(img.dataobj)
    off = int(entry.get("offset_pred", 0))
    if off:
        data = np.clip(data + off, 0, None)
    data = data.astype(np.int32)
    affine = np.asarray(img.affine, dtype=float)
    if ref is not None:
        rshape, raffine = ref
        if data.shape != tuple(rshape) or not np.allclose(affine, raffine, atol=1e-3):
            src = nib.Nifti1Image(data.astype(np.int16), affine)
            res = resample_from_to(src, (tuple(int(x) for x in rshape), raffine),
                                   order=0, mode="constant", cval=0)
            data = np.asarray(res.dataobj).astype(np.int32)
        return _canon_labels(data, entry["pred_scheme"]), ref
    return _canon_labels(data, entry["pred_scheme"]), (data.shape, affine)


# ── per-arm aggregation ──────────────────────────────────────────────────────
def _group(index: List[Dict]) -> Dict[str, Dict[str, Dict[int, Dict]]]:
    """arm -> source_id -> step -> entry (skips the GT reference arm)."""
    out: Dict[str, Dict[str, Dict[int, Dict]]] = {}
    for it in index:
        arm = it.get("arm")
        if arm not in METHODS:
            continue
        out.setdefault(arm, {}).setdefault(str(it["case"]), {})[int(it["step"])] = it
    return out


def arm_matrices(index: List[Dict], min_steps: int = 2, verbose: bool = True):
    """Compute the mean per-class cross-resolution Dice tensor for every arm.

    Returns ``{arm: {"mean_per_class": [K,M,M], "steps": [M], "eff_by_step": {step:mm},
    "n_sources": int}}`` for arms with >= ``min_steps`` shared steps.
    """
    grouped = _group(index)
    results = {}
    for arm in METHODS:
        by_src = grouped.get(arm, {})
        steps = sorted({s for stepmap in by_src.values() for s in stepmap})
        if len(steps) < min_steps:
            if verbose:
                print(f"  [{arm}] only steps {steps}; need >= {min_steps}. skip")
            continue
        # sources that have at least two of these steps
        usable = [sid for sid, sm in by_src.items() if len(sm) >= min_steps]
        if not usable:
            if verbose:
                print(f"  [{arm}] no source has >= {min_steps} steps. skip")
            continue
        cases = sorted(usable)
        preds = {s: {} for s in steps}
        eff_acc = {s: [] for s in steps}
        for sid in cases:
            sm = by_src[sid]
            ref = None
            for s in steps:                       # smallest step defines the ref grid
                if s in sm:
                    _, ref = _load_canon(sm[s], None)
                    break
            for s in steps:
                if s not in sm:
                    continue
                canon, _ = _load_canon(sm[s], ref)
                preds[s][sid] = canon
                e = sm[s].get("eff_res")
                if e is not None:
                    eff_acc[s].append(float(e))
        mean_per_class, _pc = _compute_pairwise_dice(preds, steps, cases, _NUM_CLASSES)
        eff_by_step = {s: (float(np.mean(v)) if v else 0.0) for s, v in eff_acc.items()}
        results[arm] = {"mean_per_class": mean_per_class, "steps": steps,
                        "eff_by_step": eff_by_step, "n_sources": len(cases)}
        if verbose:
            print(f"  [{arm}] {len(cases)} source(s) x steps {steps}  "
                  f"mean cross-res Dice={np.nanmean(mean_per_class):.3f}")
    return results


# ── rendering ────────────────────────────────────────────────────────────────
def _by_method_overview(results: Dict, out_path: Path) -> None:
    """One row of mean (over structures) cross-resolution heatmaps, one per arm."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = [m for m in METHODS if m in results]
    if not arms:
        return
    fig, axes = plt.subplots(1, len(arms), figsize=(3.1 * len(arms) + 0.6, 3.4))
    if len(arms) == 1:
        axes = [axes]
    im = None
    for ax, arm in zip(axes, arms):
        r = results[arm]
        mat = np.nanmean(r["mean_per_class"], axis=0)   # mean over structures [M,M]
        steps = r["steps"]
        im = ax.imshow(mat, cmap="RdYlGn", vmin=0.6, vmax=1.0,
                       interpolation="nearest", aspect="equal")
        n = len(steps)
        for i in range(n):
            for j in range(n):
                v = mat[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                            color="white" if v < 0.8 else "black")
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        labs = [f"{s}\n{r['eff_by_step'].get(s, 0):.1f}mm" for s in steps]
        ax.set_xticklabels(labs, fontsize=7); ax.set_yticklabels(labs, fontsize=7)
        ax.set_title(f"{arm}\n(n={r['n_sources']})", fontsize=9)
    fig.suptitle("Cross-resolution self-consistency (mean structure Dice between "
                 "resolutions) per method", fontsize=10.5)
    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.75, pad=0.02, label="Dice")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def render(index: List[Dict], out_dir: Path, min_steps: int = 2,
           verbose: bool = True) -> Dict:
    """Full render: per-arm mean + per-structure heatmaps + the by-method overview.

    Writes under ``<out_dir>/cross_resolution/``:
        by_method_overview.png
        <arm>/cross_res_dice_mean.png
        <arm>/cross_res_dice_<structure>.png
        <arm>/cross_res_dice_matrix.csv
    Returns the ``arm_matrices`` result dict.
    """
    out_dir = Path(out_dir)
    results = arm_matrices(index, min_steps=min_steps, verbose=verbose)
    root = out_dir / "cross_resolution"
    root.mkdir(parents=True, exist_ok=True)
    for arm, r in results.items():
        adir = root / arm.replace(" ", "_")
        adir.mkdir(parents=True, exist_ok=True)
        steps, eff = r["steps"], r["eff_by_step"]
        mean_over_struct = np.nanmean(r["mean_per_class"], axis=0)
        _plot_heatmap(mean_over_struct, steps, eff, adir / "cross_res_dice_mean.png",
                      f"Cross-resolution Dice — {arm} (mean, n={r['n_sources']})")
        for k, s in enumerate(STRUCTURES):
            _plot_heatmap(r["mean_per_class"][k], steps, eff,
                          adir / f"cross_res_dice_{s.replace(' ', '_')}.png",
                          f"Cross-resolution Dice — {arm} / {s}")
        _write_matrix_csv(adir / "cross_res_dice_matrix.csv", r, steps)
    _by_method_overview(results, root / "by_method_overview.png")
    if verbose:
        print(f"[cross_resolution] wrote {len(results)} arm(s) -> {root}/")
    return results


def _write_matrix_csv(path: Path, r: Dict, steps: List[int]) -> None:
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["structure", "step_a", "step_b", "dice"])
        for k, s in enumerate(["mean"] + STRUCTURES):
            mat = (np.nanmean(r["mean_per_class"], axis=0) if s == "mean"
                   else r["mean_per_class"][k - 1])
            for i, sa in enumerate(steps):
                for j, sb in enumerate(steps):
                    v = mat[i, j]
                    if not np.isnan(v):
                        w.writerow([s, sa, sb, f"{v:.5f}"])
