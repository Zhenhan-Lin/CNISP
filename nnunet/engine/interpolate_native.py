#!/usr/bin/env python3
"""nnUNet-only Taubin post-processing control (mask gen + native-space Dice).

This is the calculation + orchestration layer behind the thin CLI
``nnunet/interpolate_native.py``. It embeds the user's windowed-sinc
(Taubin) surface-smoothing baseline and wires it into the sparse-CT sweep:

For each ``(source, step)`` recorded in
``prediction/<exp>/sweep_manifest.json``:

1. Taubin-smooth the DEGRADED-grid nnUNet prediction
   ``prediction/<exp>/sparse_step_<tag>/{sid}.nii.gz`` on its OWN
   (identity-IJK) voxel grid (one-hot -> DiscreteFlyingEdges ->
   WindowedSinc -> stencil re-voxelize, merged by a signed-distance argmax).
2. Resample the smoothed mask (``order=0``, world-aware) onto the NATIVE CT
   grid -- the grid of the sibling ``sparse_step_<tag>_native/{sid}.nii.gz``
   -- so it can be Diced against the native GT.
3. Write ``prediction/<exp>/interpolation/sparse_step_<tag>/{sid}.nii.gz``.

``step_01`` (the dense baseline) has no degraded input: it is Taubin-smoothed
directly on the native grid (no resample).

The smoothed native masks are then Diced two ways, sharing the exact Dice
path of the nnUNet native rows (``lib.metrics.compute_nnunet_native_rows``):

* ``summarize`` -- a STANDALONE per-step bundle (CSV/CSV/CSV + 2 PNGs) under
  ``prediction/<exp>/interpolation/summary/``, mirroring
  ``build_nnunet_native_summary.py``.
* ``compare_native`` reads the same ``interpolation/`` masks to add an
  ``nnUNet-interp`` column to the head-to-head paired table.

This control is nnUNet-only: it never reads or modifies any CNISP mask.

Requires ``vtk`` and ``scipy`` at runtime (CPU only; no nnUNet inference).

Usage
-----
    python nnunet/interpolate_native.py --config nnunet/configs.yaml \\
        --experiment thick --mode all
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np

# Make ``nnunet.*`` importable when imported standalone.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nnunet.helpers.buckets import (  # noqa: E402
    DEFAULT_BUCKET_EDGES_MM,
    NNUNET_INTERP_METHOD_LABEL,
)
from nnunet.helpers.config import load_yaml  # noqa: E402
from nnunet.helpers.paired_csv import resolve_source_prefix_filters  # noqa: E402
from nnunet.lib.metrics import (  # noqa: E402
    compute_nnunet_native_rows,
    eff_res_from_sparse_manifest,
    resample_pred_onto_gt,
    resolve_test_sources,
)
from nnunet.lib.viz import (  # noqa: E402
    aggregate_native_by_eff_res,
    aggregate_native_by_step,
    plot_native_dice_vs_eff_res,
    plot_native_dice_vs_step,
    write_native_by_eff_res_csv,
    write_native_by_step_csv,
    write_native_per_source_csv,
)


# ── Taubin (windowed-sinc) surface smoothing ─────────────────────
# Embedded from the user's taubin_baseline.py. Everything runs in IDENTITY
# IJK geometry (spacing=1, origin=0); the output labelmap lives on the SAME
# voxel grid as the input by construction (no resampling, no affine math),
# exactly like Slicer's closed-surface<->labelmap round-trip.


def _require_vtk():
    """Import vtk lazily with an actionable error (it's a heavy, optional dep)."""
    try:
        import vtk  # noqa: F401
        from vtk.util import numpy_support as nps  # noqa: F401
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            "[interpolate_native] this control needs the 'vtk' package "
            "(and 'scipy'). Install it in the runtime env, e.g.\n"
            "    pip install vtk scipy\n"
            f"  (import error: {e})"
        )
    return vtk, nps


def sinc_params(f: float) -> Tuple[float, int]:
    """User smoothing factor ``f`` -> (passband, n_iterations).

    Matches Slicer's mapping. ``f=0.5`` -> passband 1e-2, 40 iters;
    ``f=0.7`` -> passband ~1.6e-3, 48 iters. Smaller passband = more
    smoothing.
    """
    passband = 10.0 ** (-4.0 * f)
    n_iterations = int(round(20 + f * 40))
    return passband, n_iterations


def np_to_vtk_ijk(arr_xyz: np.ndarray):
    """numpy (X,Y,Z) -> vtkImageData on the identity IJK grid.

    GOTCHA: VTK is x-fastest; numpy is C-order (last axis fastest). The
    ``transpose(2,1,0).ravel()`` below produces VTK order; the inverse lives
    in :func:`vtk_ijk_to_np`. Verify on an asymmetric phantom before trusting.
    """
    vtk, nps = _require_vtk()
    dims = arr_xyz.shape
    img = vtk.vtkImageData()
    img.SetDimensions(dims)
    img.SetSpacing(1.0, 1.0, 1.0)
    img.SetOrigin(0.0, 0.0, 0.0)
    flat = np.ascontiguousarray(arr_xyz.transpose(2, 1, 0)).ravel()
    vtk_arr = nps.numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    img.GetPointData().SetScalars(vtk_arr)
    return img


def vtk_ijk_to_np(img, dims_xyz: Tuple[int, int, int]) -> np.ndarray:
    vtk, nps = _require_vtk()
    flat = nps.vtk_to_numpy(img.GetPointData().GetScalars())
    X, Y, Z = dims_xyz
    return flat.reshape(Z, Y, X).transpose(2, 1, 0)


def smooth_one_label(
    binary_xyz: np.ndarray, passband: float, n_iter: int, pad: int = 1,
) -> np.ndarray:
    """one-hot binary -> smoothed binary, on the SAME grid (after un-pad)."""
    vtk, _ = _require_vtk()
    # Pad so border-touching structures still yield a CLOSED surface (open
    # surfaces make the stencil leak). Cropped back at the end.
    padded = np.pad(binary_xyz, pad, mode="constant", constant_values=0)
    img = np_to_vtk_ijk(padded.astype(np.uint8))
    ext = img.GetExtent()

    # 1. extract surface
    surf = vtk.vtkDiscreteFlyingEdges3D()
    surf.SetInputData(img)
    surf.SetValue(0, 1)
    surf.Update()

    # 2. Taubin / windowed-sinc smoothing  <-- the baseline operator
    sm = vtk.vtkWindowedSincPolyDataFilter()
    sm.SetInputConnection(surf.GetOutputPort())
    sm.SetPassBand(passband)
    sm.SetNumberOfIterations(n_iter)
    sm.NonManifoldSmoothingOn()
    sm.NormalizeCoordinatesOn()
    sm.BoundarySmoothingOff()
    sm.FeatureEdgeSmoothingOff()
    sm.Update()

    # 3. re-voxelize on the SAME identity grid  <-- grid alignment guaranteed
    stencil = vtk.vtkPolyDataToImageStencil()
    stencil.SetInputConnection(sm.GetOutputPort())
    stencil.SetOutputSpacing(1.0, 1.0, 1.0)
    stencil.SetOutputOrigin(0.0, 0.0, 0.0)
    stencil.SetOutputWholeExtent(ext)
    stencil.Update()

    blank = vtk.vtkImageData()
    blank.SetExtent(ext)
    blank.SetSpacing(1.0, 1.0, 1.0)
    blank.SetOrigin(0.0, 0.0, 0.0)
    blank.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    blank.GetPointData().GetScalars().Fill(0)

    burn = vtk.vtkImageStencil()
    burn.SetInputData(blank)
    burn.SetStencilConnection(stencil.GetOutputPort())
    burn.ReverseStencilOn()                    # fill INSIDE the surface with...
    burn.SetBackgroundValue(1)                 # ...value 1
    burn.Update()

    out_padded = vtk_ijk_to_np(burn.GetOutput(), padded.shape)
    sl = tuple(slice(pad, -pad) for _ in range(3))
    return out_padded[sl].astype(np.uint8)


def taubin_smooth_labelmap(
    labelmap_xyz: np.ndarray, K: int, f: float = 0.7,
) -> np.ndarray:
    """Taubin-smooth a multi-label mask (labels 1..K), on the SAME grid.

    The per-label smoothed one-hots are merged by a signed-distance score
    (+inside / -outside) rather than raw 0/1 argmax: raw argmax leaves GAPS
    at old inter-label boundaries and OVERLAPS where two smoothed labels both
    claim a voxel; the SDF score gives a clean, gap-free partition with a
    principled tie-break. Returns the smoothed integer labelmap.
    """
    from scipy.ndimage import distance_transform_edt

    passband, n_iter = sinc_params(f)
    best_score = np.zeros(labelmap_xyz.shape, dtype=np.float32)
    out_label = np.zeros(labelmap_xyz.shape, dtype=labelmap_xyz.dtype)

    for k in range(1, K + 1):
        binary_k = (labelmap_xyz == k).astype(np.uint8)
        if binary_k.sum() == 0:
            continue
        mask_k = smooth_one_label(binary_k, passband, n_iter)
        inside = distance_transform_edt(mask_k)
        outside = distance_transform_edt(1 - mask_k)
        score_k = (inside - outside).astype(np.float32)
        take = score_k > best_score
        out_label[take] = k
        best_score[take] = score_k[take]

    out_label[best_score <= 0] = 0
    return out_label


# ── Generation: smooth degraded pred -> resample onto native grid ─


def _parse_step_tag(tag: str) -> Tuple[int, int]:
    """``"03"`` -> (3, 0); ``"03_o1"`` -> (3, 1) for the start-offset fan-out."""
    if "_o" in tag:
        s, o = tag.split("_o", 1)
        return int(s), int(o)
    return int(tag), 0


def _save_label(arr: np.ndarray, affine: np.ndarray, ref_header, out_path: Path):
    """Save a uint8 labelmap, reusing the reference header but fixing dtype."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(arr).astype(np.uint8)
    header = ref_header.copy() if ref_header is not None else None
    if header is not None:
        header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(arr, np.asarray(affine, dtype=np.float64), header),
             str(out_path))


def generate(
    cfg: Dict,
    work_dir: Path,
    experiment: str,
    smoothing_factor: float,
    force: bool,
) -> Path:
    """Taubin-smooth the sweep preds onto the native grid. Returns the manifest."""
    pred_root = work_dir / "prediction" / experiment
    sweep_manifest = pred_root / "sweep_manifest.json"
    if not sweep_manifest.exists():
        raise SystemExit(
            f"[interpolate_native] sweep manifest not found: {sweep_manifest}\n"
            f"  Run the `nnunet-predict-sweep` phase (experiment={experiment}) "
            f"first."
        )
    with open(sweep_manifest) as f:
        nn_m = json.load(f)

    interp_root = pred_root / "interpolation"
    interp_root.mkdir(parents=True, exist_ok=True)
    interp_manifest = interp_root / "interp_manifest.json"
    out_steps: Dict[str, Dict[str, str]] = {}
    n_done = 0
    n_skipped = 0
    issues: List[str] = []

    def _flush_manifest() -> None:
        # Written incrementally (after every step) so an interrupted run still
        # leaves a valid manifest covering the masks already on disk; --mode
        # summarize / compare_native can then consume the partial result, and a
        # resumed --mode generate rebuilds it in full.
        with open(interp_manifest, "w") as fh:
            json.dump({
                "experiment": experiment,
                "smoothing_factor": smoothing_factor,
                "method": NNUNET_INTERP_METHOD_LABEL,
                "steps": out_steps,
            }, fh, indent=2)

    print(f"[interpolate_native] experiment={experiment} "
          f"smoothing_factor={smoothing_factor} -> {interp_root}", flush=True)
    _flush_manifest()  # seed an (empty) manifest up front

    for step_tag in sorted(nn_m.get("steps", {})):
        step, _start = _parse_step_tag(str(step_tag))
        sid_map = nn_m["steps"][step_tag]
        degraded_dir = pred_root / f"sparse_step_{step_tag}"
        native_dir = pred_root / f"sparse_step_{step_tag}_native"
        out_dir = interp_root / f"sparse_step_{step_tag}"
        step_map: Dict[str, str] = {}
        entries = sorted(sid_map.items())
        print(f"[interpolate_native] step_{step_tag}: {len(entries)} source(s)",
              flush=True)

        for j, (sid, basename) in enumerate(entries, 1):
            basename = Path(basename).name
            out_path = out_dir / basename
            if not force and out_path.exists():
                step_map[sid] = basename
                n_skipped += 1
                continue

            native_path = native_dir / basename
            if not native_path.exists():
                issues.append(f"step_{step_tag} {sid}: native target missing "
                              f"{native_path}")
                continue
            try:
                t0 = time.time()
                if step == 1:
                    # Dense baseline: already on the native grid -> smooth in
                    # place, no resample. (native_path is a symlink to the
                    # shared dense baseline prediction/native/.)
                    img = nib.load(str(native_path))
                    arr = np.asarray(img.dataobj)
                    arr = np.rint(arr).astype(np.uint8) if np.issubdtype(
                        arr.dtype, np.floating) else arr.astype(np.uint8)
                    K = max(int(arr.max()), 1)
                    smoothed = taubin_smooth_labelmap(arr, K, smoothing_factor)
                    _save_label(smoothed, img.affine, img.header, out_path)
                else:
                    degraded_path = degraded_dir / basename
                    if not degraded_path.exists():
                        issues.append(f"step_{step_tag} {sid}: degraded pred "
                                      f"missing {degraded_path}")
                        continue
                    dimg = nib.load(str(degraded_path))
                    darr = np.asarray(dimg.dataobj)
                    darr = np.rint(darr).astype(np.uint8) if np.issubdtype(
                        darr.dtype, np.floating) else darr.astype(np.uint8)
                    K = max(int(darr.max()), 1)
                    # 1) Taubin-smooth on the degraded grid.
                    smoothed = taubin_smooth_labelmap(darr, K, smoothing_factor)
                    # 2) Resample (order=0, world-aware) onto the native grid.
                    nimg = nib.load(str(native_path))
                    native_shape = tuple(int(x) for x in nimg.shape)
                    native_aff = np.asarray(nimg.affine, dtype=np.float64)
                    native_arr = resample_pred_onto_gt(
                        smoothed, np.asarray(dimg.affine, dtype=np.float64),
                        native_shape, native_aff,
                    )
                    _save_label(native_arr, native_aff, nimg.header, out_path)
                step_map[sid] = basename
                n_done += 1
                print(f"[interpolate_native] step_{step_tag} [{j}/{len(entries)}] "
                      f"{sid}: done ({time.time() - t0:.1f}s)", flush=True)
            except SystemExit:
                raise
            except Exception as e:  # noqa: BLE001
                issues.append(f"step_{step_tag} {sid}: smoothing failed ({e})")
                continue

        if step_map:
            out_steps[step_tag] = step_map
            _flush_manifest()  # persist progress after each completed step

    _flush_manifest()
    if issues:
        print(f"\n[interpolate_native] {len(issues)} issue(s):", file=sys.stderr)
        for line in issues:
            print(f"  - {line}", file=sys.stderr)
    print(f"\n[interpolate_native] smoothed {n_done}; skipped {n_skipped} "
          f"already-present. manifest: {interp_manifest}")
    return interp_manifest


# ── Standalone summary (mirrors build_nnunet_native_summary.py) ───


def summarize(
    cfg: Dict,
    work_dir: Path,
    experiment: str,
    include_prefixes: List[str],
    exclude_prefixes: List[str],
    out_dir: Path,
) -> List[Path]:
    """Dice the interp masks vs native GT and write the per-step bundle."""
    cnisp_paths = load_yaml(Path(cfg["cnisp_paths_yaml"]))
    out_dir.mkdir(parents=True, exist_ok=True)

    sources, _missing = resolve_test_sources(cnisp_paths)
    inc = tuple(p for p in include_prefixes if p)
    exc = tuple(p for p in exclude_prefixes if p)
    if inc or exc:
        kept = []
        for s in sources:
            if inc and not s.source_id.startswith(inc):
                continue
            if exc and s.source_id.startswith(exc):
                continue
            kept.append(s)
        print(f"[interpolate_native] source filter: include={list(inc)!r} "
              f"exclude={list(exc)!r} -> {len(kept)}/{len(sources)} sources.",
              file=sys.stderr)
        sources = kept
    if not sources:
        raise SystemExit("All sources filtered out; relax include/exclude.")

    eff_res_idx = eff_res_from_sparse_manifest(work_dir, experiment)
    wide_rows, stats = compute_nnunet_native_rows(
        work_dir, experiment, sources, eff_res_idx,
        manifest_name="interpolation/interp_manifest.json",
        pred_subdir_fmt="interpolation/sparse_step_{:02d}",
    )
    if not wide_rows:
        raise SystemExit(
            "No interp Dice rows produced -- check that "
            f"prediction/{experiment}/interpolation/sparse_step_XX/ is populated "
            f"(run --mode generate first).")

    bucket_edges = list(cfg.get("summary_bucket_edges_mm",
                                list(DEFAULT_BUCKET_EDGES_MM)))
    step_rows = aggregate_native_by_step(wide_rows)
    bucket_rows = aggregate_native_by_eff_res(wide_rows, bucket_edges)

    per_source_csv = out_dir / f"interp_native_per_source__{experiment}.csv"
    by_step_csv = out_dir / f"interp_native_by_step__{experiment}.csv"
    by_eff_csv = out_dir / f"interp_native_by_eff_res__{experiment}.csv"
    step_png = out_dir / f"interp_native_dice_vs_step__{experiment}.png"
    eff_png = out_dir / f"interp_native_dice_vs_eff_res__{experiment}.png"

    write_native_per_source_csv(wide_rows, per_source_csv)
    write_native_by_step_csv(step_rows, by_step_csv)
    write_native_by_eff_res_csv(bucket_rows, by_eff_csv)
    plot_native_dice_vs_step(step_rows, NNUNET_INTERP_METHOD_LABEL, step_png)
    plot_native_dice_vs_eff_res(bucket_rows, NNUNET_INTERP_METHOD_LABEL, eff_png)

    n_sources = len({r["source_id"] for r in wide_rows})
    print(f"[interpolate_native] {NNUNET_INTERP_METHOD_LABEL} (experiment="
          f"{experiment}): {len(wide_rows)} (source,step) row(s) across "
          f"{n_sources} source(s), {len(step_rows)} step(s).")
    print(f"  sources Diced={stats['sources']} skipped_gt={stats['skipped_gt']} "
          f"skipped_pred={stats['skipped_pred']} "
          f"atlas_grid_mismatch={stats['skipped_atlas_mismatch']} "
          f"chk_resampled={stats['resampled_chk']}")
    outs = [per_source_csv, by_step_csv, by_eff_csv, step_png, eff_png]
    for p in outs:
        print(f"  {p}")
    return outs


def run(args) -> int:
    cfg = load_yaml(Path(args.config))
    work_dir = Path(args.work_dir or cfg["work_dir"])
    if args.split == "train":
        work_dir = work_dir / "train_split"
    experiment = str(args.experiment)
    mode = args.mode

    if mode in ("generate", "all"):
        generate(cfg, work_dir, experiment,
                 smoothing_factor=args.smoothing_factor, force=args.force)
    if mode in ("summarize", "all"):
        if args.out_dir is not None:
            out_dir = Path(args.out_dir)
        else:
            out_dir = work_dir / "prediction" / experiment / "interpolation" / "summary"
        include_prefixes, exclude_prefixes = resolve_source_prefix_filters(
            args.include_source_prefixes, args.exclude_source_prefixes, cfg)
        summarize(cfg, work_dir, experiment,
                  include_prefixes, exclude_prefixes, out_dir)
    return 0
