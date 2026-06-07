#!/usr/bin/env python3
"""Sparse-CT inference that keeps nnUNet's *plan-spacing* prediction.

Why this exists (the bug this replaces)
----------------------------------------
``nnUNetv2_predict`` (and the old ``run_predict_sparse_sweep.sh`` +
``engine/upsample_sparse_preds.py`` pair) only ever gave us the
prediction **resampled back onto the sparse input grid**: nnUNet runs

    sparse CT --(resample up)--> plan spacing (iso 0.5) --> network
              --(resample down)--> sparse input grid --> save

and the CLI saves only that last, back-to-sparse mask. Our downstream
``upsample_sparse_preds.py`` then nearest-neighbour *duplicated* slices
along the through-plane axis to reach the native grid. So even the
``*_upsampled`` mask carried only sparse-resolution **content** (blocky
duplicated slices) on a dense grid -- the fine iso-0.5 prediction the
network actually produced was already thrown away at the resample-down
step and could never be recovered by NN duplication.

What this script does instead
-----------------------------
It runs inference through nnUNet's Python predictor and intercepts the
**plan-spacing logits** (the network output *before* nnUNet maps it back
to the input/original space). From one inference per input it writes
two masks per ``(source_id, step)``:

* ``prediction/sparse_step_{XX}/{sid}.nii.gz``
      nnUNet's normal output on the sparse input grid (identical to what
      ``nnUNetv2_predict`` would write for the sparse CT). Kept verbatim
      for the deployment-curve consumer
      (``engine/build_dataset835_sparse_patches.py``), which still reads
      this directory.
* ``prediction/sparse_step_{XX}_native/{sid}.nii.gz``
      the plan-spacing prediction resampled onto the **native CT grid**
      by **world coordinates** (not NN slice duplication, and not nnUNet's
      affine-blind shape resampler). This is the mask ``compare_native.py``
      Dices against the native GT; GT is never resampled.

The intermediate iso-0.5 plan-spacing mask the network produces is NOT
saved -- it had no downstream consumer (it was reference-only), and it
can be regenerated on demand straight from ``nnUNetv2_predict`` if ever
needed. Dropping it also lets step_01 skip a full per-source inference
(its native target is just a symlink to the dense baseline; see below).

How the native mask is placed (and the offset bug this fixes)
-------------------------------------------------------------
The sparse mask (output 1) uses nnUNet's own export
(``convert_predicted_logits_to_segmentation_with_correct_shape``): it
resamples logits to the sparse crop shape and pads back via the crop
bbox -- correct, because it round-trips onto the same sparse grid.

The native mask (output 2) does NOT reuse that export. nnUNet's resampler
is purely array-SHAPE based (it aligns array *extents*, ignoring the
affine). If we just scaled the crop bbox by ``step`` and re-ran it (the
previous approach), the plan FOV's extent got aligned to the native
crop's extent -- two FOVs of equal width but offset by half a coarse
voxel -- so every kept sparse slice landed at the CENTER of its
``step``-wide slab instead of at its start. That is a through-plane shift
of ``(step-1)/2`` native voxels, growing with ``step`` (0.5 vox at
step=2 ... 3.5 vox at step=8), silently dragging Dice down vs the
start=0 GT. ``compare_native``'s affine check could not catch it: the
grid geometry was right; only the *content* was shifted.

Instead we resample the plan/iso logits onto the native grid by WORLD
coordinates (``_native_seg_world_aware`` -> ``nibabel.resample_from_to``).
The plan grid's world affine is reconstructed from the *sparse CT's own*
nibabel affine -- the true start=0 sweep geometry (``_plan_affine_nib``,
FOV-preserving half-pixel, matching skimage's resize that nnUNet uses on
the forward pass). Because sparsification kept every ``step``-th slice
starting at index 0, sparse voxel ``i`` sits at native voxel ``i*step``;
the world-coordinate resample honours that to sub-voxel precision for any
``step`` (even or odd) and for both thin and thick degradation. A
self-contained numerical check of all three reconstructions lives in
``nnunet/engine/_audit_native_offset.py``. We still assert the produced
native mask matches the native CT's nibabel shape exactly as an
axis-order guard.

step_01 (dense baseline)
------------------------
step_01 isn't sparsified. Its native-grid Dice target is the existing
dense baseline ``prediction/native/{sid}.nii.gz`` (shared with other
consumers), so ``_native/01`` is just a symlink there -- step_01 runs no
inference of its own.

Output
------
* the two directories above, and
* ``prediction/sweep_manifest.json`` -- ``{steps: {XX: {sid: basename}}}``
  anchored by ``compare_native.py`` against ``sparse_step_{XX}_native/``.

Usage
-----
    python nnunet/engine/predict_sparse_iso.py --config nnunet/configs.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

# Make ``nnunet.*`` importable when run as ``python nnunet/engine/...``.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nnunet.helpers.config import load_yaml  # noqa: E402
from nnunet.engine.native_resample import resample_plan_to_native  # noqa: E402
from simulation.affine_ops import assert_start_zero as _assert_start_zero  # noqa: E402
import torch  # noqa: F401
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.export_prediction import (
    convert_predicted_logits_to_segmentation_with_correct_shape as _convert,
)


def _resolve_model_folder(cfg: Dict) -> Path:
    """``${nnUNet_results}/Dataset{ID}_{NAME}/{trainer}__{plan}__{cfg}``.

    Mirrors how ``nnUNetv2_predict`` locates the trained model from
    ``-d/-tr/-p/-c`` so a custom predictor stays bit-compatible with the
    CLI sweep it replaces.
    """
    results = os.environ.get("nnUNet_results")
    if not results:
        raise SystemExit(
            "[predict_sparse_iso] env var nnUNet_results is unset -- set it "
            "to the same value used by nnUNetv2_predict."
        )
    ds_id = int(cfg["dataset_id"])
    ds_name = cfg.get("dataset_name", "")
    ds_folder = f"Dataset{ds_id:03d}_{ds_name}"
    trainer = cfg.get("trainer", "nnUNetTrainer")
    plan = cfg.get("plan", "nnUNetPlans")
    configuration = cfg.get("configuration", "3d_fullres")
    model_folder = Path(results) / ds_folder / f"{trainer}__{plan}__{configuration}"
    if not model_folder.is_dir():
        raise SystemExit(
            f"[predict_sparse_iso] model folder not found:\n  {model_folder}\n"
            f"  Check dataset_id/dataset_name/trainer/plan/configuration in "
            f"the config and that nnUNet_results points at the trained model."
        )
    return model_folder


def _resolve_folds(model_folder: Path, folds_cfg, checkpoint_name: str):
    """Map ``folds`` config to trained fold ids. 'best'=highest val Dice,
    'all'=every trained fold, or an explicit list (untrained ones dropped)."""
    avail = sorted(int(d.name[5:]) for d in Path(model_folder).glob("fold_*")
                   if (d / checkpoint_name).is_file())
    if not avail:
        raise SystemExit(f"[predict_sparse_iso] no trained fold under {model_folder}")
    if folds_cfg == "all":
        return tuple(avail)
    if folds_cfg == "best":
        def _dice(f):
            s = Path(model_folder) / f"fold_{f}" / "validation" / "summary.json"
            try:
                return json.loads(s.read_text())["foreground_mean"]["Dice"]
            except Exception:
                return -1.0
        return (max(avail, key=_dice),)
    req = [folds_cfg] if isinstance(folds_cfg, int) else [int(f) for f in folds_cfg]
    keep = tuple(f for f in req if f in avail)
    if not keep:
        raise SystemExit(f"[predict_sparse_iso] requested folds {req} not trained "
                         f"(available {avail})")
    return keep


def _init_predictor(cfg: Dict):
    gpu_id = cfg.get("gpu_id", 0)
    # Respect the config GPU the same way the shell scripts do.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(gpu_id))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_folder = _resolve_model_folder(cfg)
    folds = _resolve_folds(
        model_folder,
        cfg.get("folds", [0]),
        cfg.get("checkpoint_name", "checkpoint_final.pth"),
    )
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=(device.type == "cuda"),
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(model_folder),
        use_folds=folds,
        checkpoint_name=cfg.get("checkpoint_name", "checkpoint_final.pth"),
    )
    # nnUNet's own reader/writer. CRITICAL: for .nii.gz this is SimpleITKIO,
    # whose numpy axis order is the REVERSE of nibabel's. Every segmentation
    # nnUNet returns is in this "as-read" order, so we must save the consumed
    # masks through this same rw (round-trip-safe) rather than pairing an
    # as-read array with a nibabel affine.
    rw = predictor.plans_manager.image_reader_writer_class()
    print(f"[predict_sparse_iso] model:  {model_folder}")
    print(f"[predict_sparse_iso] folds:  {folds}  device: {device}")
    print(f"[predict_sparse_iso] reader/writer: "
          f"{type(rw).__name__} (nnUNet as-read axis order)")
    return predictor, torch, _convert, rw


def _detect_io2nib(rw, path) -> List[int]:
    """Permutation ``p`` such that ``nib_array == rw_array.transpose(p)``.

    i.e. nibabel spatial axis ``k`` corresponds to nnUNet-as-read axis
    ``p[k]``. Resolved per file by matching (shape, spacing) between how
    nibabel reads the file and how nnUNet's reader_writer reads it, so it is
    correct regardless of which reader (SimpleITKIO reverses axes; NibabelIO
    does not) and regardless of the scan's anatomical orientation.
    """
    nimg = nib.load(str(path))
    nib_shape = tuple(int(x) for x in nimg.shape[:3])
    nib_zooms = tuple(float(z) for z in nimg.header.get_zooms()[:3])
    data, props = rw.read_images([str(path)])
    rw_shape = tuple(int(x) for x in np.asarray(data).shape[-3:])
    rw_spacing = tuple(float(z) for z in props["spacing"][:3])

    perm: List[Optional[int]] = [None, None, None]
    used: set = set()
    for k in range(3):
        cands = [
            i for i in range(3)
            if i not in used and rw_shape[i] == nib_shape[k]
            and abs(rw_spacing[i] - nib_zooms[k]) <= 1e-3 * max(nib_zooms[k], 1e-6)
        ]
        if len(cands) != 1:
            # Degenerate (≥2 axes share shape AND spacing): fall back to the
            # pure-reversal (SimpleITK) or identity convention if it is at
            # least shape-consistent; otherwise fail loudly.
            if tuple(reversed(rw_shape)) == nib_shape:
                return [2, 1, 0]
            if rw_shape == nib_shape:
                return [0, 1, 2]
            raise SystemExit(
                f"[predict_sparse_iso] cannot resolve nnUNet<->nibabel axis "
                f"order for {path}: rw {rw_shape}/{rw_spacing} vs nibabel "
                f"{nib_shape}/{nib_zooms}. Report so the map can be set."
            )
        perm[k] = cands[0]
        used.add(cands[0])
    assert sorted(p for p in perm if p is not None) == [0, 1, 2]
    return [int(p) for p in perm]  # type: ignore[arg-type]


def _native_geom(rw, ct_path, cache: Dict) -> Tuple:
    """Cached native-CT geometry: as-read + nibabel views + the permutation.

    Returns ``(rw_shape, rw_spacing, rw_props, io2nib, nib_shape, nib_affine)``.

    The as-read view (``rw_*`` from nnUNet's reader_writer) is kept only for
    the legacy shape bookkeeping; the world-aware native resampling uses the
    nibabel view (``nib_shape``/``nib_affine``) so the produced mask lands on
    EXACTLY the native CT's voxel grid (and therefore the GT's grid, which
    ``compare_native`` Dices against without resampling). Cached because every
    step of a source re-uses the same native grid.
    """
    key = str(Path(ct_path).resolve())
    if key not in cache:
        data, props = rw.read_images([str(ct_path)])
        rw_shape = tuple(int(x) for x in np.asarray(data).shape[-3:])
        rw_spacing = tuple(float(z) for z in props["spacing"][:3])
        io2nib = _detect_io2nib(rw, ct_path)
        nimg = nib.load(str(ct_path))
        nib_shape = tuple(int(x) for x in nimg.shape[:3])
        nib_affine = np.asarray(nimg.affine, dtype=np.float64)
        cache[key] = (rw_shape, rw_spacing, props, io2nib, nib_shape, nib_affine)
    return cache[key]


def _predict_logits(predictor, torch, input_file: Path):
    """Preprocess ``input_file`` and return ``(logits, properties)``.

    ``logits`` is a torch tensor ``(n_classes, *spatial)`` at the plan
    spacing in nnUNet's internal (transposed) axis order; ``properties``
    carries the spacing / crop bookkeeping nnUNet's export path needs.
    """
    preprocessor = predictor.configuration_manager.preprocessor_class(verbose=False)
    data, _seg, properties = preprocessor.run_case(
        [str(input_file)],
        None,
        predictor.plans_manager,
        predictor.configuration_manager,
        predictor.dataset_json,
    )
    data_t = torch.from_numpy(np.ascontiguousarray(data)).float()
    logits = predictor.predict_logits_from_preprocessed_data(data_t)
    return logits, properties


def _segmentation_from_logits(convert_fn, predictor, logits, properties):
    """Run nnUNet's logits->segmentation+resample+crop-undo for ``properties``.

    Returns the segmentation as a numpy array in the file's axis order.
    """
    seg = convert_fn(
        logits,
        predictor.plans_manager,
        predictor.configuration_manager,
        predictor.label_manager,
        properties,
        return_probabilities=False,
    )
    return np.asarray(seg)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="nnunet/configs.yaml")
    ap.add_argument("--force", action="store_true",
                    help="Recompute masks even if both outputs exist.")
    ap.add_argument("--experiment", choices=["thin", "thick", "real"],
                    default="thin",
                    help="Experiment directory layer. Reads sparse inputs "
                         "from input/<experiment>/ and writes preds to "
                         "prediction/<experiment>/ so thin/thick sweeps "
                         "coexist. The shared native/ dense baseline is NOT "
                         "exp-scoped.")
    args = ap.parse_args()

    cfg = load_yaml(Path(args.config))
    work_dir = Path(cfg["work_dir"])
    experiment = args.experiment

    sparse_manifest = work_dir / "input" / experiment / "sparse_manifest.json"
    if not sparse_manifest.exists():
        print(f"[predict_sparse_iso] missing {sparse_manifest} -- run "
              f"nnunet/data_prep/sparsify_inputs.py --experiment "
              f"{experiment} first.", file=sys.stderr)
        return 2
    with open(sparse_manifest) as f:
        sparse_m = json.load(f)

    source_to_path = work_dir / "source_to_path.json"
    if not source_to_path.exists():
        print(f"[predict_sparse_iso] missing {source_to_path} -- run "
              f"nnunet/data_prep/prepare_inputs.py first.", file=sys.stderr)
        return 2
    with open(source_to_path) as f:
        src_to_path = json.load(f)

    pred_root = work_dir / "prediction"
    dense_pred_dir = pred_root / "native"          # shared dense baseline
    sparse_pred_root = pred_root / experiment       # exp-scoped sparse sweep
    sparse_pred_root.mkdir(parents=True, exist_ok=True)

    predictor, torch, convert_fn, rw = _init_predictor(cfg)
    transpose_forward = list(predictor.plans_manager.transpose_forward)
    # Cache of per-source native geometry (rw shape/spacing/props + the
    # nibabel<->as-read permutation), filled lazily and reused across steps.
    geom_cache: Dict[str, Tuple] = {}

    # The world-aware native resampling reorders nnUNet's INTERNAL-order
    # logits/crop-bbox into nibabel order via _internal_to_nib_perm, so it is
    # correct for any transpose_forward. The usual 3d_fullres plan is identity
    # anyway; if it is NOT, spot-check that each sparse_step_XX_native mask
    # overlays its CT correctly (the perm path is exercised but rarely).
    if transpose_forward != [0, 1, 2]:
        print(f"[predict_sparse_iso] NOTE: transpose_forward="
              f"{transpose_forward} is non-identity. The native resampler "
              f"reorders internal->nibabel explicitly; spot-check that each "
              f"sparse_step_XX_native mask overlays its CT correctly.",
              flush=True)

    n_sparse_jobs = sum(
        len(v) for v in sparse_m.get("by_step", {}).values()
    )
    print(f"[predict_sparse_iso] workload: step_01 dense baseline "
          f"{len(src_to_path)} source(s) (symlink only, no inference); "
          f"sparse sweep {n_sparse_jobs} (source, step) pair(s). "
          f"Each inference is silent for several minutes -- per-case "
          f"progress is logged below.", flush=True)

    out_steps: Dict[str, Dict[str, str]] = {}
    n_inferred = 0
    n_skipped = 0
    issues: List[str] = []

    # ── step_01: dense baseline ────────────────────────────────
    # _native/01 is just a symlink to the shared dense baseline, so step_01
    # runs no inference of its own (the previous iso-upsampled output had no
    # consumer and has been removed).
    native_01 = sparse_pred_root / "sparse_step_01_native"
    step_01_map: Dict[str, str] = {}
    step_01_ids = sorted(src_to_path)
    for i, sid in enumerate(step_01_ids, 1):
        dense_pred = dense_pred_dir / f"{sid}.nii.gz"
        if not dense_pred.exists():
            issues.append(f"step_01 {sid}: no dense baseline at {dense_pred}")
            continue
        # _native/01 -> symlink the shared dense baseline.
        native_01.mkdir(parents=True, exist_ok=True)
        dst_native = native_01 / f"{sid}.nii.gz"
        if dst_native.is_symlink() or dst_native.exists():
            dst_native.unlink()
        dst_native.symlink_to(dense_pred.resolve())
        step_01_map[sid] = dst_native.name
    if step_01_map:
        out_steps["01"] = step_01_map
        print(f"[predict_sparse_iso] step_01: {len(step_01_map)} dense "
              f"baseline(s) -> _native (symlink, no inference)")

    # ── steps >= 2: sparse-CT sweep ─────────────────────────────
    for step_tag in sorted(sparse_m.get("by_step", {}).keys()):
        step = int(step_tag)
        sparse_dir = sparse_pred_root / f"sparse_step_{step_tag}"
        native_dir = sparse_pred_root / f"sparse_step_{step_tag}_native"
        step_map: Dict[str, str] = {}
        step_entries = sorted(sparse_m["by_step"][step_tag].items())
        n_step = len(step_entries)
        print(f"[predict_sparse_iso] step_{step_tag}: {n_step} source(s) "
              f"-> sparse + _native", flush=True)

        for j, (sid, info) in enumerate(step_entries, 1):
            sparse_input = Path(info["input"])
            if not sparse_input.exists():
                issues.append(f"step_{step_tag} {sid}: sparse input missing "
                              f"{sparse_input}")
                continue

            dst_sparse = sparse_dir / f"{sid}.nii.gz"
            dst_native = native_dir / f"{sid}.nii.gz"
            if (not args.force and dst_sparse.exists()
                    and dst_native.exists()):
                step_map[sid] = dst_native.name
                n_skipped += 1
                print(f"[predict_sparse_iso] step_{step_tag} [{j}/{n_step}] "
                      f"{sid}: all outputs exist -- skip", flush=True)
                continue

            ct_info = src_to_path.get(sid)
            if not ct_info or "ct_image_path" not in ct_info:
                issues.append(f"step_{step_tag} {sid}: no ct_image_path in "
                              f"source_to_path.json")
                continue
            ct_path = ct_info["ct_image_path"]
            # Native geometry. The world-aware native resampling uses the
            # nibabel (shape, affine) view so the produced mask lands on the
            # native CT's exact grid (== GT grid). io2nib carries the nnUNet
            # internal/as-read <-> nibabel permutation used to reorder the
            # plan logits and crop bbox into nibabel order.
            (native_rw_shape, native_rw_spacing, native_rw_props, io2nib,
             native_nib_shape, native_nib_affine) = _native_geom(
                rw, ct_path, geom_cache)

            try:
                t0 = time.time()
                print(f"[predict_sparse_iso] step_{step_tag} [{j}/{n_step}] "
                      f"{sid}: inference ...", flush=True)
                logits, props = _predict_logits(predictor, torch, sparse_input)

                # 1) sparse-grid mask (nnUNet's normal output). Saved through
                # nnUNet's own writer so the as-read array and its geometry
                # round-trip exactly (no nibabel/SimpleITK axis-order mix).
                sparse_seg = _segmentation_from_logits(
                    convert_fn, predictor, logits.clone(), props,
                )
                expected_sparse = tuple(int(x) for x in props["shape_before_cropping"])
                if tuple(sparse_seg.shape) != expected_sparse:
                    issues.append(
                        f"step_{step_tag} {sid}: sparse mask shape "
                        f"{sparse_seg.shape} != as-read input {expected_sparse}"
                    )
                    continue
                dst_sparse.parent.mkdir(parents=True, exist_ok=True)
                rw.write_seg(sparse_seg.astype(np.uint8), str(dst_sparse), props)

                # 2) native-grid mask via WORLD-COORDINATE resampling of the
                # plan/iso logits (see _native_seg_world_aware). This places
                # each kept sparse slice at native voxel i*step (start=0),
                # instead of nnUNet's affine-blind resampler which centred it
                # in the step-wide slab (a (step-1)/2 through-plane shift).
                # Enforce the mode and start=0 invariants from the manifest:
                # the start=0 sweep geometry is what _plan_affine_nib assumes.
                assert info.get("mode", "thin") in ("thin", "thick"), (
                    f"step_{step_tag} {sid}: unexpected mode={info.get('mode')}"
                )
                # Legacy manifests pre-date the "start" field; default to 0
                # (which was the only value ever produced).
                _assert_start_zero(int(info.get("start", 0)))
                sparse_affine = np.asarray(
                    nib.load(str(sparse_input)).affine, dtype=np.float64
                )
                # argmax the plan-spacing logits (INTERNAL axis order), then
                # resample that label map onto the native grid by world coords.
                plan_internal = np.asarray(
                    logits.argmax(0).cpu().numpy()
                ).astype(np.uint8)
                native_seg = resample_plan_to_native(
                    plan_internal, transpose_forward, io2nib,
                    props["bbox_used_for_cropping"], sparse_affine,
                    native_nib_shape, native_nib_affine,
                )
                # Hard guard: world-aware resample is constructed to output the
                # native CT's exact nibabel shape; a mismatch means an axis or
                # geometry slip, so skip loudly rather than write a bad Dice
                # target.
                if tuple(native_seg.shape) != tuple(native_nib_shape):
                    issues.append(
                        f"step_{step_tag} {sid}: native mask shape "
                        f"{native_seg.shape} != native CT {native_nib_shape} "
                        f"(nibabel order). Axis-order/geometry bug -- NOT writing."
                    )
                    continue
                dst_native.parent.mkdir(parents=True, exist_ok=True)
                nib.save(
                    nib.Nifti1Image(native_seg, native_nib_affine),
                    str(dst_native),
                )

                step_map[sid] = dst_native.name
                n_inferred += 1
                print(f"[predict_sparse_iso] step_{step_tag} [{j}/{n_step}] "
                      f"{sid}: done ({time.time() - t0:.1f}s)", flush=True)
            except Exception as e:  # noqa: BLE001
                issues.append(f"step_{step_tag} {sid}: inference failed ({e})")
                continue

        if step_map:
            out_steps[step_tag] = step_map
            print(f"[predict_sparse_iso] step_{step_tag}: finished "
                  f"{len(step_map)} source(s)", flush=True)

    sweep_manifest_path = sparse_pred_root / "sweep_manifest.json"
    with open(sweep_manifest_path, "w") as f:
        json.dump({"experiment": experiment, "steps": out_steps}, f, indent=2)

    if issues:
        print(f"\n[predict_sparse_iso] {len(issues)} issue(s):", file=sys.stderr)
        for line in issues:
            print(f"  - {line}", file=sys.stderr)

    print(f"\n[predict_sparse_iso] inferred {n_inferred}; skipped "
          f"{n_skipped} already-complete.")
    print(f"[predict_sparse_iso] manifest: {sweep_manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
