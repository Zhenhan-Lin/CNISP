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
three masks per ``(source_id, step)``:

* ``prediction/sparse_step_{XX}/{sid}.nii.gz``
      nnUNet's normal output on the sparse input grid. Kept verbatim for
      the deployment-curve consumer
      (``engine/build_dataset835_sparse_patches.py``), which still reads
      this directory.
* ``prediction/sparse_step_{XX}_upsampled/{sid}.nii.gz``
      the **isotropic plan-spacing** prediction == ``argmax`` of the
      logits (nonzero-cropped FOV, iso 0.5 mm grid). This is the genuine
      network output the old pipeline discarded.
* ``prediction/sparse_step_{XX}_native/{sid}.nii.gz``
      the plan-spacing prediction resampled onto the **native CT grid**
      using **nnUNet's own segmentation resampler** (not NN slice
      duplication). This is the mask ``compare_native.py`` Dices against
      the native GT; GT is never resampled.

How the native mask reuses nnUNet's resampler honestly
------------------------------------------------------
nnUNet's ``convert_predicted_logits_to_segmentation_with_correct_shape``
resamples logits from plan spacing to ``properties['shape_after_...']``
and pads back to ``properties['shape_before_cropping']`` via the crop
bbox, then transposes to the file's axis order. We call it twice:

1. with the *original* (sparse) properties  -> the sparse-grid mask;
2. with a *native* properties dict we synthesize -- the sparse crop
   geometry with the through-plane axis scaled by ``step`` -- so the very
   same nnUNet resampler lands the logits directly on the native grid.

Because sparsification kept every ``step``-th slice starting at index 0
(``slice_start_id=0`` in ``data_prep/sparsify_inputs.py``), sparse voxel
``i`` along the through-plane axis sits at native voxel ``i * step``, so
the sparse crop ``[lo, hi)`` maps to native ``[lo*step, hi*step)`` (clipped
to the native extent). We assert the resulting native mask matches the
native CT's shape exactly -- a hard guard against any axis-order slip in
the internal/original transpose bookkeeping.

step_01 (dense baseline)
------------------------
step_01 isn't sparsified. Its native-grid Dice target is the existing
dense baseline ``prediction/native/{sid}.nii.gz`` (shared with other
consumers), so ``_native/01`` is symlinked there rather than recomputed.
``_upsampled/01`` is the genuine iso-0.5 prediction of the native CT,
produced through the same predictor for consistency with the sweep.

Output
------
* the three directories above, and
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

# Make ``nnunet.*`` importable when run as ``python nnunet/engine/...``.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nnunet.helpers.config import load_yaml  # noqa: E402
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


def _init_predictor(cfg: Dict):
    gpu_id = cfg.get("gpu_id", 0)
    # Respect the config GPU the same way the shell scripts do.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(gpu_id))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    folds = cfg.get("folds", [0])
    if not isinstance(folds, (list, tuple)):
        folds = [folds]
    folds = tuple(int(f) for f in folds)

    model_folder = _resolve_model_folder(cfg)
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


def _rw_vec_to_nib(vec_rw: List, io2nib: List[int]) -> List:
    """Reorder a per-axis vector from nnUNet-as-read order to nibabel order."""
    return [vec_rw[io2nib[k]] for k in range(3)]


def _native_geom(rw, ct_path, cache: Dict) -> Tuple:
    """Cached ``(rw_shape, rw_spacing, rw_props, io2nib)`` for a native CT.

    Read once per source via nnUNet's reader_writer (so shape/spacing are in
    as-read order and ``rw_props`` carries the writer metadata) plus the
    nibabel<->as-read permutation. Cached because every step of a source
    re-uses the same native grid.
    """
    key = str(Path(ct_path).resolve())
    if key not in cache:
        data, props = rw.read_images([str(ct_path)])
        rw_shape = tuple(int(x) for x in np.asarray(data).shape[-3:])
        rw_spacing = tuple(float(z) for z in props["spacing"][:3])
        io2nib = _detect_io2nib(rw, ct_path)
        cache[key] = (rw_shape, rw_spacing, props, io2nib)
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


def _internal_to_original(values_internal: List, transpose_forward: List[int]) -> List:
    """Reorder a per-axis list from nnUNet internal order to file order.

    nnUNet builds the internal array as ``original.transpose(transpose_
    forward)``, so internal axis ``i`` carries original axis
    ``transpose_forward[i]``. Therefore the file-order value for original
    axis ``transpose_forward[i]`` is the internal value at ``i``.
    """
    out = [None, None, None]
    for i, a in enumerate(transpose_forward):
        out[a] = values_internal[i]
    return out


def _save_uint8(arr: np.ndarray, affine: np.ndarray, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(arr.astype(np.uint8), affine)
    img.set_qform(affine)
    img.set_sform(affine)
    nib.save(img, str(dst))


def _iso_mask_from_logits(
    logits, torch, transpose_backward: List[int], io2nib: List[int],
) -> np.ndarray:
    """Plan-spacing prediction == argmax over classes, in NIBABEL axis order.

    Labels {0:bg, 1:ON, 2:recti, 3:globe, 4:fat} are consecutive and the
    dataset is not region-based, so a plain argmax reproduces nnUNet's
    label values without the region-merge logic.

    ``transpose_backward`` brings the argmax from nnUNet internal order to
    its as-read order; ``io2nib`` then brings it to nibabel order so it can
    be paired with a nibabel affine (this iso mask is reference-only and is
    saved with a hand-built affine, not through the reader_writer).
    """
    seg_internal = torch.argmax(logits, dim=0).cpu().numpy().astype(np.uint8)
    seg_rw = seg_internal.transpose(transpose_backward)   # nnUNet as-read order
    seg_nib = seg_rw.transpose(io2nib)                    # nibabel file order
    return np.ascontiguousarray(seg_nib)


def _iso_affine(
    sparse_affine: np.ndarray,
    sparse_bbox_internal: List,
    plan_spacing_internal: List[float],
    transpose_forward: List[int],
    io2nib: List[int],
) -> np.ndarray:
    """Affine for the cropped iso prediction in the sparse CT's world frame.

    The iso volume is nnUNet's nonzero-crop of the sparse CT resampled to
    the plan (iso) spacing. Its voxel (0,0,0) is the crop's lower corner
    in sparse-voxel space; its direction matches the sparse CT with each
    axis rescaled to the plan spacing.

    ``transpose_forward`` maps the internal bbox/spacing to nnUNet as-read
    order; ``io2nib`` then maps to nibabel voxel order so the vectors line up
    with ``sparse_affine`` (a nibabel affine whose columns are nibabel axes).
    """
    bbox_lower_internal = [lo for (lo, _hi) in sparse_bbox_internal]
    bbox_lower_rw = _internal_to_original(bbox_lower_internal, transpose_forward)
    p_rw = _internal_to_original(list(plan_spacing_internal), transpose_forward)
    bbox_lower_orig = _rw_vec_to_nib(bbox_lower_rw, io2nib)
    p_orig = _rw_vec_to_nib(p_rw, io2nib)

    R = np.asarray(sparse_affine[:3, :3], dtype=np.float64)
    col_norms = np.linalg.norm(R, axis=0)
    col_norms[col_norms == 0] = 1.0
    unit = R / col_norms
    iso_R = unit * np.asarray(p_orig, dtype=np.float64)[None, :]

    corner_world = R @ np.asarray(bbox_lower_orig, dtype=np.float64) \
        + np.asarray(sparse_affine[:3, 3], dtype=np.float64)

    iso_affine = np.eye(4, dtype=np.float64)
    iso_affine[:3, :3] = iso_R
    iso_affine[:3, 3] = corner_world
    return iso_affine


def _native_properties(
    sparse_props: Dict,
    native_shape_orig: Tuple[int, int, int],
    native_zooms_orig: Tuple[float, float, float],
    step: int,
    step_axis_orig: int,
    transpose_forward: List[int],
) -> Dict:
    """Synthesize an nnUNet ``properties`` dict describing the native grid.

    Only the through-plane axis differs between the sparse and native
    grids (native has ``step`` times more slices over the same FOV), so we
    take the sparse crop bookkeeping and scale that one axis by ``step``.
    ``native_shape_orig`` / ``native_zooms_orig`` / ``step_axis_orig`` are
    in nnUNet's AS-READ (reader_writer) order, matching ``sparse_props``;
    ``spacing`` is in that same file order (nnUNet transposes it forward
    itself). Exact when transpose_forward is identity (see caller's guard).

    INVARIANT: This remap is exact only when the sparse input was produced
    with start=0 (first slice index = 0). All CNISP deployment paths
    enforce this via simulation.affine_ops.assert_start_zero.
    """
    assert isinstance(step, int) and step >= 1, f"step must be int>=1, got {step}"
    tf = list(transpose_forward)
    internal_step_axis = tf.index(int(step_axis_orig))

    # Native full shape in internal order (file shape, transposed forward).
    native_full_internal = [int(native_shape_orig[a]) for a in tf]

    sparse_bbox = [list(map(int, b)) for b in sparse_props["bbox_used_for_cropping"]]
    native_bbox = [list(b) for b in sparse_bbox]
    lo, hi = native_bbox[internal_step_axis]
    native_bbox[internal_step_axis] = [
        lo * step,
        min(hi * step, native_full_internal[internal_step_axis]),
    ]
    native_after_crop = [int(hi_ - lo_) for (lo_, hi_) in native_bbox]

    props = dict(sparse_props)
    props["spacing"] = [float(z) for z in native_zooms_orig]  # file order
    props["shape_before_cropping"] = tuple(native_full_internal)
    props["bbox_used_for_cropping"] = native_bbox
    props["shape_after_cropping_and_before_resampling"] = tuple(native_after_crop)
    return props


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
                    help="Recompute masks even if all three outputs exist.")
    args = ap.parse_args()

    cfg = load_yaml(Path(args.config))
    work_dir = Path(cfg["work_dir"])

    sparse_manifest = work_dir / "input" / "sparse_manifest.json"
    if not sparse_manifest.exists():
        print(f"[predict_sparse_iso] missing {sparse_manifest} -- run "
              f"nnunet/data_prep/sparsify_inputs.py first.", file=sys.stderr)
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

    input_root = work_dir / "input"
    pred_root = work_dir / "prediction"
    dense_pred_dir = pred_root / "native"
    pred_root.mkdir(parents=True, exist_ok=True)

    predictor, torch, convert_fn, rw = _init_predictor(cfg)
    transpose_forward = list(predictor.plans_manager.transpose_forward)
    transpose_backward = list(predictor.plans_manager.transpose_backward)
    plan_spacing_internal = list(predictor.configuration_manager.spacing)
    # Cache of per-source native geometry (rw shape/spacing/props + the
    # nibabel<->as-read permutation), filled lazily and reused across steps.
    geom_cache: Dict[str, Tuple] = {}

    # The native-grid synthesis (_native_properties) keeps the sparse crop
    # bookkeeping and scales the through-plane axis. nnUNet records bbox /
    # shape_before_cropping in as-read order; this code is exact when the
    # plan's transpose_forward is identity (internal order == as-read order),
    # which is the usual 3d_fullres case. If it is NOT identity, the per-axis
    # bbox/shape ordering must be re-verified against the native CT overlay.
    if transpose_forward != [0, 1, 2]:
        print(f"[predict_sparse_iso] WARNING: transpose_forward="
              f"{transpose_forward} is non-identity. Native-grid masks assume "
              f"as-read order == internal order; spot-check that each "
              f"sparse_step_XX_native mask overlays its CT correctly.",
              flush=True)

    n_sparse_jobs = sum(
        len(v) for v in sparse_m.get("by_step", {}).values()
    )
    print(f"[predict_sparse_iso] workload: step_01 iso up to "
          f"{len(src_to_path)} source(s); sparse sweep "
          f"{n_sparse_jobs} (source, step) pair(s). "
          f"Each inference is silent for several minutes -- per-case "
          f"progress is logged below.", flush=True)

    out_steps: Dict[str, Dict[str, str]] = {}
    n_inferred = 0
    n_skipped = 0
    issues: List[str] = []

    # ── step_01: dense baseline ────────────────────────────────
    # _native/01 reuses the shared dense pred; _upsampled/01 is the genuine
    # iso prediction of the native CT (same predictor path as the sweep).
    up_01 = pred_root / "sparse_step_01_upsampled"
    native_01 = pred_root / "sparse_step_01_native"
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

        # _upsampled/01 -> iso prediction of the native CT.
        dst_iso = up_01 / f"{sid}.nii.gz"
        if dst_iso.exists() and not args.force:
            print(f"[predict_sparse_iso] step_01 [{i}/{len(step_01_ids)}] "
                  f"{sid}: _upsampled exists -- skip")
            continue
        native_input = input_root / "native" / f"{sid}_0000.nii.gz"
        if not native_input.exists():
            issues.append(f"step_01 {sid}: native input {native_input} missing; "
                          f"iso mask skipped (native baseline still symlinked).")
            continue
        try:
            t0 = time.time()
            print(f"[predict_sparse_iso] step_01 [{i}/{len(step_01_ids)}] "
                  f"{sid}: iso inference ...", flush=True)
            logits, props = _predict_logits(predictor, torch, native_input)
            _, _, _, io2nib = _native_geom(rw, native_input, geom_cache)
            iso_arr = _iso_mask_from_logits(
                logits, torch, transpose_backward, io2nib,
            )
            iso_aff = _iso_affine(
                nib.load(str(native_input)).affine,
                props["bbox_used_for_cropping"],
                plan_spacing_internal,
                transpose_forward,
                io2nib,
            )
            _save_uint8(iso_arr, iso_aff, dst_iso)
            n_inferred += 1
            print(f"[predict_sparse_iso] step_01 [{i}/{len(step_01_ids)}] "
                  f"{sid}: done ({time.time() - t0:.1f}s)", flush=True)
        except Exception as e:  # noqa: BLE001
            issues.append(f"step_01 {sid}: iso inference failed ({e})")
    if step_01_map:
        out_steps["01"] = step_01_map
        print(f"[predict_sparse_iso] step_01: {len(step_01_map)} dense "
              f"baseline(s) -> _native (symlink) + _upsampled (iso)")

    # ── steps >= 2: sparse-CT sweep ─────────────────────────────
    for step_tag in sorted(sparse_m.get("by_step", {}).keys()):
        step = int(step_tag)
        sparse_dir = pred_root / f"sparse_step_{step_tag}"
        up_dir = pred_root / f"sparse_step_{step_tag}_upsampled"
        native_dir = pred_root / f"sparse_step_{step_tag}_native"
        step_map: Dict[str, str] = {}
        step_entries = sorted(sparse_m["by_step"][step_tag].items())
        n_step = len(step_entries)
        print(f"[predict_sparse_iso] step_{step_tag}: {n_step} source(s) "
              f"-> sparse + _upsampled(iso) + _native", flush=True)

        for j, (sid, info) in enumerate(step_entries, 1):
            sparse_input = Path(info["input"])
            if not sparse_input.exists():
                issues.append(f"step_{step_tag} {sid}: sparse input missing "
                              f"{sparse_input}")
                continue

            dst_sparse = sparse_dir / f"{sid}.nii.gz"
            dst_iso = up_dir / f"{sid}.nii.gz"
            dst_native = native_dir / f"{sid}.nii.gz"
            if (not args.force and dst_sparse.exists()
                    and dst_iso.exists() and dst_native.exists()):
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
            # Native geometry in nnUNet's as-read order (the same convention
            # every nnUNet segmentation comes back in). The sparse input was
            # written from this CT preserving orientation, so they share the
            # nibabel<->as-read permutation.
            (native_rw_shape, native_rw_spacing,
             native_rw_props, io2nib) = _native_geom(rw, ct_path, geom_cache)
            # step_axis from the manifest is a nibabel voxel-axis index; map
            # it to nnUNet's as-read order for the native-grid synthesis.
            nib_step_axis = int(info["step_axis"])
            rw_step_axis = io2nib[nib_step_axis]

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

                # 2) iso plan-spacing mask (genuine network output).
                # Reference-only: saved with a hand-built nibabel affine, so
                # the array is converted to nibabel order inside the helpers.
                iso_arr = _iso_mask_from_logits(
                    logits, torch, transpose_backward, io2nib,
                )
                iso_aff = _iso_affine(
                    nib.load(str(sparse_input)).affine,
                    props["bbox_used_for_cropping"],
                    plan_spacing_internal,
                    transpose_forward,
                    io2nib,
                )
                _save_uint8(iso_arr, iso_aff, dst_iso)

                # 3) native-grid mask via nnUNet's own resampler.
                # Position-exactness invariant: sparse inputs are produced
                # with start=0, so sparse voxel i maps to native voxel i*step.
                # This makes the bbox scaling below exact. Enforce both the
                # mode and start=0 invariants from the manifest.
                assert info.get("mode", "thin") in ("thin", "thick"), (
                    f"step_{step_tag} {sid}: unexpected mode={info.get('mode')}"
                )
                # Legacy manifests pre-date the "start" field; default to 0
                # (which was the only value ever produced).
                _assert_start_zero(int(info.get("start", 0)))
                # All geometry args are in nnUNet as-read order, matching the
                # sparse props (also as-read), so _native_properties stays
                # self-consistent. The shape assert below is a hard guard:
                # if anything is mis-ordered the source is skipped loudly
                # rather than silently writing a transposed Dice target.
                native_props = _native_properties(
                    props, native_rw_shape, native_rw_spacing,
                    step, rw_step_axis, transpose_forward,
                )
                native_seg = _segmentation_from_logits(
                    convert_fn, predictor, logits.clone(), native_props,
                )
                if tuple(native_seg.shape) != native_rw_shape:
                    issues.append(
                        f"step_{step_tag} {sid}: native mask shape "
                        f"{native_seg.shape} != native CT {native_rw_shape} "
                        f"(as-read order). Axis-order/scale bug -- NOT writing."
                    )
                    continue
                dst_native.parent.mkdir(parents=True, exist_ok=True)
                rw.write_seg(native_seg.astype(np.uint8), str(dst_native),
                             native_rw_props)

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

    sweep_manifest_path = pred_root / "sweep_manifest.json"
    with open(sweep_manifest_path, "w") as f:
        json.dump({"steps": out_steps}, f, indent=2)

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
