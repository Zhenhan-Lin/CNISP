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
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np

# Make ``nnunet.*`` importable when run as ``python nnunet/engine/...``.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nnunet.helpers.config import load_yaml  # noqa: E402


def _import_nnunet():
    """Import the nnUNetv2 bits we need, failing loudly if absent."""
    try:
        import torch  # noqa: F401
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        from nnunetv2.inference.export_prediction import (
            convert_predicted_logits_to_segmentation_with_correct_shape,
        )
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            f"[predict_sparse_iso] could not import nnUNetv2 / torch: {e}\n"
            f"  This script needs the same environment that runs "
            f"nnUNetv2_predict. Activate it and retry."
        )
    import torch  # noqa: F811
    return (
        torch,
        nnUNetPredictor,
        convert_predicted_logits_to_segmentation_with_correct_shape,
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
    torch, nnUNetPredictor, _convert = _import_nnunet()
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
    print(f"[predict_sparse_iso] model:  {model_folder}")
    print(f"[predict_sparse_iso] folds:  {folds}  device: {device}")
    return predictor, torch, _convert


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
    logits, torch, transpose_backward: List[int],
) -> np.ndarray:
    """Plan-spacing prediction == argmax over classes, in file axis order.

    Labels {0:bg, 1:ON, 2:recti, 3:globe, 4:fat} are consecutive and the
    dataset is not region-based, so a plain argmax reproduces nnUNet's
    label values without the region-merge logic.
    """
    seg_internal = torch.argmax(logits, dim=0).cpu().numpy().astype(np.uint8)
    return np.ascontiguousarray(seg_internal.transpose(transpose_backward))


def _iso_affine(
    sparse_affine: np.ndarray,
    sparse_bbox_internal: List,
    plan_spacing_internal: List[float],
    transpose_forward: List[int],
) -> np.ndarray:
    """Affine for the cropped iso prediction in the sparse CT's world frame.

    The iso volume is nnUNet's nonzero-crop of the sparse CT resampled to
    the plan (iso) spacing. Its voxel (0,0,0) is the crop's lower corner
    in sparse-voxel space; its direction matches the sparse CT with each
    axis rescaled to the plan spacing.
    """
    bbox_lower_internal = [lo for (lo, _hi) in sparse_bbox_internal]
    bbox_lower_orig = _internal_to_original(bbox_lower_internal, transpose_forward)
    p_orig = _internal_to_original(list(plan_spacing_internal), transpose_forward)

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
    All shape/bbox fields are in nnUNet's internal (transposed) order;
    ``spacing`` is in file order (nnUNet transposes it forward itself).
    """
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

    predictor, torch, convert_fn = _init_predictor(cfg)
    transpose_forward = list(predictor.plans_manager.transpose_forward)
    transpose_backward = list(predictor.plans_manager.transpose_backward)
    plan_spacing_internal = list(predictor.configuration_manager.spacing)

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
    for sid in sorted(src_to_path):
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
            continue
        native_input = input_root / "native" / f"{sid}_0000.nii.gz"
        if not native_input.exists():
            issues.append(f"step_01 {sid}: native input {native_input} missing; "
                          f"iso mask skipped (native baseline still symlinked).")
            continue
        try:
            logits, props = _predict_logits(predictor, torch, native_input)
            iso_arr = _iso_mask_from_logits(logits, torch, transpose_backward)
            iso_aff = _iso_affine(
                nib.load(str(native_input)).affine,
                props["bbox_used_for_cropping"],
                plan_spacing_internal,
                transpose_forward,
            )
            _save_uint8(iso_arr, iso_aff, dst_iso)
            n_inferred += 1
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

        for sid, info in sorted(sparse_m["by_step"][step_tag].items()):
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
                continue

            ct_info = src_to_path.get(sid)
            if not ct_info or "ct_image_path" not in ct_info:
                issues.append(f"step_{step_tag} {sid}: no ct_image_path in "
                              f"source_to_path.json")
                continue
            native_ct = nib.load(str(ct_info["ct_image_path"]))
            native_shape = tuple(int(x) for x in native_ct.shape[:3])
            native_zooms = tuple(float(z) for z in native_ct.header.get_zooms()[:3])
            step_axis = int(info["step_axis"])

            try:
                logits, props = _predict_logits(predictor, torch, sparse_input)

                # 1) sparse-grid mask (nnUNet's normal output).
                sparse_seg = _segmentation_from_logits(
                    convert_fn, predictor, logits.clone(), props,
                )
                sparse_img = nib.load(str(sparse_input))
                if tuple(sparse_seg.shape) != tuple(int(x) for x in sparse_img.shape[:3]):
                    issues.append(
                        f"step_{step_tag} {sid}: sparse mask shape "
                        f"{sparse_seg.shape} != sparse input {sparse_img.shape[:3]}"
                    )
                    continue
                _save_uint8(sparse_seg, sparse_img.affine, dst_sparse)

                # 2) iso plan-spacing mask (genuine network output).
                iso_arr = _iso_mask_from_logits(logits, torch, transpose_backward)
                iso_aff = _iso_affine(
                    sparse_img.affine,
                    props["bbox_used_for_cropping"],
                    plan_spacing_internal,
                    transpose_forward,
                )
                _save_uint8(iso_arr, iso_aff, dst_iso)

                # 3) native-grid mask via nnUNet's own resampler.
                native_props = _native_properties(
                    props, native_shape, native_zooms,
                    step, step_axis, transpose_forward,
                )
                native_seg = _segmentation_from_logits(
                    convert_fn, predictor, logits.clone(), native_props,
                )
                if tuple(native_seg.shape) != native_shape:
                    issues.append(
                        f"step_{step_tag} {sid}: native mask shape "
                        f"{native_seg.shape} != native CT {native_shape}. "
                        f"Axis-order/scale bug -- NOT writing this source."
                    )
                    continue
                _save_uint8(native_seg, native_ct.affine, dst_native)

                step_map[sid] = dst_native.name
                n_inferred += 1
            except Exception as e:  # noqa: BLE001
                issues.append(f"step_{step_tag} {sid}: inference failed ({e})")
                continue

        if step_map:
            out_steps[step_tag] = step_map
            print(f"[predict_sparse_iso] step_{step_tag}: {len(step_map)} "
                  f"source(s) -> sparse + _upsampled(iso) + _native")

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
