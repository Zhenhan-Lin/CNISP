#!/usr/bin/env python3
"""nnUNet predictor construction + per-case inference helpers.

The *calculation* layer behind ``nnunet/predict_sparse_iso.py``: it locates
the trained model, builds an ``nnUNetPredictor``, resolves the
nnUNet<->nibabel axis order per file, and turns a preprocessed case into
plan-spacing logits / a segmentation. ``predict_sparse_iso.run`` orchestrates
these into the two-mask-per-(source, step) sweep; the world-aware native
resample lives in :mod:`nnunet.lib.native_resample`.

Imports torch + nnunetv2 at module load, exactly like the original inline
predictor did -- this module is only imported by the predict entry point,
which always runs inside the nnUNet environment.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np

import torch  # noqa: F401
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.export_prediction import (
    convert_predicted_logits_to_segmentation_with_correct_shape as _convert,
)


def resolve_model_folder(cfg: Dict) -> Path:
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


def resolve_folds(model_folder: Path, folds_cfg, checkpoint_name: str):
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


def init_predictor(cfg: Dict):
    """Build the nnUNetPredictor + reader/writer. Returns ``(predictor, torch,
    convert_fn, rw)`` matching the original inline helper's tuple."""
    gpu_id = cfg.get("gpu_id", 0)
    # Respect the config GPU the same way the shell scripts do.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(gpu_id))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_folder = resolve_model_folder(cfg)
    folds = resolve_folds(
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


def detect_io2nib(rw, path) -> List[int]:
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
            # Degenerate (>=2 axes share shape AND spacing): fall back to the
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


def native_geom(rw, ct_path, cache: Dict) -> Tuple:
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
        io2nib = detect_io2nib(rw, ct_path)
        nimg = nib.load(str(ct_path))
        nib_shape = tuple(int(x) for x in nimg.shape[:3])
        nib_affine = np.asarray(nimg.affine, dtype=np.float64)
        cache[key] = (rw_shape, rw_spacing, props, io2nib, nib_shape, nib_affine)
    return cache[key]


def predict_logits(predictor, torch, input_file: Path):
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


def segmentation_from_logits(convert_fn, predictor, logits, properties):
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
