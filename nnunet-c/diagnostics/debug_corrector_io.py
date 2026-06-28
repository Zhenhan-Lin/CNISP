#!/usr/bin/env python3
"""Toy diagnostic: dump every intermediate variable that decides whether the
corrector actually USES its prelabel channels (ch1..ch4) at test time.

Motivation
----------
Symptom: corrector val-Dice ~0.94 during training, but test Dice stuck ~0.6
(fragmented), and (per observation) the GRID=gt test output is byte-identical to
the GRID=iso output. If two builds that differ in ch1..ch4 yield the SAME
prediction, the network is almost certainly NOT responding to ch1..ch4 at test.

This script prints, in one place, the variables that localize that:

  [A] plan + dataset.json: channel_names, normalization per channel, target
      spacing, #input channels.  -> how predict treats each channel.
  [B] raw built 5-ch inputs (imagesTr train case AND imagesTs test case):
      per-channel shape/spacing/min/max/mean/#nonzero/#unique.
      -> confirms ch1..4 carry the prelabel in BOTH builds.
  [C] nnUNet-PREPROCESSED train tensor (what the net actually trained on):
      per-channel stats from $nnUNet_preprocessed/.../<case>.npy|.npz.
      -> are ch1..4 binary {0,1} after preprocessing, or splined/zeroed?
  [D] FIRST-CONV per-input-channel weight L2 norm from the finetuned checkpoint.
      -> if ch1..4 norms << ch0, the model ignores the prelabel (THE smoking gun
         for gt==iso==baseline).
  [E] (optional, --run-predict-preproc) run nnUNet's predict-time preprocessing
      on the test imagesTs case and print per-channel stats. Compare to [C].

Nothing here mutates state; it only reads. Run on the GPU box where nnUNet env
($nnUNet_raw / $nnUNet_preprocessed / $nnUNet_results) + nnunetv2 are available.

Example
-------
    python nnunet-c/diagnostics/debug_corrector_io.py \
        --dataset-id 845 --dataset-name PHOTON_CT_CORR_C_cnisp \
        --plans nnUNetPlansFinetune --config 3d_fullres \
        --trainer nnUNetTrainer --fold 0 --chk checkpoint_best.pth \
        --train-case corr_chk_14455_step03 \
        --test-images nnunet-c/test_input/PHOTON_CT_CORR_C_cnisp/imagesTs \
        --test-case-id corr_chk_14455_step03 \
        --run-predict-preproc
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def _sep(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _chan_stats(arr: np.ndarray, name: str) -> None:
    a = np.asarray(arr)
    uniq = np.unique(a)
    nz = int((a != 0).sum())
    is_binary = set(np.unique(a).tolist()) <= {0, 1} or (
        uniq.size <= 2 and np.allclose(uniq, np.round(uniq))
    )
    print(f"    {name:12s} shape={tuple(a.shape)} dtype={a.dtype} "
          f"min={float(a.min()):.4f} max={float(a.max()):.4f} "
          f"mean={float(a.mean()):.4f} nonzero={nz} "
          f"n_unique={uniq.size} binary={is_binary} "
          f"uniq_head={np.round(uniq[:6], 4).tolist()}")


def _voxel_spacing_from_affine(affine: np.ndarray) -> np.ndarray:
    return np.sqrt((np.asarray(affine)[:3, :3] ** 2).sum(axis=0))


# ── [A] plan + dataset.json ─────────────────────────────────────────────
def dump_plan_and_dataset(preproc_root: Path, ds_folder: str, plans: str,
                          config: str) -> dict:
    _sep("[A] plan + dataset.json (how predict treats each channel)")
    out = {}
    plan_json = preproc_root / ds_folder / f"{plans}.json"
    ds_json = preproc_root / ds_folder / "dataset.json"
    if plan_json.is_file():
        plan = json.load(open(plan_json))
        cfg = plan.get("configurations", {}).get(config, {})
        print(f"  plan: {plan_json}")
        print(f"    configuration={config}")
        print(f"    spacing            = {cfg.get('spacing')}")
        print(f"    patch_size         = {cfg.get('patch_size')}")
        print(f"    normalization_schemes        = {cfg.get('normalization_schemes')}")
        print(f"    use_mask_for_norm            = {cfg.get('use_mask_for_norm')}")
        print(f"    resampling_fn_data           = {cfg.get('resampling_fn_data')}")
        print(f"    resampling_fn_seg            = {cfg.get('resampling_fn_seg')}")
        fp = plan.get("foreground_intensity_properties_per_channel", {})
        print(f"    fg_intensity_props channels  = {sorted(fp.keys())}")
        for k in sorted(fp.keys()):
            v = fp[k]
            print(f"      ch{k}: mean={v.get('mean'):.3f} std={v.get('std'):.3f} "
                  f"p0.5={v.get('percentile_00_5')} p99.5={v.get('percentile_99_5')}")
        out["plan"] = plan
    else:
        print(f"  [!] plan json not found: {plan_json}")
    if ds_json.is_file():
        dj = json.load(open(ds_json))
        print(f"  dataset.json: {ds_json}")
        print(f"    channel_names = {dj.get('channel_names') or dj.get('modality')}")
        print(f"    labels        = {dj.get('labels')}")
        print(f"    numTraining   = {dj.get('numTraining')}")
        print(f"    file_ending   = {dj.get('file_ending')}")
        out["dataset_json"] = dj
    else:
        print(f"  [!] dataset.json not found: {ds_json}")
    return out


# ── [B] raw built 5-ch inputs ───────────────────────────────────────────
def dump_raw_channels(label: str, images_dir: Path, case_id: str,
                      n_channels: int = 5, file_ending: str = ".nii.gz") -> None:
    _sep(f"[B] raw built channels ({label}): {case_id}  in {images_dir}")
    try:
        import nibabel as nib
    except Exception as e:  # noqa: BLE001
        print(f"  [!] nibabel unavailable: {e}")
        return
    for c in range(n_channels):
        p = images_dir / f"{case_id}_{c:04d}{file_ending}"
        if not p.is_file():
            print(f"    ch{c}: MISSING {p}")
            continue
        img = nib.load(str(p))
        arr = np.asanyarray(img.dataobj)
        sp = _voxel_spacing_from_affine(img.affine)
        print(f"    ch{c} ({p.name}) spacing={np.round(sp,3).tolist()}")
        _chan_stats(arr, f"ch{c}")


# ── [C] nnUNet-preprocessed train tensor ────────────────────────────────
def dump_preprocessed_train(preproc_root: Path, ds_folder: str, plans: str,
                            config: str, case_id: str) -> None:
    _sep(f"[C] PREPROCESSED TRAIN tensor (what the net trained on): {case_id}")
    folder = preproc_root / ds_folder / f"{plans}_{config}"
    npy = folder / f"{case_id}.npy"
    npz = folder / f"{case_id}.npz"
    data = None
    if npy.is_file():
        print(f"  loading {npy}")
        data = np.load(str(npy))
    elif npz.is_file():
        print(f"  loading {npz} (key 'data')")
        with np.load(str(npz)) as z:
            data = z["data"]
    else:
        print(f"  [!] no preprocessed tensor for {case_id} under {folder}")
        print(f"      (looked for {npy.name} / {npz.name}); "
              f"is the case in this dataset + were the .npz unpacked?")
        return
    print(f"  data shape = {data.shape}  (expected [C, X, Y, Z])")
    for c in range(data.shape[0]):
        _chan_stats(data[c], f"ch{c}")
    print("  >>> KEY: are ch1..ch4 still binary {0,1} here? If they are SPLINED "
          "(continuous) or all-zero, the train-time prelabel itself is wrong.")


# ── [D] first-conv per-input-channel weight norms ───────────────────────
def dump_first_conv_weights(results_root: Path, ds_folder: str, plans: str,
                            config: str, trainer: str, fold: int, chk: str,
                            n_in: int = 5) -> None:
    _sep(f"[D] first-conv per-input-channel weight L2 norm  ({chk})")
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        print(f"  [!] torch unavailable: {e}")
        return
    ckpt_path = (results_root / ds_folder
                 / f"{trainer}__{plans}__{config}" / f"fold_{fold}" / chk)
    if not ckpt_path.is_file():
        print(f"  [!] checkpoint not found: {ckpt_path}")
        return
    print(f"  loading {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = (ckpt.get("network_weights")
          or ckpt.get("state_dict")
          or ckpt.get("model")
          or ckpt)
    # Find the first conv weight whose in-channels == n_in (the stem conv).
    cand = None
    for k, v in sd.items():
        if hasattr(v, "ndim") and v.ndim in (4, 5) and int(v.shape[1]) == n_in:
            cand = (k, v)
            break
    if cand is None:
        print(f"  [!] no conv weight with in_channels=={n_in} found. "
              f"First few weight keys/shapes:")
        for i, (k, v) in enumerate(sd.items()):
            if hasattr(v, "ndim") and v.ndim >= 2:
                print(f"      {k}  shape={tuple(v.shape)}")
            if i > 30:
                break
        return
    k, w = cand
    w = w.detach().float()
    print(f"  stem conv weight: {k}  shape={tuple(w.shape)} "
          f"(out={w.shape[0]}, in={w.shape[1]}, kernel={tuple(w.shape[2:])})")
    print("  per-INPUT-channel weight L2 norm (sum over out + kernel dims):")
    norms = []
    for c in range(w.shape[1]):
        n = float(w[:, c].norm().item())
        norms.append(n)
        print(f"    in-ch{c}: |W| = {n:.5f}")
    if norms:
        ratio = (np.mean(norms[1:]) / norms[0]) if norms[0] > 0 else float("inf")
        print(f"  ch1..ch{n_in-1} mean / ch0 = {ratio:.4f}")
        print("  >>> KEY: if ch1..ch4 norms are ~0 (or << ch0), the corrector "
              "IGNORES the prelabel -> output collapses to the CT-only baseline "
              "regardless of what ch1..4 contain (explains gt==iso==fragmented).")


# ── [E] predict-time preprocessing (optional) ───────────────────────────
def dump_predict_preproc(preproc_root: Path, ds_folder: str, plans: str,
                         config: str, images_dir: Path, case_id: str,
                         n_channels: int = 5, file_ending: str = ".nii.gz") -> None:
    _sep(f"[E] PREDICT-time preprocessing on test case: {case_id}")
    try:
        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
        from nnunetv2.preprocessing.preprocessors.default_preprocessor import (
            DefaultPreprocessor,
        )
    except Exception as e:  # noqa: BLE001
        print(f"  [!] nnunetv2 import failed ({e}); skip predict-preproc.")
        return
    plan_json = preproc_root / ds_folder / f"{plans}.json"
    ds_json = preproc_root / ds_folder / "dataset.json"
    if not (plan_json.is_file() and ds_json.is_file()):
        print(f"  [!] need {plan_json} and {ds_json}")
        return
    plans_manager = PlansManager(str(plan_json))
    dataset_json = json.load(open(ds_json))
    cm = plans_manager.get_configuration(config)
    image_files = [str(images_dir / f"{case_id}_{c:04d}{file_ending}")
                   for c in range(n_channels)]
    missing = [f for f in image_files if not Path(f).is_file()]
    if missing:
        print(f"  [!] missing channel files: {missing}")
        return
    pp = DefaultPreprocessor(verbose=True)
    print(f"  running DefaultPreprocessor.run_case on {len(image_files)} channels")
    print("  NOTE: this uses whatever resampling_fn the plan references. If the "
          "predict wrapper installs a custom per-channel resampler, replicate "
          "that here too for a faithful comparison.")
    data, seg, props = pp.run_case(image_files, None, plans_manager, cm, dataset_json)
    print(f"  preprocessed predict data shape = {data.shape}")
    for c in range(data.shape[0]):
        _chan_stats(data[c], f"ch{c}")
    print("  >>> KEY: compare these ch1..ch4 stats to section [C]. If [C] is "
          "binary but [E] is splined/zeroed (or vice-versa), train and predict "
          "feed the net DIFFERENT prelabel channels -> the mismatch.")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nnunet-preprocessed", default=os.environ.get("nnUNet_preprocessed"))
    ap.add_argument("--nnunet-results", default=os.environ.get("nnUNet_results"))
    ap.add_argument("--dataset-id", type=int, required=True)
    ap.add_argument("--dataset-name", required=True)
    ap.add_argument("--plans", default="nnUNetPlansFinetune")
    ap.add_argument("--config", default="3d_fullres")
    ap.add_argument("--trainer", default="nnUNetTrainer")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--chk", default="checkpoint_best.pth")
    ap.add_argument("--n-channels", type=int, default=5)
    ap.add_argument("--file-ending", default=".nii.gz")
    # raw train build (imagesTr) — optional
    ap.add_argument("--train-images", default=None,
                    help="dir with train _0000.._000N (default: $nnUNet_raw/<ds>/imagesTr)")
    ap.add_argument("--train-case", default=None,
                    help="preprocessed/raw TRAIN case id, e.g. corr_chk_14455_step03")
    # raw test build (imagesTs) — optional
    ap.add_argument("--test-images", default=None, help="dir with test _0000.._000N")
    ap.add_argument("--test-case-id", default=None, help="test case id (stem)")
    ap.add_argument("--run-predict-preproc", action="store_true",
                    help="also run nnUNet predict-time preprocessing on the test case")
    args = ap.parse_args()

    ds_folder = f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    preproc_root = Path(args.nnunet_preprocessed) if args.nnunet_preprocessed else None
    results_root = Path(args.nnunet_results) if args.nnunet_results else None

    print(f"dataset folder : {ds_folder}")
    print(f"nnUNet_preprocessed : {preproc_root}")
    print(f"nnUNet_results      : {results_root}")

    # [A]
    if preproc_root:
        dump_plan_and_dataset(preproc_root, ds_folder, args.plans, args.config)

    # [B] train raw
    if args.train_case:
        train_dir = (Path(args.train_images) if args.train_images else
                     (Path(os.environ.get("nnUNet_raw", ".")) / ds_folder / "imagesTr"))
        dump_raw_channels("TRAIN imagesTr", train_dir, args.train_case,
                          args.n_channels, args.file_ending)
    # [B] test raw
    if args.test_case_id and args.test_images:
        dump_raw_channels("TEST imagesTs", Path(args.test_images), args.test_case_id,
                          args.n_channels, args.file_ending)

    # [C] preprocessed train
    if preproc_root and args.train_case:
        dump_preprocessed_train(preproc_root, ds_folder, args.plans, args.config,
                                args.train_case)

    # [D] first conv weights
    if results_root:
        dump_first_conv_weights(results_root, ds_folder, args.plans, args.config,
                                args.trainer, args.fold, args.chk, args.n_channels)

    # [E] predict preproc
    if args.run_predict_preproc and preproc_root and args.test_images and args.test_case_id:
        dump_predict_preproc(preproc_root, ds_folder, args.plans, args.config,
                             Path(args.test_images), args.test_case_id,
                             args.n_channels, args.file_ending)

    print("\n[done] Read the [D] norms and [C]-vs-[E] channel stats first.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
