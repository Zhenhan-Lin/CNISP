#!/usr/bin/env python3
"""Verify: is "native-spacing decode + resample to 0.5" ~= "direct 0.5 decode"?

WHY
---
The corrector's ch1..ch4 today are produced by CNISP decoding the test-optimised
latent at the case's NATIVE (pre-degradation) spacing, inverting to native, then
(order-0) resampling toward the iso-0.5 plan. The proposed change decodes the
SAME latent DIRECTLY at iso-0.5. Both paths share the identical flip / reorient /
sub-patch placement geometry, so the ONLY variable is the decode spacing plus one
nearest-neighbour resample. This script isolates exactly that variable IN THE
CANONICAL 64 mm SUB-PATCH FRAME (no inversion needed), so it answers:

    Does switching test's ch1..ch4 to a direct 0.5 query change what the
    corrector sees vs the (already trained) native path?  -> Dice / disagreement.

If Dice ~= 1.0 and disagreement ~= 0 across cases: the switch is train/test-safe
(no retrain). If not: STOP and report -- a real distribution shift.

This script ONLY READS CNISP artifacts (latent .npy + alignment metadata + the
frozen model). It does NOT modify or invoke CNISP's own test pipeline.

WHAT IT COMPARES (per saved latent, in the canonical sub-patch frame)
--------------------------------------------------------------------
  A = predict_dense(latent, spacing=native_sp)  -> order-0 zoom to the 0.5 grid
  B = predict_dense(latent, spacing=[iso,iso,iso])
  -> per-structure Dice(A, B) + fraction of disagreeing voxels.

native_sp comes from the alignment metadata's ``patch_spacing`` (the canonical
patch spacing == the sub-patch spacing, since inner_crop_64mm only crops). The
saved latent is already the Delta-corrected alpha_hat (optimize_latent applies
Delta before returning), so decoding it with no Delta reproduces the prediction.

Both decodes are in CANONICAL labels {1:ON,2:Globe,3:Fat,4:Recti}; Dice is
per-class so the scheme does not matter here.

Usage
-----
    python nnunet-c/diagnostics/verify_ch14_iso_vs_native.py \
        --paths        orbital_shape_prior_st1/configs/paths.yaml \
        --train-config orbital_shape_prior_st1/configs/train_v6_5_gt.yaml \
        --model-name   orbital_ad_v6_5_gt --checkpoint latest \
        --latent-dir   nnunet-c/data/cnisp_pred/latent \
        --aligned-dir  nnunet-c/data/aligned_patch \
        --iso-mm 0.5 --max-cases 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CNISP_DIR = REPO_ROOT / "orbital_shape_prior_st1"
for _p in (str(CNISP_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402
from scipy import ndimage  # noqa: E402

from engine.train import create_model  # noqa: E402
from engine.infer import load_model_checkpoint, _AUTOCAST_DTYPE  # noqa: E402

# Canonical foreground labels (multiclass_ad decode output).
CANON = {"ON": 1, "Globe": 2, "Fat": 3, "Recti": 4}


def _parse_latent_name(p: Path) -> Optional[Tuple[str, int]]:
    """``chk_14455_OD_step03.npy`` -> ("chk_14455_OD", 3)."""
    stem = p.name[:-4] if p.name.endswith(".npy") else p.name
    if "_step" not in stem:
        return None
    casename, step_tag = stem.rsplit("_step", 1)
    try:
        return casename, int(step_tag)
    except ValueError:
        return None


def _find_meta(casename: str, meta_dirs: List[Path]) -> Optional[dict]:
    for d in meta_dirs:
        p = d / f"{casename}.json"
        if p.is_file():
            return json.load(open(p))
    return None


def _decode(net, latent_t: torch.Tensor, spacing_mm) -> np.ndarray:
    """predict_dense at the given (per-axis) spacing -> int label volume."""
    sp = torch.as_tensor(spacing_mm, dtype=torch.float32)
    image_size = net.image_size.detach().cpu().float()
    target_shape = torch.round(image_size / sp).long()
    out = net.predict_dense(
        latent_t, target_shape.to(latent_t.device), sp.to(latent_t.device),
        autocast_dtype=_AUTOCAST_DTYPE,
    )
    return out.cpu().numpy().astype(np.int16)


def _zoom_to_shape(vol: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    """Order-0 (nearest) resample of a label volume to ``target_shape``.

    Mimics the corrector path's native->0.5 step (nnUNet's segmentation
    resampler is order 0). zoom output rounding is then exactly cropped/padded
    to target_shape so A and B live on an identical grid for voxel Dice.
    """
    zoom = [t / s for t, s in zip(target_shape, vol.shape)]
    z = ndimage.zoom(vol, zoom=zoom, order=0)
    out = np.zeros(target_shape, dtype=vol.dtype)
    sl = tuple(slice(0, min(z.shape[a], target_shape[a])) for a in range(3))
    out[sl] = z[sl]
    return out


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    pa, pb = float(a.sum()), float(b.sum())
    denom = pa + pb
    if denom == 0.0:
        return 1.0
    inter = float(np.logical_and(a, b).sum())
    return 2.0 * inter / denom


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paths", default=str(CNISP_DIR / "configs" / "paths.yaml"))
    ap.add_argument("--train-config",
                    default=str(CNISP_DIR / "configs" / "train_v6_5_gt.yaml"))
    ap.add_argument("--model-name", default="orbital_ad_v6_5_gt")
    ap.add_argument("--model-basedir", default=None,
                    help="override model_basedir from paths.yaml")
    ap.add_argument("--checkpoint", default="latest", choices=["best", "latest"])
    ap.add_argument("--latent-dir",
                    default=str(REPO_ROOT / "nnunet-c" / "data" / "cnisp_pred" / "latent"),
                    help="dir of saved latents ({casename}_step{XX}.npy)")
    ap.add_argument("--aligned-dir",
                    default=str(REPO_ROOT / "nnunet-c" / "data" / "aligned_patch"),
                    help="aligned_dir holding metadata*/ trees (for patch_spacing)")
    ap.add_argument("--meta-subdirs", default="metadata_dataset835,metadata",
                    help="comma list of metadata subdirs to search under aligned-dir")
    ap.add_argument("--iso-mm", type=float, default=0.5)
    ap.add_argument("--max-cases", type=int, default=0, help="0 = all")
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    params: dict = {}
    for y in (args.paths, args.train_config):
        with open(y) as f:
            params.update(yaml.safe_load(f) or {})
    if args.model_basedir:
        params["model_basedir"] = args.model_basedir
    params["model_name"] = args.model_name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = Path(params["model_basedir"]) / args.model_name
    model_state, _ = load_model_checkpoint(model_dir, args.checkpoint, verbose=True)
    net = create_model(params, torch.ones(3)).to(device).eval()
    net.load_state_dict(model_state["net"], strict=True)

    latent_dir = Path(args.latent_dir)
    meta_dirs = [Path(args.aligned_dir) / s.strip()
                 for s in args.meta_subdirs.split(",") if s.strip()]
    lat_files = sorted(latent_dir.glob("*.npy"))
    if not lat_files:
        print(f"[verify] no latents under {latent_dir}. Generate them first "
              f"(032 / 03_infer now save one per (case,step)).", file=sys.stderr)
        return 2

    print("=" * 70)
    print(f"[verify] model={args.model_name} ckpt={args.checkpoint} device={device}")
    print(f"[verify] iso-mm={args.iso_mm}  latents={len(lat_files)} in {latent_dir}")
    print(f"[verify] compare: decode@native + nearest->iso  vs  decode@iso")
    print("=" * 70)

    rows = []
    per_struct = {k: [] for k in CANON}
    disagree_fracs = []
    n_done = n_skip = 0
    for lf in lat_files:
        if args.max_cases and n_done >= args.max_cases:
            break
        parsed = _parse_latent_name(lf)
        if parsed is None:
            n_skip += 1
            continue
        casename, step = parsed
        meta = _find_meta(casename, meta_dirs)
        if meta is None:
            print(f"  {lf.name}: no metadata in {[str(d) for d in meta_dirs]}; skip")
            n_skip += 1
            continue
        native_sp = [float(x) for x in meta["patch_spacing"]]

        latent_np = np.load(str(lf)).astype(np.float32).reshape(1, -1)
        latent_t = torch.from_numpy(latent_np).to(device)

        a_native = _decode(net, latent_t, native_sp)          # native sub-patch
        b_iso = _decode(net, latent_t, [args.iso_mm] * 3)     # 0.5 sub-patch
        a_on_iso = _zoom_to_shape(a_native, b_iso.shape)      # native -> 0.5 grid

        dices = []
        for name, lab in CANON.items():
            d = _dice(a_on_iso == lab, b_iso == lab)
            per_struct[name].append(d)
            dices.append(d)
        fg = (a_on_iso > 0) | (b_iso > 0)
        disagree = float((a_on_iso != b_iso)[fg].sum()) / max(int(fg.sum()), 1)
        disagree_fracs.append(disagree)

        row = {"casename": casename, "step": step,
               "native_sp": "x".join(f"{s:.2f}" for s in native_sp),
               "dice_mean": round(float(np.mean(dices)), 5),
               "disagree_frac": round(disagree, 5)}
        for name, d in zip(CANON, dices):
            row[f"dice_{name}"] = round(d, 5)
        rows.append(row)
        detail = " ".join(f"{n}={d:.3f}" for n, d in zip(CANON, dices))
        print(f"  {casename} step{step:02d} sp=[{row['native_sp']}]: "
              f"{detail} mean={row['dice_mean']:.3f} disagree={disagree:.4f}")
        n_done += 1

    if not rows:
        print("[verify] no comparable cases.", file=sys.stderr)
        return 1

    print("-" * 70)
    print(f"[verify] cases={len(rows)} skipped={n_skip}")
    for name in CANON:
        vals = per_struct[name]
        print(f"  {name:6s}: mean Dice = {np.mean(vals):.4f}  "
              f"min = {np.min(vals):.4f}  (n={len(vals)})")
    overall = float(np.mean([np.mean(per_struct[n]) for n in CANON]))
    print(f"  {'MEAN':6s}: mean Dice = {overall:.4f}")
    print(f"  disagree voxel frac: mean={np.mean(disagree_fracs):.5f} "
          f"max={np.max(disagree_fracs):.5f}")
    print("-" * 70)
    if overall >= 0.99 and np.max(disagree_fracs) <= 0.02:
        print("[verify] VERDICT: native and direct-0.5 ch1..ch4 are ~identical "
              "-> switching test to direct-0.5 is train/test-SAFE (no retrain).")
    else:
        print("[verify] VERDICT: NON-NEGLIGIBLE difference -> STOP. Switching "
              "test would shift ch1..ch4 vs the trained corrector; reassess.")

    if args.out_csv:
        import csv
        out = Path(args.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        fields = (["casename", "step", "native_sp"]
                  + [f"dice_{n}" for n in CANON]
                  + ["dice_mean", "disagree_frac"])
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"[verify] per-case CSV -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
