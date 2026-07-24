#!/usr/bin/env python3
"""Shared evaluation primitives for the corrector (thickness) and FOV-completion
evaluators (revised-plan §11). Both eval_corrector.py and fov_completion_eval.py
reuse the SAME resampling + metric implementations so thickness and FOV analysis
never diverge.

Two layers:
  * PURE metric math (numpy only) — fully unit-tested here (``--self-test``):
      _dice, compute_region_confusion, compute_volume_metrics, compute_whole_metrics
  * FILE I/O with the pinned resample direction (needs nibabel + lib.resample,
    on the GPU box): load_and_remap_gt, load_prediction_on_gt_grid,
    load_fov_mask_on_gt_grid. The PIN (identical to eval_corrector): predictions
    and FOV masks are resampled onto the GT's native grid; the GT is never moved.

The FOV validity mask (revised-plan §6.3, P0-2) is the key correctness fix: instead
of slicing the GT array with a source-grid ``visible_box`` (which can silently
mis-index if the array axis order differs), the acquired-FOV mask is stored on the
truncated-CT grid and resampled (nearest) to the GT grid via affines — exactly like
the prediction — so both land on the GT grid consistently.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


# ── pure metric math (numpy only) ─────────────────────────────────────────────
def _dice(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    p = float(pred_bin.sum())
    g = float(gt_bin.sum())
    denom = p + g
    if denom == 0.0:                       # both empty -> perfect by convention
        return 1.0
    return 2.0 * float(np.logical_and(pred_bin, gt_bin).sum()) / denom


def compute_region_confusion(pred_k: np.ndarray, gt_k: np.ndarray,
                             mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Per-structure confusion INSIDE ``mask`` (whole volume if None). Returns Dice
    plus the TP/FP/FN voxel counts, recall, precision, and raw voxel totals — the
    inputs for completion vs hallucination analysis (revised-plan §8.2/8.3).

    FP inside the MISSING mask = anatomy hallucinated where the FOV did not acquire
    it; FN there = anatomy the corrector failed to complete."""
    pk = np.asarray(pred_k, dtype=bool)
    gk = np.asarray(gt_k, dtype=bool)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        pk = pk & m
        gk = gk & m
    tp = int(np.logical_and(pk, gk).sum())
    fp = int(np.logical_and(pk, ~gk).sum())
    fn = int(np.logical_and(~pk, gk).sum())
    pred_vox = tp + fp
    gt_vox = tp + fn
    denom = pred_vox + gt_vox
    return {
        "dice": (2.0 * tp / denom) if denom else 1.0,
        "tp_voxels": tp, "fp_voxels": fp, "fn_voxels": fn,
        "pred_voxels": pred_vox, "gt_voxels": gt_vox,
        "recall": (tp / gt_vox) if gt_vox else float("nan"),
        "precision": (tp / pred_vox) if pred_vox else float("nan"),
    }


def compute_volume_metrics(pred_k: np.ndarray, gt_k: np.ndarray, voxel_mm3: float,
                           mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Physical-volume metrics (mm^3) INSIDE ``mask`` (whole if None): predicted /
    GT volume, signed bias (pred-gt), absolute error, and centroid distance in
    voxels (NaN if either side empty). (revised-plan §8.1/8.2)."""
    pk = np.asarray(pred_k, dtype=bool)
    gk = np.asarray(gt_k, dtype=bool)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        pk = pk & m
        gk = gk & m
    vp = float(pk.sum()) * voxel_mm3
    vg = float(gk.sum()) * voxel_mm3
    out = {"vol_pred_mm3": vp, "vol_gt_mm3": vg,
           "vol_signed_bias_mm3": vp - vg, "vol_abs_error_mm3": abs(vp - vg),
           "vol_recovery_ratio": (vp / vg) if vg > 0 else float("nan")}
    if pk.any() and gk.any():
        cp = np.argwhere(pk).mean(0)
        cg = np.argwhere(gk).mean(0)
        out["centroid_error_vox"] = float(np.linalg.norm(cp - cg))
    else:
        out["centroid_error_vox"] = float("nan")
    return out


def compute_whole_metrics(pred_k: np.ndarray, gt_k: np.ndarray, spacing_zyx,
                          tau_mm: float = 1.0, surface: bool = False) -> Dict[str, float]:
    """Whole-structure Dice (+ optional ASSD/HD95/NSD via the existing
    simulation.evaluation.metrics.surface_metrics — REUSED, not re-derived).
    Surface metrics are computed on the WHOLE structure (revised-plan §8.1) to avoid
    the artificial FOV-cut face. ``surface=True`` needs the simulation package."""
    pk = np.asarray(pred_k, dtype=bool)
    gk = np.asarray(gt_k, dtype=bool)
    out: Dict[str, float] = {"dice": _dice(pk, gk)}
    if surface:
        from simulation.evaluation.metrics import surface_metrics   # noqa: WPS433 (reuse)
        sm = surface_metrics(pk, gk, np.asarray(spacing_zyx, dtype=float), float(tau_mm))
        out.update({"assd": float(sm["assd"]), "hd95": float(sm["hd95"]), "nsd": float(sm["nsd"])})
    return out


# ── file I/O with the pinned resample (GPU box; nibabel + lib.resample) ────────
def load_and_remap_gt(gt_path, struct_to_value, structures):
    """Load a native GT label and remap it to nnU-Net {1,2,3,4}. Returns
    (gt_nn, gt_img)."""
    import nibabel as nib                                     # noqa: WPS433
    from lib.labels import remap_to_nnunet                     # noqa: WPS433
    gt_img = nib.load(str(gt_path))
    gt_arr = np.asanyarray(gt_img.dataobj)
    stv = {k: int(v) for k, v in struct_to_value.items()}
    return remap_to_nnunet(gt_arr, stv, structures), gt_img


def load_prediction_on_gt_grid(pred_path, gt_img):
    """Resample a prediction onto the GT grid (order 0). PIN: never move the GT."""
    import nibabel as nib                                     # noqa: WPS433
    from lib import resample as _rs                            # noqa: WPS433
    pred_img = nib.load(str(pred_path))
    pred_rs = _rs.resample_to_grid(pred_img, gt_img.shape[:3], gt_img.affine, order=0)
    return np.asanyarray(pred_rs.dataobj).astype(np.int16)


def load_fov_mask_on_gt_grid(mask_path, gt_img) -> np.ndarray:
    """Resample the acquired-FOV validity mask (stored on the truncated-CT grid)
    onto the GT grid with NEAREST interpolation (revised-plan §6.3). Returns
    ``M_visible`` (bool): True = acquired/visible, False = missing/unacquired."""
    import nibabel as nib                                     # noqa: WPS433
    from lib import resample as _rs                            # noqa: WPS433
    m_img = nib.load(str(mask_path))
    m_rs = _rs.resample_to_grid(m_img, gt_img.shape[:3], gt_img.affine, order=0)
    return np.asanyarray(m_rs.dataobj) > 0.5


# ── self-test (pure metric math) ──────────────────────────────────────────────
def _selftest() -> int:
    S = 20
    gt = np.zeros((S, S, S), bool)
    gt[5:15, 5:15, 5:15] = True                     # a 10^3 cube = 1000 voxels
    mask = np.zeros((S, S, S), bool)
    mask[:10] = True                                # visible = lower half (z<10)

    # perfect pred -> dice 1, no FP/FN
    c = compute_region_confusion(gt, gt.copy())
    assert abs(c["dice"] - 1.0) < 1e-9 and c["fp_voxels"] == 0 and c["fn_voxels"] == 0
    assert c["tp_voxels"] == 1000 and abs(c["recall"] - 1.0) < 1e-9

    # hallucinate a block OUTSIDE gt, only in the missing region (z>=10)
    pred = gt.copy()
    pred[16:19, 5:8, 5:8] = True                    # 27 FP voxels, all missing side
    M_missing = ~mask
    cm = compute_region_confusion(pred, gt, M_missing)
    # gt cube z5:15 split by the cut at z=10 -> missing side z10:15 (5 slices) = 500
    assert cm["gt_voxels"] == 500 and cm["fp_voxels"] == 27 and cm["fn_voxels"] == 0
    assert cm["tp_voxels"] == 500
    cv = compute_region_confusion(pred, gt, mask)   # visible side z5:10 (5 slices) = 500, 0 fp
    assert cv["gt_voxels"] == 500 and cv["fp_voxels"] == 0
    print("region confusion:", {k: cm[k] for k in ("dice", "tp_voxels", "fp_voxels", "gt_voxels")})

    # volume metrics: 1 mm^3 voxels -> pred volume = gt + 27, positive bias
    vm = compute_volume_metrics(pred, gt, 1.0)
    assert abs(vm["vol_gt_mm3"] - 1000.0) < 1e-6 and abs(vm["vol_pred_mm3"] - 1027.0) < 1e-6
    assert abs(vm["vol_signed_bias_mm3"] - 27.0) < 1e-6
    assert vm["vol_recovery_ratio"] > 1.0 and abs(vm["centroid_error_vox"]) >= 0.0

    # empty gt in a region -> recovery ratio NaN, dice by convention on empties
    e = compute_region_confusion(np.zeros_like(gt), np.zeros_like(gt))
    assert e["dice"] == 1.0 and np.isnan(e["recall"])
    wm = compute_whole_metrics(pred, gt, (1.0, 1.0, 1.0), surface=False)
    assert abs(wm["dice"] - _dice(pred, gt)) < 1e-12
    print("EVAL-COMMON SELF-TEST PASSED")
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.parse_args()
    raise SystemExit(_selftest())
