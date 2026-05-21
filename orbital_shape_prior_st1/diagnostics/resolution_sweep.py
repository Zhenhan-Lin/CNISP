"""
Resolution sweep utilities for orbital shape prior evaluation.

Evaluates reconstruction quality across effective through-plane resolutions
by varying sparsification step_size.

IMPORTANT: No imports from engine.* — receives model and optimize_fn as
arguments to avoid circular imports.
"""

from __future__ import annotations

import csv
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from data_prep.sparsify import sparsen_volume

try:
    import nibabel as nib
except ImportError:
    nib = None


# ── Adaptive step / eff-res helpers ──────────────────────────────

DEFAULT_BUCKET_EDGES_MM: tuple = (1.0, 2.0, 3.0, 4.0, 5.0, 6.5, 8.5, 11.0, 13.0)


def _sweep_autocast_dtype() -> torch.dtype:
    """Pick autocast dtype for sweep no_grad forward passes."""
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


_SWEEP_AUTOCAST_DTYPE = _sweep_autocast_dtype()


def adaptive_steps_for_case(
    spacing_axis: float,
    target_eff_res_increment_mm: float = 1.0,
    max_num_steps_per_case: int = 5,
    max_eff_resolution_mm: float = 12.0,
) -> List[int]:
    """
    Per-case adaptive step list for the resolution sweep (Rule A).

      delta_step = max(1, round(target_eff_res_increment_mm / spacing_axis))
      n_total    = max_num_steps_per_case + (delta_step - 1)
      steps      = [1, 1+delta_step, 1+2*delta_step, ...]
      truncate where eff_res = step * spacing_axis > max_eff_resolution_mm
      (the dense baseline step=1 is always kept)

    Examples:
      spacing 0.5  -> delta=2, n_total=6  -> [1, 3, 5, 7, 9, 11]
      spacing 1.25 -> delta=1, n_total=5  -> [1, 2, 3, 4, 5]
      spacing 3.0  -> delta=1, n_total=5  -> [1, 2, 3, 4] (5*3=15>12 -> cut)
    """
    if spacing_axis <= 0:
        return [1]
    delta_step = max(
        1, int(round(target_eff_res_increment_mm / float(spacing_axis)))
    )
    n_total = max_num_steps_per_case + (delta_step - 1)
    steps: List[int] = []
    for k in range(n_total):
        s = 1 + k * delta_step
        if s > 1 and s * spacing_axis > max_eff_resolution_mm:
            break
        steps.append(s)
    return steps


def assign_eff_res_bucket(eff_res: float,
                          bucket_edges: Sequence[float]) -> int:
    """
    Return bucket index for `eff_res`. `bucket_edges` are sorted upper
    bounds in mm; the last bucket catches anything above the highest edge.
    """
    for i, ub in enumerate(bucket_edges):
        if eff_res <= ub + 1e-6:
            return i
    return len(bucket_edges)  # overflow bucket


def _bucket_label(idx: int, bucket_edges: Sequence[float]) -> str:
    """Human-readable label for a bucket index (e.g. '(2.0, 3.0]')."""
    if idx >= len(bucket_edges):
        return f"({bucket_edges[-1]:.1f}, inf]"
    lower = 0.0 if idx == 0 else float(bucket_edges[idx - 1])
    upper = float(bucket_edges[idx])
    return f"({lower:.1f}, {upper:.1f}]"


# ── Dice (self-contained) ────────────────────────────────────────

def _hard_dice(pred: np.ndarray, gt: np.ndarray, num_classes: int) -> Dict:
    per_class = []
    for c in range(1, num_classes):
        p, g = (pred == c), (gt == c)
        inter = np.sum(p & g)
        total = np.sum(p) + np.sum(g)
        per_class.append(2.0 * inter / (total + 1e-5))
    return {"mean": float(np.mean(per_class)), "per_class": per_class}


# ── Single case at one resolution ────────────────────────────────

def eval_case_at_resolution(
    net: torch.nn.Module,
    optimize_fn: Callable,
    label_dense: torch.Tensor,
    spacing_dense: torch.Tensor,
    step_size: int,
    step_axis: int,
    params: dict,
    device: torch.device,
    use_thick_slices: bool = False,
) -> Dict:
    """
    Sparsify → optimize latent → predict dense → Dice vs GT.

    Args:
        net: trained model (eval mode, on device)
        optimize_fn: signature (net, labels, coords, latent_dim=, lr=,
            lat_reg_lambda=, num_iters=, max_num_const_dsc=, device=) → latent
        label_dense: [D1, D2, D3] full-resolution GT
        spacing_dense: [3] voxel spacing
        step_size: 1 = dense baseline, N = keep every Nth slice
        step_axis: axis to sparsify
        params: config dict (latent_dim, lat_reg_lambda, etc.)
        device: torch device
    """
    t0 = time.time()
    offset_dense = spacing_dense / 2.0

    if step_size <= 1:
        label_obs = label_dense
        spacing_obs = spacing_dense
        offset_obs = offset_dense
    else:
        label_obs, spacing_obs, offset_obs = sparsen_volume(
            label_dense, spacing_dense, offset_dense,
            step_axis, step_size, 0, use_thick_slices,
        )

    # ── Build coordinates for latent optimization ──────────────────
    individual = [torch.arange(label_obs.shape[d]) for d in range(3)]
    meshed = torch.meshgrid(individual, indexing="ij")
    voxel_ids = torch.stack(meshed, dim=-1)
    coords = (voxel_ids.float() * spacing_obs + offset_obs
              ).unsqueeze(0).to(device)
    labels_batch = label_obs.unsqueeze(0).to(device)

    latent = optimize_fn(
        net,
        labels_batch,
        coords,
        latent_dim=params["latent_dim"],
        lr=params.get("latent_lr", 1e-2),
        lat_reg_lambda=params["lat_reg_lambda"],
        num_iters=params.get("latent_num_iters", 1200),
        max_num_const_dsc=params.get("max_num_const_train_dsc", -1),
        device=device,
    )

    # ── Dense prediction with adaptive bounding box ─────────────
    # 1. Initial bbox from sparse foreground + 1 voxel padding
    # 2. Predict within bbox
    # 3. Iteratively expand any face that has foreground on it
    # 4. Stop when all 6 faces are fully background

    full_shape = (net.image_size.cpu() / spacing_dense).ceil().long()
    offset_dense = spacing_dense / 2.0

    fg_vox = torch.nonzero(label_obs > 0, as_tuple=False)  # [M, 3]
    if fg_vox.shape[0] > 0:
        fg_coords_mm = fg_vox.float() * spacing_obs + offset_obs
        fg_dense_vox = ((fg_coords_mm - offset_dense) / spacing_dense).round().long()
        bbox_min = (fg_dense_vox.min(dim=0).values - 1).clamp(min=0)
        bbox_max = (fg_dense_vox.max(dim=0).values + 2).clamp(max=full_shape)
    else:
        bbox_min = torch.zeros(3, dtype=torch.long)
        bbox_max = full_shape

    use_amp = (device.type == "cuda"
               and _SWEEP_AUTOCAST_DTYPE != torch.float32)

    def _predict_voxels(vox_indices):
        """Predict labels for a set of voxel indices [N, 3] -> [N] int (CPU)."""
        coords = vox_indices.float() * spacing_dense + offset_dense
        coords_batch = coords.reshape(1, -1, 1, 1, 3).to(device)
        n = coords_batch.shape[1]
        # no_grad: 2M chunk is comfortably within 8 GB on any modern GPU
        # (the MLP is 128 wide, no activation accumulation across layers).
        chunk = 2_000_000
        preds_gpu = torch.empty(n, dtype=torch.int32, device=device)
        with torch.no_grad():
            for c0 in range(0, n, chunk):
                c1 = min(c0 + chunk, n)
                with torch.autocast(device_type=device.type,
                                    dtype=_SWEEP_AUTOCAST_DTYPE,
                                    enabled=use_amp):
                    lg = net(latent, coords_batch[:, c0:c1])
                preds_gpu[c0:c1] = (
                    lg.squeeze(0).squeeze(1).squeeze(1)
                      .argmax(dim=-1).to(torch.int32)
                )
        return preds_gpu.cpu()

    def _build_bbox_grid(bmin, bmax):
        """Build voxel index grid [B1, B2, B3, 3] within bounding box."""
        individual = [torch.arange(bmin[d], bmax[d]) for d in range(3)]
        meshed = torch.meshgrid(individual, indexing="ij")
        return torch.stack(meshed, dim=-1)

    # Initial prediction within bbox
    grid = _build_bbox_grid(bbox_min, bbox_max)
    bbox_shape = grid.shape[:3]
    flat_vox = grid.reshape(-1, 3)
    pred_flat = _predict_voxels(flat_vox)
    pred_vol = pred_flat.reshape(bbox_shape).numpy().astype(np.int32)

    # Iterative expansion: check each face, expand if foreground present
    MAX_EXPAND = 20
    for _ in range(MAX_EXPAND):
        expanded = False
        for axis in range(3):
            for side in [0, 1]:  # 0 = low face, 1 = high face
                # Extract the face slice
                sl = [slice(None)] * 3
                sl[axis] = 0 if side == 0 else pred_vol.shape[axis] - 1
                face = pred_vol[tuple(sl)]

                if np.any(face > 0):
                    # Expand this face by 1 voxel
                    if side == 0 and bbox_min[axis] > 0:
                        bbox_min[axis] -= 1
                        # Predict the new slice
                        new_sl_idx = bbox_min[axis].item()
                        ranges = [torch.arange(bbox_min[d], bbox_max[d]) for d in range(3)]
                        ranges[axis] = torch.tensor([new_sl_idx])
                        new_grid = torch.meshgrid(ranges, indexing="ij")
                        new_vox = torch.stack(new_grid, dim=-1).reshape(-1, 3)
                        new_pred = _predict_voxels(new_vox).reshape(
                            *[r.shape[0] for r in ranges]
                        ).numpy().astype(np.int32)
                        pred_vol = np.concatenate([new_pred, pred_vol], axis=axis)
                        expanded = True
                    elif side == 1 and bbox_max[axis] < full_shape[axis]:
                        bbox_max[axis] += 1
                        new_sl_idx = bbox_max[axis].item() - 1
                        ranges = [torch.arange(bbox_min[d], bbox_max[d]) for d in range(3)]
                        ranges[axis] = torch.tensor([new_sl_idx])
                        new_grid = torch.meshgrid(ranges, indexing="ij")
                        new_vox = torch.stack(new_grid, dim=-1).reshape(-1, 3)
                        new_pred = _predict_voxels(new_vox).reshape(
                            *[r.shape[0] for r in ranges]
                        ).numpy().astype(np.int32)
                        pred_vol = np.concatenate([pred_vol, new_pred], axis=axis)
                        expanded = True
        if not expanded:
            break

    # Place into full volume
    pred_np = np.zeros(full_shape.tolist(), dtype=np.int32)
    pred_np[
        bbox_min[0]:bbox_max[0],
        bbox_min[1]:bbox_max[1],
        bbox_min[2]:bbox_max[2],
    ] = pred_vol

    bbox_points = int(np.prod(pred_vol.shape))
    full_points = int(np.prod(full_shape.tolist()))

    # ── Dice (only if GT available, i.e. evaluation mode) ─────
    gt_np = label_dense.numpy() if label_dense is not None else None
    dice_dense = {"mean": 0.0, "per_class": []}
    dice_observed = {"mean": 0.0, "per_class": []}

    if gt_np is not None:
        common = tuple(min(pred_np.shape[d], gt_np.shape[d]) for d in range(3))
        pred_eval = pred_np[:common[0], :common[1], :common[2]]
        gt_eval = gt_np[:common[0], :common[1], :common[2]]

        dice_dense = _hard_dice(pred_eval, gt_eval, net.num_classes)

        if step_size > 1:
            obs_slices = list(range(0, common[step_axis], step_size))
            sl = [slice(None)] * 3
            sl[step_axis] = obs_slices
            dice_observed = _hard_dice(pred_eval[tuple(sl)], gt_eval[tuple(sl)],
                                       net.num_classes)
        else:
            dice_observed = dice_dense

    n_total = full_shape[step_axis].item()
    n_obs = len(range(0, n_total, max(step_size, 1)))

    return {
        "dice": dice_dense,
        "dice_observed": dice_observed,
        "pred_class_map": pred_np,
        "gt_class_map": gt_np,
        "latent": latent.cpu().squeeze(0).numpy(),
        "latent_missing": False,
        "spacing": spacing_dense.numpy(),
        "step_size": step_size,
        "effective_resolution_mm": float(spacing_dense[step_axis]) * step_size,
        "n_observed_slices": n_obs,
        "n_total_slices": n_total,
        "bbox_min": bbox_min.tolist(),
        "bbox_max": bbox_max.tolist(),
        "bbox_points": bbox_points,
        "full_points": full_points,
        "time_s": time.time() - t0,
    }


# ── Resume support: load cached predictions ──────────────────────

def _try_load_cached(output_dir, casename, step, step_axis,
                     label_dense, spacing_dense, num_classes):
    """
    Check if step_XX/pred/{casename}_pred.nii.gz exists.
    If so, load it (plus its sidecar latent if available), compute Dice
    vs dense GT, and return a result dict. Returns None if not cached.

    Latent recovery
    ---------------
    Downstream consumers (e.g. iso reconstruction in infer.py) call
    ``net.predict_dense(latent, ...)``, so a missing latent silently
    corrupts those outputs. We look for an optional sidecar
    ``step_XX/latents/{casename}.npy`` and load it when present; if it is
    absent we mark the result with ``latent_missing=True`` and ship a
    placeholder ``latent`` of shape (0,) so consumers can detect this and
    skip latent-dependent steps instead of producing garbage.
    """
    if output_dir is None or nib is None:
        return None
    pred_path = output_dir / f"step_{step:02d}" / "pred" / f"{casename}_pred.nii.gz"
    if not pred_path.exists():
        return None

    pred_nii = nib.load(str(pred_path))
    pred_np = np.asarray(pred_nii.dataobj).astype(np.int32)
    gt_np = label_dense.numpy() if isinstance(label_dense, torch.Tensor) else label_dense

    common = tuple(min(pred_np.shape[d], gt_np.shape[d]) for d in range(3))
    pred_eval = pred_np[:common[0], :common[1], :common[2]]
    gt_eval = gt_np[:common[0], :common[1], :common[2]]

    dice_dense = _hard_dice(pred_eval, gt_eval, num_classes)

    if step > 1:
        obs_slices = list(range(0, common[step_axis], step))
        sl = [slice(None)] * 3
        sl[step_axis] = obs_slices
        dice_observed = _hard_dice(pred_eval[tuple(sl)], gt_eval[tuple(sl)], num_classes)
    else:
        dice_observed = dice_dense

    sp = spacing_dense.numpy() if isinstance(spacing_dense, torch.Tensor) else spacing_dense
    n_total = pred_np.shape[step_axis]

    latent_path = output_dir / f"step_{step:02d}" / "latents" / f"{casename}.npy"
    if latent_path.exists():
        latent_np = np.load(str(latent_path)).astype(np.float32)
        latent_missing = False
    else:
        latent_np = np.zeros((0,), dtype=np.float32)
        latent_missing = True

    return {
        "dice": dice_dense,
        "dice_observed": dice_observed,
        "pred_class_map": pred_np,
        "gt_class_map": gt_np,
        "latent": latent_np,
        "latent_missing": latent_missing,
        "spacing": sp,
        "step_size": step,
        "effective_resolution_mm": float(sp[step_axis]) * step,
        "n_observed_slices": len(range(0, n_total, max(step, 1))),
        "n_total_slices": n_total,
        "time_s": 0.0,
        "casename": casename,
    }


# ── Sweep all cases × all resolutions ────────────────────────────

def run_sweep(
    net: torch.nn.Module,
    optimize_fn: Callable,
    casenames: List[str],
    labels_dense: List[torch.Tensor],
    spacings_dense: List[torch.Tensor],
    step_axis: int,
    params: dict,
    device: torch.device,
    sweep_cfg: Optional[dict] = None,
    output_dir: Optional[Path] = None,
) -> List[Dict]:
    """
    Per-case adaptive resolution sweep.

    Each case's step list is computed from its own through-plane spacing
    via ``adaptive_steps_for_case`` (Rule A). All cases share the same
    ``step_axis`` but may have different step lists, so on-disk
    ``step_XX/`` directories may contain different subsets of cases.

    sweep_cfg keys (all optional; defaults reproduce the original behaviour):
        target_eff_res_increment_mm  (default 1.0)
        max_num_steps_per_case       (default 5)
        max_eff_resolution_mm        (default 12.0)
    """
    cfg = sweep_cfg or {}
    target_inc = float(cfg.get("target_eff_res_increment_mm", 1.0))
    max_count = int(cfg.get("max_num_steps_per_case", 5))
    max_eff = float(cfg.get("max_eff_resolution_mm", 12.0))

    all_results: List[Dict] = []
    for ci, casename in enumerate(casenames):
        spacing_axis = float(spacings_dense[ci][step_axis])
        steps = adaptive_steps_for_case(
            spacing_axis,
            target_eff_res_increment_mm=target_inc,
            max_num_steps_per_case=max_count,
            max_eff_resolution_mm=max_eff,
        )
        eff_res_list = [s * spacing_axis for s in steps]

        print(f"\n{'='*60}")
        print(f"Case {ci+1}/{len(casenames)}: {casename}")
        print(f"  spacing[axis={step_axis}] = {spacing_axis:.3f} mm")
        print(f"  adaptive steps = {steps}")
        print(f"  eff_res (mm)   = [" + ", ".join(f"{e:.2f}" for e in eff_res_list) + "]")
        print(f"{'='*60}")

        for step in steps:
            eff_res = step * spacing_axis

            cached = _try_load_cached(
                output_dir, casename, step, step_axis,
                labels_dense[ci], spacings_dense[ci], net.num_classes,
            ) if output_dir else None

            if cached is not None:
                tag = "CACHED" if not cached.get("latent_missing") else "CACHED (no z)"
                print(f"  step={step:>2d} (eff_res={eff_res:.2f}mm) ... "
                      f"{tag} dense={cached['dice']['mean']:.3f}  "
                      f"obs={cached['dice_observed']['mean']:.3f}")
                all_results.append(cached)
                continue

            print(f"  step={step:>2d} (eff_res={eff_res:.2f}mm) ... ",
                  end="", flush=True)

            result = eval_case_at_resolution(
                net=net, optimize_fn=optimize_fn,
                label_dense=labels_dense[ci],
                spacing_dense=spacings_dense[ci],
                step_size=step, step_axis=step_axis,
                params=params, device=device,
                use_thick_slices=params.get("use_thick_slices", False),
            )
            result["casename"] = casename

            dice = result["dice"]
            dice_obs = result["dice_observed"]
            bbox_pct = (result.get("bbox_points", 0) /
                        max(result.get("full_points", 1), 1) * 100)
            print(f"dense={dice['mean']:.3f}  obs={dice_obs['mean']:.3f}  "
                  f"({result['n_observed_slices']}/{result['n_total_slices']} slices, "
                  f"bbox={bbox_pct:.0f}%, {result['time_s']:.1f}s)")

            all_results.append(result)
    return all_results


# ── Summary printing ──────────────────────────────────────────────

def _group_by_bucket(all_results: List[Dict],
                     bucket_edges: Sequence[float]) -> Dict[int, List[Dict]]:
    """Group results by effective-resolution bucket."""
    grouped: Dict[int, List[Dict]] = defaultdict(list)
    for r in all_results:
        bi = assign_eff_res_bucket(r["effective_resolution_mm"], bucket_edges)
        grouped[bi].append(r)
    return grouped


def print_sweep_summary(all_results: List[Dict],
                        class_names: List[str],
                        bucket_edges: Sequence[float] = DEFAULT_BUCKET_EDGES_MM,
                        ckpt_info: str = ""):
    """
    Print a cross-case summary grouped by effective-resolution bucket.

    With adaptive per-case step lists, the raw `step` is not directly
    comparable across cases (e.g. step=3 means 1.5 mm for a 0.5 mm-spacing
    case but 3.75 mm for a 1.25 mm-spacing case). We bucket by physical
    effective resolution instead.
    """
    n_cases = len(set(r["casename"] for r in all_results))
    print(f"\n\n{'='*78}")
    print(f"TEST RESULTS - Controlled Reconstruction (per-case adaptive sweep)")
    print(f"{'='*78}")
    if ckpt_info:
        print(ckpt_info)
    print(f"Test cases: {n_cases}   Bucket edges (mm): {list(bucket_edges)}")

    header = (f"{'Eff.Res bucket':>18s} {'n_obs':>6s} "
              f"{'eff_res(mm)':>14s} "
              f"{'Dense Dice':>14s} {'Obs Dice':>14s}")
    for cn in class_names:
        header += f" {cn:>8s}"
    print(f"\n{header}")
    print("-" * len(header))

    grouped = _group_by_bucket(all_results, bucket_edges)
    n_buckets = len(bucket_edges) + 1  # +1 for the overflow bucket
    for bi in range(n_buckets):
        results_bi = grouped.get(bi, [])
        if not results_bi:
            continue
        effs = [r["effective_resolution_mm"] for r in results_bi]
        dices = [r["dice"]["mean"] for r in results_bi]
        dices_obs = [r["dice_observed"]["mean"] for r in results_bi]
        per_class = np.array([r["dice"]["per_class"] for r in results_bi])
        eff_summary = f"{np.mean(effs):.2f}±{np.std(effs):.2f}"
        row = (f"{_bucket_label(bi, bucket_edges):>18s} "
               f"{len(results_bi):>6d} "
               f"{eff_summary:>14s} "
               f"{np.mean(dices):>7.3f}±{np.std(dices):.3f} "
               f"{np.mean(dices_obs):>7.3f}±{np.std(dices_obs):.3f}")
        for ci_col in range(per_class.shape[1]):
            row += f" {np.mean(per_class[:, ci_col]):>7.3f}"
        print(row)
    print(f"{'='*78}")


# ── CSV export ────────────────────────────────────────────────────

def save_sweep_csvs(all_results: List[Dict],
                    class_names: List[str],
                    output_dir,
                    bucket_edges: Sequence[float] = DEFAULT_BUCKET_EDGES_MM):
    """
    Write two CSVs:

    test_results.csv   one row per (case, step) - the raw observation
    test_summary.csv   one row per effective-resolution bucket -
                       aggregated across the cases that contributed a
                       result in that bucket
    """
    output_dir = Path(output_dir)

    csv_path = output_dir / "test_results.csv"
    fieldnames = (["casename", "step_size", "effective_resolution_mm",
                   "eff_res_bucket",
                   "dice_dense_mean", "dice_observed_mean"]
                  + [f"dice_dense_{cn}" for cn in class_names]
                  + [f"dice_obs_{cn}" for cn in class_names]
                  + ["n_observed_slices", "n_total_slices", "time_s"])
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_results:
            bi = assign_eff_res_bucket(r["effective_resolution_mm"], bucket_edges)
            row = {
                "casename": r["casename"],
                "step_size": r["step_size"],
                "effective_resolution_mm": f"{r['effective_resolution_mm']:.3f}",
                "eff_res_bucket": _bucket_label(bi, bucket_edges),
                "dice_dense_mean": f"{r['dice']['mean']:.4f}",
                "dice_observed_mean": f"{r['dice_observed']['mean']:.4f}",
                "n_observed_slices": r["n_observed_slices"],
                "n_total_slices": r["n_total_slices"],
                "time_s": f"{r['time_s']:.1f}",
            }
            for ci_col, cn in enumerate(class_names):
                row[f"dice_dense_{cn}"] = f"{r['dice']['per_class'][ci_col]:.4f}"
                row[f"dice_obs_{cn}"] = f"{r['dice_observed']['per_class'][ci_col]:.4f}"
            w.writerow(row)
    print(f"\nPer-case results: {csv_path}")

    summary_path = output_dir / "test_summary.csv"
    with open(summary_path, "w", newline="") as f:
        fields = (["eff_res_bucket",
                   "eff_res_mean_mm", "eff_res_std_mm",
                   "n_observations",
                   "dice_dense_mean", "dice_dense_std",
                   "dice_observed_mean", "dice_observed_std"]
                  + [f"{cn}_dense_mean" for cn in class_names]
                  + [f"{cn}_dense_std" for cn in class_names]
                  + [f"{cn}_obs_mean" for cn in class_names]
                  + [f"{cn}_obs_std" for cn in class_names])
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        grouped = _group_by_bucket(all_results, bucket_edges)
        n_buckets = len(bucket_edges) + 1
        for bi in range(n_buckets):
            results_bi = grouped.get(bi, [])
            if not results_bi:
                continue
            effs = [r["effective_resolution_mm"] for r in results_bi]
            d_dense = [r["dice"]["mean"] for r in results_bi]
            d_obs = [r["dice_observed"]["mean"] for r in results_bi]
            pc_dense = np.array([r["dice"]["per_class"] for r in results_bi])
            pc_obs = np.array([r["dice_observed"]["per_class"] for r in results_bi])
            row = {
                "eff_res_bucket": _bucket_label(bi, bucket_edges),
                "eff_res_mean_mm": f"{np.mean(effs):.3f}",
                "eff_res_std_mm": f"{np.std(effs):.3f}",
                "n_observations": len(results_bi),
                "dice_dense_mean": f"{np.mean(d_dense):.4f}",
                "dice_dense_std": f"{np.std(d_dense):.4f}",
                "dice_observed_mean": f"{np.mean(d_obs):.4f}",
                "dice_observed_std": f"{np.std(d_obs):.4f}",
            }
            for ci_col, cn in enumerate(class_names):
                row[f"{cn}_dense_mean"] = f"{np.mean(pc_dense[:, ci_col]):.4f}"
                row[f"{cn}_dense_std"] = f"{np.std(pc_dense[:, ci_col]):.4f}"
                row[f"{cn}_obs_mean"] = f"{np.mean(pc_obs[:, ci_col]):.4f}"
                row[f"{cn}_obs_std"] = f"{np.std(pc_obs[:, ci_col]):.4f}"
            w.writerow(row)
    print(f"Summary: {summary_path}")