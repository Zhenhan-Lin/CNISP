"""
Resolution sweep utilities for orbital shape prior evaluation.

Evaluates reconstruction quality across effective through-plane resolutions
by varying sparsification step_size.

IMPORTANT: receives model and optimize_fn as arguments to avoid circular
imports (otherwise engine.train / engine.infer would round-trip back).
``engine.dataset`` is imported directly because it has no back-edge into
this module.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import nibabel as nib
import numpy as np
import torch

# Ensure repo root is on sys.path so `simulation/` is importable.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data_prep.sparsify import sparsen_volume
from engine.dataset import INNER_PATCH_SIZE_MM, inner_crop_64mm
from simulation.degradation import degrade_thick
from simulation.slice_profile import get_kernel
from simulation.registration import register_mask_to_gt


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
    label_obs_override: Optional[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = None,
    mode: str = "thin",
    modality: str = "ct",
    num_classes: int = 5,
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
        label_obs_override: optional (label_obs, spacing_obs, offset_obs)
            tuple to use **in place of** the internally-sparsified GT.
            Used by the Option C deployment curve, where the latent-opt
            input is a per-step canonical-aligned Dataset835 sparse-CT
            prediction rather than a sparsified copy of the GT. The
            override's patch-local mm frame is consumed for latent opt;
            the dense prediction + Dice are still computed against
            ``label_dense`` on the GT's voxel grid, so the latent z just
            transfers between the two patch frames (any centroid jitter
            between the input patch and the GT patch becomes part of the
            Dice penalty -- that's the intended deployment signal).
        mode: "thin" (point-sample) or "thick" (profile-conv).
            Ignored if label_obs_override is provided.
        modality: "ct" or "mri" (kernel selection for thick mode).
        num_classes: number of label classes (for thick mode argmax).
    """
    t0 = time.time()
    offset_dense = spacing_dense / 2.0

    if label_obs_override is not None:
        # Deployment-mode latent-opt input: skip sparsen_volume entirely
        # and consume the pre-built override patch as-is.
        label_obs, spacing_obs, offset_obs = label_obs_override
    elif step_size <= 1:
        label_obs = label_dense
        spacing_obs = spacing_dense
        offset_obs = offset_dense
    elif mode == "thick":
        kernel = get_kernel(modality, step_size)
        label_obs, spacing_obs, offset_obs = degrade_thick(
            label_dense, spacing_dense, offset_dense,
            step_axis, step_size, start=0,
            kernel=kernel, is_label=True, num_classes=num_classes,
        )
    else:
        label_obs, spacing_obs, offset_obs = sparsen_volume(
            label_dense, spacing_dense, offset_dense,
            step_axis, step_size, 0, use_thick_slices,
        )

    # ── 64 mm inner crop around the visible-LCC centroid ──────────
    # Identical to engine/dataset.py training: feed the MLP a 64 mm sub-
    # patch centred on the SPARSE foreground's largest connected
    # component, so the prior learns/uses the same drift-corrected input
    # at train and inference time.  ``sub_crop_lo_vox_dense`` and
    # ``sub_crop_shape_vox_dense`` describe where this 64 mm sub-patch
    # lives in the original 80 mm disk patch (dense voxel grid); inference
    # unmap uses them later to compose sub-patch -> disk -> full volume.
    assert label_dense is not None, (
        "_run_case requires label_dense in this pipeline — both eval and "
        "cache paths in resolution_sweep.run_sweep pass a non-None GT. If "
        "you intentionally added a no-GT inference mode, decide the target "
        "grid explicitly instead of falling back to net.image_size envelope."
    )
    disk_patch_dense_shape = list(label_dense.shape)
    inner_info = inner_crop_64mm(
        volume_sparse=label_obs,
        spacing_sparse=spacing_obs,
        offset_sparse=offset_obs,
        volume_dense=label_dense,
        spacing_dense=spacing_dense,
        offset_dense=offset_dense,
    )
    # Re-bind the working volumes/offsets to the 64 mm sub-patch frame.
    # The original disk-patch tensors are now only used via inner_info
    # bookkeeping (e.g. for native unmap and cache sidecars).
    label_obs = inner_info["sub_sparse"]
    label_dense_sub = inner_info["sub_dense"]
    offset_obs = inner_info["sub_offset_sparse_local"]
    offset_dense_sub = inner_info["sub_offset_dense_local"]
    sub_crop_lo_vox_dense = inner_info["sub_crop_lo_vox_dense"]
    sub_crop_shape_vox_dense = inner_info["sub_crop_shape_vox_dense"]
    sub_origin_mm_in_disk = inner_info["sub_origin_mm_in_disk"]
    visible_lcc_count = inner_info["visible_lcc_voxel_count"]
    visible_total_fg = inner_info["visible_total_fg_count"]

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
        soft=bool(params.get("latent_fit_soft", False)),
        label_smoothing=float(params.get("latent_fit_label_smoothing", 0.1)),
    )

    # ── Dense prediction with adaptive bounding box ─────────────
    # 1. Initial bbox from sparse foreground + 1 voxel padding
    # 2. Predict within bbox
    # 3. Iteratively expand any face that has foreground on it
    # 4. Stop when all 6 faces are fully background
    #
    # All voxel coords here are in the 64 mm SUB-PATCH dense frame.
    # ``label_dense_sub`` is the dense GT cropped to the same physical
    # region as ``label_obs``, so pred and GT share a voxel grid.
    full_shape = torch.as_tensor(label_dense_sub.shape, dtype=torch.long)
    offset_dense = offset_dense_sub

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

    # ── Dice ──────────────────────────────────────────────────
    # Pred and GT are both in the 64 mm sub-patch frame; their voxel
    # grids agree by construction (label_dense_sub uses the disk's dense
    # spacing, just cropped/padded to the sub-patch position).
    gt_np = label_dense_sub.numpy()
    assert pred_np.shape == gt_np.shape, (
        f"[_run_case] step={step_size}: pred {pred_np.shape} != gt "
        f"{gt_np.shape}. full_shape was built from label_dense_sub.shape "
        f"so this should never trigger — investigate inner_crop_64mm "
        f"shape rounding."
    )
    print(f"  [_run_case] step={step_size}  pred={pred_np.shape}  "
          f"gt={gt_np.shape}  bbox_pts={bbox_points}/{full_points}  "
          f"spacing={tuple(round(float(s), 3) for s in spacing_dense)}  "
          f"visible_fg={visible_total_fg} lcc={visible_lcc_count}")

    dice_dense = _hard_dice(pred_np, gt_np, net.num_classes)
    if step_size > 1:
        # ``step_axis`` is the disk-frame sparsify axis. After inner crop
        # the sub-patch grid is still axis-aligned with the disk grid
        # (spacing is unchanged along each axis), so ``step_axis`` indexes
        # the same physical direction in the sub-patch frame.
        obs_slices = list(range(0, pred_np.shape[step_axis], step_size))
        sl = [slice(None)] * 3
        sl[step_axis] = obs_slices
        dice_observed = _hard_dice(pred_np[tuple(sl)], gt_np[tuple(sl)],
                                   net.num_classes)
    else:
        dice_observed = dice_dense

    # n_total is reported in DISK-PATCH units so step-size accounting
    # matches what infer.py records in step metadata.
    n_total = disk_patch_dense_shape[step_axis]
    n_obs = len(range(0, n_total, max(step_size, 1)))

    return {
        "dice": dice_dense,
        "dice_observed": dice_observed,
        "pred_class_map": pred_np,
        "gt_class_map": gt_np,
        # Sub-patch bookkeeping for native unmap and cache reload.
        "sub_crop_lo_vox_dense": sub_crop_lo_vox_dense,
        "sub_crop_shape_vox_dense": sub_crop_shape_vox_dense,
        "sub_origin_mm_in_disk": sub_origin_mm_in_disk,
        "disk_patch_dense_shape": disk_patch_dense_shape,
        "visible_lcc_voxel_count": int(visible_lcc_count),
        "visible_total_fg_count": int(visible_total_fg),
        "latent": latent.cpu().squeeze(0).numpy(),
        "latent_missing": False,
        "spacing": spacing_dense.numpy(),
        "step_size": step_size,
        "step_axis": int(step_axis),
        "effective_resolution_mm": float(spacing_dense[step_axis]) * step_size,
        "n_observed_slices": n_obs,
        "n_total_slices": n_total,
        "bbox_min": bbox_min.tolist(),
        "bbox_max": bbox_max.tolist(),
        "bbox_points": bbox_points,
        "full_points": full_points,
        "time_s": time.time() - t0,
    }


# ── Real paired-data eval (Turella sim3) ─────────────────────────

def _decode_grid(
    net: torch.nn.Module,
    latent: torch.Tensor,
    spacing: torch.Tensor,
    offset: torch.Tensor,
    shape: Tuple[int, int, int],
    device: torch.device,
) -> np.ndarray:
    """Decode the latent over a full voxel grid -> [*shape] int32 class map.

    Used by the real_pair path, which decodes a dense hi-res reconstruction
    in the INPUT patch's local frame (no GT-derived bbox available because
    the GT is a separate acquisition). The 64 mm sub-patch at GT spacing is
    ~128^3, comfortably within a single chunk, so no bbox expansion is
    needed here.
    """
    use_amp = (device.type == "cuda" and _SWEEP_AUTOCAST_DTYPE != torch.float32)
    g0, g1, g2 = shape
    ii = [torch.arange(g0), torch.arange(g1), torch.arange(g2)]
    grid = torch.stack(torch.meshgrid(ii, indexing="ij"), dim=-1)
    vox = grid.reshape(-1, 3)
    coords = (vox.float() * spacing + offset).reshape(1, -1, 1, 1, 3).to(device)
    n = coords.shape[1]
    out = torch.empty(n, dtype=torch.int32, device=device)
    chunk = 2_000_000
    with torch.no_grad():
        for c0 in range(0, n, chunk):
            c1 = min(c0 + chunk, n)
            with torch.autocast(device_type=device.type,
                                dtype=_SWEEP_AUTOCAST_DTYPE, enabled=use_amp):
                lg = net(latent, coords[:, c0:c1])
            out[c0:c1] = (
                lg.squeeze(0).squeeze(1).squeeze(1).argmax(dim=-1).to(torch.int32)
            )
    return out.cpu().reshape(shape).numpy().astype(np.int32)


def eval_case_real_pair(
    net: torch.nn.Module,
    optimize_fn: Callable,
    input_obs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    gt_dense: torch.Tensor,
    gt_spacing: torch.Tensor,
    step_axis: int,
    params: dict,
    device: torch.device,
    reg_kind: str = "rigid",
) -> Dict:
    """Turella sim3: reconstruct from a REAL low-res scan, register to GT.

    Unlike the simulated curves, the low-res input and the hi-res GT are
    SEPARATE real acquisitions in different physical frames -- there is no
    voxel correspondence to crop against. We therefore:

      1. Inner-crop the input around its own visible-LCC centroid (input
         frame) and fit the latent on that 64 mm sub-patch.
      2. Decode a dense reconstruction at GT spacing in the input frame.
      3. Inner-crop the GT around its own visible-LCC centroid (GT frame).
      4. Rigidly register the reconstructed mask to the GT mask (post-hoc)
         and resample it onto the GT voxel grid (Turella's protocol).
      5. Dice on the GT grid.

    The returned dict mirrors ``eval_case_at_resolution`` so the existing
    export / native-mapping code in ``infer.py`` works unchanged. The
    sub-patch bookkeeping refers to the GT inner crop (the frame the
    registered prediction now lives in).
    """
    t0 = time.time()
    in_vol, in_sp, in_off = input_obs
    num_classes = net.num_classes

    # 1. input sub-patch (self inner-crop in the input frame)
    in_inner = inner_crop_64mm(
        volume_sparse=in_vol, spacing_sparse=in_sp, offset_sparse=in_off,
        volume_dense=in_vol, spacing_dense=in_sp, offset_dense=in_off,
    )
    label_obs = in_inner["sub_sparse"]
    off_obs = in_inner["sub_offset_sparse_local"]

    individual = [torch.arange(label_obs.shape[d]) for d in range(3)]
    voxel_ids = torch.stack(torch.meshgrid(individual, indexing="ij"), dim=-1)
    coords = (voxel_ids.float() * in_sp + off_obs).unsqueeze(0).to(device)
    labels_batch = label_obs.unsqueeze(0).to(device)

    latent = optimize_fn(
        net, labels_batch, coords,
        latent_dim=params["latent_dim"],
        lr=params.get("latent_lr", 1e-2),
        lat_reg_lambda=params["lat_reg_lambda"],
        num_iters=params.get("latent_num_iters", 1200),
        max_num_const_dsc=params.get("max_num_const_train_dsc", -1),
        device=device,
        soft=bool(params.get("latent_fit_soft", False)),
        label_smoothing=float(params.get("latent_fit_label_smoothing", 0.1)),
    )

    # 2. decode a dense reconstruction at GT spacing, in the INPUT frame.
    # Same shared lower-corner origin as the input sub-patch (offset =
    # spacing/2), so the input-fit latent decodes correctly on this grid.
    hr_sp = gt_spacing
    hr_off = hr_sp / 2.0
    hr_shape = tuple(
        int(max(round(float(INNER_PATCH_SIZE_MM) / float(hr_sp[d])), 1))
        for d in range(3)
    )
    pred_hr = torch.from_numpy(_decode_grid(net, latent, hr_sp, hr_off, hr_shape, device))

    # 3. GT sub-patch (self inner-crop in the GT frame)
    gt_off = gt_spacing / 2.0
    gt_inner = inner_crop_64mm(
        volume_sparse=gt_dense, spacing_sparse=gt_spacing, offset_sparse=gt_off,
        volume_dense=gt_dense, spacing_dense=gt_spacing, offset_dense=gt_off,
    )
    gt_sub = gt_inner["sub_dense"]
    gt_sub_off = gt_inner["sub_offset_dense_local"]

    # 4. rigid post-hoc registration -> resample pred onto GT grid.
    reg_pred, reg_info = register_mask_to_gt(
        pred_hr, hr_sp, hr_off,
        gt_sub, gt_spacing, gt_sub_off,
        kind=reg_kind,
    )

    pred_np = reg_pred.numpy().astype(np.int32)
    gt_np = gt_sub.numpy().astype(np.int32)
    assert pred_np.shape == gt_np.shape, (
        f"[real_pair] registered pred {pred_np.shape} != gt {gt_np.shape}; "
        f"register_mask_to_gt should resample onto the GT grid."
    )

    dice_dense = _hard_dice(pred_np, gt_np, num_classes)

    # Real anisotropy ratio (informational): how much coarser the input's
    # through-plane sampling is vs the GT grid.
    step_ratio = max(int(round(float(in_sp[step_axis]) / float(gt_spacing[step_axis]))), 1)
    disk_shape = list(gt_dense.shape)
    n_total = disk_shape[step_axis]

    print(f"  [real_pair] pred={pred_np.shape} gt={gt_np.shape} "
          f"in_sp={tuple(round(float(s),3) for s in in_sp)} "
          f"reg={'on' if reg_info.get('applied') else 'off'} "
          f"rms={reg_info.get('icp_rms_mm', float('nan')):.3f}mm "
          f"dice={dice_dense['mean']:.3f}")

    return {
        "dice": dice_dense,
        # No "observed slices" notion for real pairs; mirror dense.
        "dice_observed": dice_dense,
        "pred_class_map": pred_np,
        "gt_class_map": gt_np,
        # Sub-patch bookkeeping refers to the GT inner crop (the frame the
        # registered prediction now lives in) for native unmap.
        "sub_crop_lo_vox_dense": gt_inner["sub_crop_lo_vox_dense"],
        "sub_crop_shape_vox_dense": gt_inner["sub_crop_shape_vox_dense"],
        "sub_origin_mm_in_disk": gt_inner["sub_origin_mm_in_disk"],
        "disk_patch_dense_shape": disk_shape,
        "visible_lcc_voxel_count": int(gt_inner["visible_lcc_voxel_count"]),
        "visible_total_fg_count": int(gt_inner["visible_total_fg_count"]),
        "latent": latent.cpu().squeeze(0).numpy(),
        "latent_missing": False,
        "spacing": gt_spacing.numpy(),
        # On-disk layout uses step_01 (single observation per real pair);
        # the real anisotropy is carried by effective_resolution_mm and the
        # separate anisotropy_ratio field below.
        "step_size": 1,
        "anisotropy_ratio": step_ratio,
        "step_axis": int(step_axis),
        "effective_resolution_mm": float(in_sp[step_axis]),
        "n_observed_slices": len(range(0, n_total, step_ratio)),
        "n_total_slices": n_total,
        "bbox_min": [0, 0, 0],
        "bbox_max": list(pred_np.shape),
        "bbox_points": int(np.prod(pred_np.shape)),
        "full_points": int(np.prod(pred_np.shape)),
        "registration": reg_info,
        "time_s": time.time() - t0,
    }


# ── Resume support: load cached predictions ──────────────────────

def _crop_disk_to_subpatch(label_disk_np, sub_crop_lo, sub_crop_shape):
    """Zero-padded crop of a disk-patch volume to a sub-patch position.

    Mirrors ``pad_or_crop_to_voxel_bbox`` from ``engine.dataset`` but
    operates on a numpy array. Used by the cache reload path to recover
    the same 64 mm dense GT frame the live ``inner_crop_64mm`` would have
    produced; we deliberately don't re-import the torch helper here to
    keep the cache path numpy-only and side-effect free.
    """
    lo = np.asarray(sub_crop_lo, dtype=np.int64)
    sh = np.asarray(sub_crop_shape, dtype=np.int64)
    hi = lo + sh
    src_lo = np.maximum(lo, 0)
    src_hi = np.minimum(hi, np.asarray(label_disk_np.shape, dtype=np.int64))
    dst_lo = src_lo - lo
    dst_hi = dst_lo + (src_hi - src_lo)
    out = np.zeros(tuple(sh.tolist()), dtype=label_disk_np.dtype)
    if np.all(src_hi > src_lo):
        out[dst_lo[0]:dst_hi[0], dst_lo[1]:dst_hi[1], dst_lo[2]:dst_hi[2]] = \
            label_disk_np[
                src_lo[0]:src_hi[0],
                src_lo[1]:src_hi[1],
                src_lo[2]:src_hi[2],
            ]
    return out


def _try_load_cached(output_dir, casename, step, step_axis,
                     label_dense, spacing_dense, num_classes):
    """
    Check if step_XX/pred/{casename}_pred.nii.gz exists.
    If so, load it (plus its sidecar latent if available), compute Dice
    vs dense GT, and return a result dict. Returns None if not cached.

    Sub-patch sidecar
    -----------------
    Each cached pred has a sidecar ``{casename}_sub_crop.json`` next to
    it that records where the 64 mm sub-patch sits inside the 80 mm
    disk patch. We use it to crop ``label_dense`` (= the disk-patch
    dense GT) to the same sub-patch region the live path used. If the
    sidecar is missing, the cache is from a pre-inner-crop run and is
    treated as invalid (returns None so the live path re-computes).

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
    if output_dir is None:
        return None
    pred_path = output_dir / f"step_{step:02d}" / "pred" / f"{casename}_pred.nii.gz"
    if not pred_path.exists():
        return None

    # Sub-patch sidecar is mandatory under the new (LCC + inner crop)
    # pipeline. A cached pred without it is from a pre-refactor run and
    # MUST be regenerated.
    sub_crop_path = (output_dir / f"step_{step:02d}" / "pred"
                     / f"{casename}_sub_crop.json")
    if not sub_crop_path.exists():
        print(f"  [cache stale] {casename}  step={step}  no sub_crop sidecar; "
              f"deleting and re-running")
        try:
            pred_path.unlink()
        except OSError:
            pass
        return None
    with open(sub_crop_path) as f:
        sub_info = json.load(f)
    sub_crop_lo = sub_info["sub_crop_lo_vox_dense"]
    sub_crop_shape = sub_info["sub_crop_shape_vox_dense"]
    sub_origin_mm_in_disk = sub_info.get("sub_origin_mm_in_disk")
    disk_patch_dense_shape = sub_info.get(
        "disk_patch_dense_shape",
        list(label_dense.shape) if hasattr(label_dense, "shape") else None,
    )

    pred_nii = nib.load(str(pred_path))
    pred_np = np.asarray(pred_nii.dataobj).astype(np.int32)

    label_disk_np = (label_dense.numpy()
                     if isinstance(label_dense, torch.Tensor)
                     else label_dense)
    gt_np = _crop_disk_to_subpatch(label_disk_np, sub_crop_lo, sub_crop_shape)

    # Stale-cache trap: pred and the sub-patch GT must agree. If they don't
    # the sidecar JSON disagrees with the pred (e.g. sidecar manually
    # edited, or pred saved with a different shape rounding); surface it.
    assert pred_np.shape == gt_np.shape, (
        f"[_try_load_cached] {casename} step={step}: cached pred shape "
        f"{pred_np.shape} != sub-patch gt shape {gt_np.shape}. Sidecar "
        f"sub_crop_shape={sub_crop_shape}; delete "
        f"step_{step:02d}/pred/{casename}_pred.nii.gz (+ .json + latent) "
        f"and re-run."
    )
    print(f"  [cache hit] {casename}  step={step}  shape={pred_np.shape}")

    dice_dense = _hard_dice(pred_np, gt_np, num_classes)
    if step > 1:
        obs_slices = list(range(0, pred_np.shape[step_axis], step))
        sl = [slice(None)] * 3
        sl[step_axis] = obs_slices
        dice_observed = _hard_dice(pred_np[tuple(sl)], gt_np[tuple(sl)], num_classes)
    else:
        dice_observed = dice_dense

    sp = spacing_dense.numpy() if isinstance(spacing_dense, torch.Tensor) else spacing_dense
    n_total = (disk_patch_dense_shape[step_axis]
               if disk_patch_dense_shape is not None
               else pred_np.shape[step_axis])

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
        "sub_crop_lo_vox_dense": sub_crop_lo,
        "sub_crop_shape_vox_dense": sub_crop_shape,
        "sub_origin_mm_in_disk": sub_origin_mm_in_disk,
        "disk_patch_dense_shape": disk_patch_dense_shape,
        "latent": latent_np,
        "latent_missing": latent_missing,
        "spacing": sp,
        "step_size": step,
        "step_axis": int(step_axis),
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
    step_axis: Union[int, Sequence[int]],
    params: dict,
    device: torch.device,
    sweep_cfg: Optional[dict] = None,
    output_dir: Optional[Path] = None,
    label_obs_override_loader: Optional[
        Callable[[str, int],
                 Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]
    ] = None,
    real_pair: bool = False,
) -> List[Dict]:
    """
    Per-case adaptive resolution sweep.

    Each case's step list is computed from its own through-plane spacing
    via ``adaptive_steps_for_case`` (Rule A). Cases can either share a
    single ``step_axis`` (legacy ``slice_step_axis: <int>`` mode) or use
    per-case axes (``slice_step_axis: auto`` mode, where each scan is
    sparsified along its own natural through-plane direction). On-disk
    ``step_XX/`` directories may contain different subsets of cases
    either way, since step lists are adaptive.

    sweep_cfg keys (all optional; defaults reproduce the original behaviour):
        target_eff_res_increment_mm  (default 1.0)
        max_num_steps_per_case       (default 5)
        max_eff_resolution_mm        (default 12.0)

    label_obs_override_loader
        Optional ``(casename, step) -> (label_obs, spacing_obs, offset_obs)``
        callable. When provided AND returns a non-None tuple for a given
        (case, step), ``eval_case_at_resolution`` uses that tuple as the
        latent-opt input instead of sparsifying ``labels_dense[ci]``. The
        ceiling-curve path (test_label_source=atlas_gt) sets this to
        None; the deployment-curve path (test_label_source=nnunet_pred)
        loads per-step canonical-aligned Dataset835 sparse-CT patches
        through this hook. Returning ``None`` for a (case, step) causes
        the sweep to skip that row entirely -- the deployment curve uses
        this when nnUNet failed to localise a globe at high sparsity.
    """
    cfg = sweep_cfg or {}
    target_inc = float(cfg.get("target_eff_res_increment_mm", 1.0))
    max_count = int(cfg.get("max_num_steps_per_case", 5))
    max_eff = float(cfg.get("max_eff_resolution_mm", 12.0))

    # Normalize step_axis to a per-case list. Legacy callers pass a
    # single int; "auto" callers pass a sequence resolved from each
    # case's patch spacing (see data_prep.sparsify.resolve_slice_step_axes).
    if isinstance(step_axis, int):
        step_axes: List[int] = [int(step_axis)] * len(casenames)
    else:
        step_axes = [int(a) for a in step_axis]
        if len(step_axes) != len(casenames):
            raise ValueError(
                f"step_axis sequence length {len(step_axes)} does not match "
                f"number of cases {len(casenames)}."
            )

    all_results: List[Dict] = []
    for ci, casename in enumerate(casenames):
        case_axis = step_axes[ci]
        spacing_axis = float(spacings_dense[ci][case_axis])

        # ── real_pair: single observation per case, no resolution sweep ──
        if real_pair:
            print(f"\n{'='*60}")
            print(f"Case {ci+1}/{len(casenames)} [real_pair]: {casename}")
            print(f"{'='*60}")
            if label_obs_override_loader is None:
                raise ValueError(
                    "real_pair sweep requires a label_obs_override_loader to "
                    "supply the real low-res input patch."
                )
            input_obs = label_obs_override_loader(casename, 1)
            if input_obs is None:
                print(f"  real_pair ... SKIP (no input patch for {casename})")
                continue
            result = eval_case_real_pair(
                net=net, optimize_fn=optimize_fn,
                input_obs=input_obs,
                gt_dense=labels_dense[ci],
                gt_spacing=spacings_dense[ci],
                step_axis=case_axis,
                params=params, device=device,
                reg_kind=params.get("realpair_reg_kind", "rigid"),
            )
            result["casename"] = casename
            all_results.append(result)
            continue

        steps = adaptive_steps_for_case(
            spacing_axis,
            target_eff_res_increment_mm=target_inc,
            max_num_steps_per_case=max_count,
            max_eff_resolution_mm=max_eff,
        )
        eff_res_list = [s * spacing_axis for s in steps]

        print(f"\n{'='*60}")
        print(f"Case {ci+1}/{len(casenames)}: {casename}")
        print(f"  spacing[axis={case_axis}] = {spacing_axis:.3f} mm")
        print(f"  adaptive steps = {steps}")
        print(f"  eff_res (mm)   = [" + ", ".join(f"{e:.2f}" for e in eff_res_list) + "]")
        print(f"{'='*60}")

        for step in steps:
            eff_res = step * spacing_axis

            cached = _try_load_cached(
                output_dir, casename, step, case_axis,
                labels_dense[ci], spacings_dense[ci], net.num_classes,
            ) if output_dir else None

            if cached is not None:
                tag = "CACHED" if not cached.get("latent_missing") else "CACHED (no z)"
                print(f"  step={step:>2d} (eff_res={eff_res:.2f}mm) ... "
                      f"{tag} dense={cached['dice']['mean']:.3f}  "
                      f"obs={cached['dice_observed']['mean']:.3f}")
                all_results.append(cached)
                continue

            override = None
            if label_obs_override_loader is not None:
                override = label_obs_override_loader(casename, step)
                if override is None:
                    # Deployment-mode signal: skip this (case, step) row
                    # because we couldn't build a usable input patch
                    # (e.g. nnUNet missed the globe under high sparsity).
                    print(f"  step={step:>2d} (eff_res={eff_res:.2f}mm) ... "
                          f"SKIP (no input patch available)")
                    continue

            print(f"  step={step:>2d} (eff_res={eff_res:.2f}mm) ... ",
                  end="", flush=True)

            result = eval_case_at_resolution(
                net=net, optimize_fn=optimize_fn,
                label_dense=labels_dense[ci],
                spacing_dense=spacings_dense[ci],
                step_size=step, step_axis=case_axis,
                params=params, device=device,
                use_thick_slices=params.get("use_thick_slices", False),
                label_obs_override=override,
                mode=params.get("sweep_mode", "thin"),
                modality=params.get("sweep_modality", "ct"),
                num_classes=params.get("num_classes", 5),
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
    Write ``test_results.csv`` -- one row per (case, step), the raw
    observation. This is the only CSV the inference loop emits now;
    by-eff_res aggregates and all sweep figures are produced later by
    ``nnunet/build_method_summary.py`` against
    ``paired_per_source.csv`` (so CNISP and nnUNet always share the
    same source set + bucket edges).
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