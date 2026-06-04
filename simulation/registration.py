"""
Post-hoc rigid mask-to-mask registration (Turella-style paired-data eval).

For the real paired-data line (Turella et al., "High-Resolution Segmentation
of Lumbar Vertebrae from Conventional Thick-Slice MRI"), the low-resolution
input scan and the high-resolution GT scan are SEPARATE acquisitions in
different physical frames. Following Turella's protocol, we reconstruct the
shape from the low-res input independently and then RIGIDLY register the
reconstructed mask to the GT mask before computing Dice/HD, to absorb the
subject's inter-acquisition repositioning.

This module is intentionally dependency-free (NumPy only; SciPy's cKDTree is
used opportunistically if available, with a NumPy brute-force fallback). No
image registration is performed — only mask/point-set rigid alignment in
physical (mm) space, which is cheap and reproducible.

Convention: a voxel at integer index ``i`` (per axis) sits at physical
position ``coord = i * spacing + offset`` (the same half-pixel convention used
across the CNISP pipeline).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch


def _foreground_coords_mm(
    label: torch.Tensor,
    spacing: torch.Tensor,
    offset: torch.Tensor,
) -> np.ndarray:
    """Return [N, 3] physical (mm) coordinates of foreground voxels."""
    lab = label.detach().cpu().numpy()
    sp = spacing.detach().cpu().numpy().astype(np.float64)
    off = offset.detach().cpu().numpy().astype(np.float64)
    idx = np.argwhere(lab > 0).astype(np.float64)  # [N, 3] voxel indices
    if idx.shape[0] == 0:
        return idx
    return idx * sp[None, :] + off[None, :]


def _subsample(points: np.ndarray, max_points: int, seed: int = 0) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points
    rng = np.random.default_rng(seed)
    sel = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[sel]


def _nearest_neighbors(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """For each src point return the index of its nearest dst point.

    Uses SciPy cKDTree when available, else NumPy brute force.
    """
    try:
        from scipy.spatial import cKDTree  # type: ignore

        tree = cKDTree(dst)
        _d, nn = tree.query(src, k=1)
        return np.asarray(nn, dtype=np.int64)
    except Exception:
        # Brute force: chunked to bound memory.
        nn = np.empty(src.shape[0], dtype=np.int64)
        chunk = 4096
        for s in range(0, src.shape[0], chunk):
            seg = src[s:s + chunk]
            d2 = ((seg[:, None, :] - dst[None, :, :]) ** 2).sum(axis=2)
            nn[s:s + chunk] = d2.argmin(axis=1)
        return nn


def _kabsch(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Solve for rigid (R, t) mapping src -> dst given correspondences.

    Returns (R [3,3], t [3]) minimizing ||R @ src + t - dst||.
    Reflection is prevented (proper rotation, det(R)=+1).
    """
    src_c = src.mean(axis=0)
    dst_c = dst.mean(axis=0)
    H = (src - src_c).T @ (dst - dst_c)
    U, _S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = dst_c - R @ src_c
    return R, t


def _pca_axes(points: np.ndarray) -> np.ndarray:
    """Principal axes (columns) of a point cloud, for coarse orientation init."""
    c = points.mean(axis=0)
    cov = np.cov((points - c).T)
    _w, v = np.linalg.eigh(cov)
    # eigh returns ascending eigenvalues; order columns descending
    return v[:, ::-1]


def rigid_icp(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    *,
    max_iter: int = 50,
    tol: float = 1e-4,
    pca_init: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Iterative Closest Point rigid alignment of src_points -> dst_points.

    Returns (R [3,3], t [3], rms) where rms is the final RMS residual (mm).

    ``pca_init`` defaults to False: the inputs here are already canonical-
    aligned to RAS (with the OS->OD flip applied), so the residual rotation
    between two acquisitions is small and a centroid init + ICP converges
    reliably. PCA-axis init is risky on the near-symmetric globe (eigvector
    sign ambiguity can seed a 180 degree flip ICP cannot escape); enable it
    only for grossly mis-oriented inputs.
    """
    if src_points.shape[0] == 0 or dst_points.shape[0] == 0:
        return np.eye(3), np.zeros(3), float("inf")

    R = np.eye(3)
    t = dst_points.mean(axis=0) - src_points.mean(axis=0)  # centroid init

    if pca_init:
        # Coarse rotation aligning principal axes (helps when orientations
        # differ between acquisitions). Refined by ICP below.
        a_src = _pca_axes(src_points)
        a_dst = _pca_axes(dst_points)
        R0 = a_dst @ a_src.T
        if np.linalg.det(R0) < 0:  # avoid reflection
            a_dst[:, -1] *= -1
            R0 = a_dst @ a_src.T
        R = R0
        t = dst_points.mean(axis=0) - R @ src_points.mean(axis=0)

    prev_rms = float("inf")
    rms = float("inf")
    for _ in range(max_iter):
        transformed = (R @ src_points.T).T + t
        nn = _nearest_neighbors(transformed, dst_points)
        matched = dst_points[nn]
        R, t = _kabsch(src_points, matched)
        residual = (R @ src_points.T).T + t - matched
        rms = float(np.sqrt((residual ** 2).sum(axis=1).mean()))
        if abs(prev_rms - rms) < tol:
            break
        prev_rms = rms
    return R, t, rms


def _resample_label_to_grid(
    pred_label: torch.Tensor,
    pred_spacing: torch.Tensor,
    pred_offset: torch.Tensor,
    R: np.ndarray,
    t: np.ndarray,
    gt_shape: Tuple[int, int, int],
    gt_spacing: torch.Tensor,
    gt_offset: torch.Tensor,
) -> torch.Tensor:
    """Resample pred_label onto the GT grid using the rigid transform.

    The transform maps pred mm coords to GT mm coords: ``p_gt = R @ p_pred + t``.
    To sample pred at each GT voxel we invert: ``p_pred = R^T @ (p_gt - t)``,
    then nearest-neighbor sample from pred_label.
    """
    pred = pred_label.detach().cpu().numpy()
    psp = pred_spacing.detach().cpu().numpy().astype(np.float64)
    poff = pred_offset.detach().cpu().numpy().astype(np.float64)
    gsp = gt_spacing.detach().cpu().numpy().astype(np.float64)
    goff = gt_offset.detach().cpu().numpy().astype(np.float64)

    g0, g1, g2 = gt_shape
    gi, gj, gk = np.meshgrid(
        np.arange(g0), np.arange(g1), np.arange(g2), indexing="ij"
    )
    gt_idx = np.stack([gi.ravel(), gj.ravel(), gk.ravel()], axis=1).astype(np.float64)
    gt_mm = gt_idx * gsp[None, :] + goff[None, :]

    # Invert rigid: pred_mm = R^T (gt_mm - t)
    pred_mm = (R.T @ (gt_mm - t[None, :]).T).T
    pred_idx = np.rint((pred_mm - poff[None, :]) / psp[None, :]).astype(np.int64)

    out = np.zeros(gt_shape, dtype=pred.dtype).ravel()
    in_bounds = (
        (pred_idx[:, 0] >= 0) & (pred_idx[:, 0] < pred.shape[0]) &
        (pred_idx[:, 1] >= 0) & (pred_idx[:, 1] < pred.shape[1]) &
        (pred_idx[:, 2] >= 0) & (pred_idx[:, 2] < pred.shape[2])
    )
    pi = pred_idx[in_bounds]
    out[in_bounds] = pred[pi[:, 0], pi[:, 1], pi[:, 2]]
    return torch.from_numpy(out.reshape(gt_shape).astype(pred.dtype))


def register_mask_to_gt(
    pred_label: torch.Tensor,
    pred_spacing: torch.Tensor,
    pred_offset: torch.Tensor,
    gt_label: torch.Tensor,
    gt_spacing: torch.Tensor,
    gt_offset: torch.Tensor,
    *,
    kind: str = "rigid",
    max_points: int = 20000,
    max_iter: int = 50,
) -> Tuple[torch.Tensor, Dict]:
    """Rigidly register a predicted mask to a GT mask (Turella paired eval).

    Reconstructs the shape independently, then aligns the predicted mask to
    the GT mask to absorb inter-acquisition repositioning. The returned mask
    is resampled onto the GT voxel grid so Dice/HD can be computed directly
    against ``gt_label``.

    Parameters
    ----------
    pred_label / gt_label : [D,D,D] integer label tensors (multi-class OK)
    *_spacing / *_offset  : [3] mm geometry of each grid
    kind : "rigid" (only supported mode) or "none" (identity, for ablation)
    max_points : cap foreground point count for ICP (subsampled if exceeded)
    max_iter : ICP iteration cap

    Returns
    -------
    (registered_pred_on_gt_grid, info_dict). info_dict carries the transform
    and diagnostics for traceability.
    """
    if kind == "none":
        return pred_label, {"kind": "none", "applied": False}

    if kind != "rigid":
        raise ValueError(f"Unsupported registration kind: {kind!r}")

    src = _foreground_coords_mm(pred_label, pred_spacing, pred_offset)
    dst = _foreground_coords_mm(gt_label, gt_spacing, gt_offset)

    if src.shape[0] == 0 or dst.shape[0] == 0:
        # Nothing to register against; return pred resampled by identity so
        # downstream Dice still runs (will simply be poor).
        return pred_label, {
            "kind": "rigid", "applied": False,
            "reason": "empty foreground in pred or gt",
        }

    src_s = _subsample(src, max_points, seed=0)
    dst_s = _subsample(dst, max_points, seed=1)

    R, t, rms = rigid_icp(src_s, dst_s, max_iter=max_iter)

    registered = _resample_label_to_grid(
        pred_label, pred_spacing, pred_offset,
        R, t, tuple(int(x) for x in gt_label.shape),
        gt_spacing, gt_offset,
    )

    info = {
        "kind": "rigid",
        "applied": True,
        "rotation": R.tolist(),
        "translation_mm": t.tolist(),
        "icp_rms_mm": rms,
        "n_src_points": int(src.shape[0]),
        "n_dst_points": int(dst.shape[0]),
    }
    return registered, info
