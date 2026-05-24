"""
Multi-view training diagnostics for Strategy B (multi-offset) training.

Monitors shape prior quality WITHOUT participating in the loss.

Metrics:
    merged_dice:
        For each scan, predict dense from each offset's latent,
        majority-vote merge, Dice vs dense GT.
        → ceiling indicator: all partial views combined.

    multiview_accuracy:
        Each foreground voxel at slice z is observed by offset (z % step).
        The remaining offsets predict it blindly. This metric is the fraction
        of those blind predictions that match GT, averaged over all
        foreground voxels.
        → shape prior quality: how well does the prior reconstruct unseen slices.

Usage (called from train.py):
    from diagnostics import compute_multiview_metrics, print_multiview_report

    metrics = compute_multiview_metrics(net, dataset, latents, device, num_classes)
    print_multiview_report(metrics)
"""

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch


# ── Grouping helper ──────────────────────────────────────────────
def _hard_dice(pred_map, gt_map, num_classes):
    """Per-class and mean hard Dice between integer label maps."""
    per_class = []
    for c in range(1, num_classes):
        p, g = (pred_map == c), (gt_map == c)
        inter = np.sum(p & g)
        total = np.sum(p) + np.sum(g)
        per_class.append(2.0 * inter / (total + 1e-5))
    return {"mean": float(np.mean(per_class)), "per_class": per_class}

def _group_by_scan(dataset) -> Dict[int, List[tuple]]:
    """Group dataset item indices by scan_id. Returns {scan_id: [(item_idx, offset), ...]}."""
    groups = defaultdict(list)
    for item_idx in range(len(dataset)):
        scan_id = dataset.scan_ids[item_idx]
        offset = dataset.sparsify_offsets_used[item_idx]
        groups[scan_id].append((item_idx, offset))
    return groups


def _select_multi_scans(dataset, max_scans: Optional[int] = None) -> Dict[int, List[tuple]]:
    """Return scan groups that have >1 offset, optionally subsampled."""
    groups = _group_by_scan(dataset)
    multi = {k: v for k, v in groups.items() if len(v) > 1}
    if not multi:
        return {}
    if max_scans and len(multi) > max_scans:
        rng = np.random.RandomState(0)
        keys = rng.choice(list(multi.keys()), max_scans, replace=False).tolist()
        multi = {k: multi[k] for k in keys}
    return multi


# ── Metric: merged Dice ──────────────────────────────────────────

@torch.no_grad()
def _compute_merged_dice(net, dataset, latents, device, num_classes,
                         scan_groups: Dict[int, List[tuple]]) -> List[dict]:
    """Per-scan merged Dice: majority-vote across offsets vs dense GT."""
    net.eval()
    results = []
    for scan_id, items in scan_groups.items():
        gt = dataset.labels_dense[scan_id]
        spacing = dataset.spacings_dense[scan_id]
        target_shape = torch.tensor(gt.shape)

        preds = []
        for item_idx, _off in items:
            latent = latents[dataset.caseids[item_idx]].unsqueeze(0).to(device)
            pred = net.predict_dense(latent, target_shape.to(device),
                                     spacing.to(device))
            preds.append(pred)

        stacked = torch.stack(preds, dim=0)
        merged = torch.mode(stacked, dim=0).values

        dice = _hard_dice(merged.numpy(), gt.numpy(), num_classes)
        results.append({
            "scan_id": scan_id,
            "merged_dice_mean": dice["mean"],
            "merged_dice_per_class": dice["per_class"],
        })
    return results


# ── Metric: multi-view accuracy ──────────────────────────────────

@torch.no_grad()
def _compute_multiview_accuracy(net, dataset, latents, device, num_classes,
                                scan_groups: Dict[int, List[tuple]]) -> List[dict]:
    """
    Per-scan multi-view accuracy: for each foreground voxel observed by
    one offset, how accurately do the other offsets reconstruct it?

    Correctness note
    ----------------
    A voxel at slice index `idx` is observed by item k (with
    slice_start_id `off_k`) iff `idx % step == off_k` -- this holds for
    ANY `off_k ∈ [0, step)`, regardless of whether the offsets came from
    Strategy B's fixed grid or Strategy A's random starts. We therefore
    compare `idx % step` to each item's recorded `off_k`. If an offset
    landed outside `[0, step)` the formula silently aliases, so we
    assert.

    Broadcasting layout:
        observed_by[z]          = z % step_size           [1, 1, D3]
        offset_ids[k]           = off_k of item k         [K, 1, 1, 1]
        not_observed[k, ...]    = (observed_by != off_k)  [K, D1, D2, D3]
        correct[k, ...]         = (pred_k == gt)          [K, D1, D2, D3]
        mask                    = not_observed & fg       [K, D1, D2, D3]
        accuracy                = correct[mask].mean()
    """
    net.eval()
    # Per-case axes (length == number of scans loaded); under legacy
    # ``slice_step_axis: <int>`` every entry is the same, under ``auto``
    # they vary per scan.
    axes = dataset.slice_step_axes
    step = dataset.slice_step_size
    results = []

    for scan_id, items in scan_groups.items():
        gt = dataset.labels_dense[scan_id]
        spacing = dataset.spacings_dense[scan_id]
        target_shape = torch.tensor(gt.shape)
        axis = int(axes[scan_id])

        offsets_used = []
        preds = []
        for item_idx, off in items:
            assert 0 <= off < step, (
                f"slice_start_id={off} out of [0, {step}) for item "
                f"{item_idx}; observed_by mask would alias."
            )
            latent = latents[dataset.caseids[item_idx]].unsqueeze(0).to(device)
            pred = net.predict_dense(latent, target_shape.to(device),
                                     spacing.to(device))
            preds.append(pred)
            offsets_used.append(off)

        pred_stack = torch.stack(preds, dim=0)        # [K, D1, D2, D3]
        K = pred_stack.shape[0]

        # Which offset observed each voxel (along the sparsify axis)
        axis_size = gt.shape[axis]
        shape_bc = [1, 1, 1]
        shape_bc[axis] = axis_size
        observed_by = torch.arange(axis_size).reshape(shape_bc) % step

        offset_ids = torch.tensor(offsets_used).reshape(K, 1, 1, 1)
        not_observed = (observed_by.unsqueeze(0) != offset_ids)   # [K, D1, D2, D3]
        correct = (pred_stack == gt.unsqueeze(0))                 # [K, D1, D2, D3]
        fg = (gt > 0)
        mask = not_observed & fg.unsqueeze(0)                     # [K, D1, D2, D3]

        n_valid = mask.sum().item()
        if n_valid == 0:
            continue

        acc = correct[mask].float().mean().item()

        # Per-class accuracy
        per_class_acc = []
        for c in range(1, num_classes):
            class_mask = mask & (gt.unsqueeze(0) == c)
            if class_mask.sum() > 0:
                per_class_acc.append(correct[class_mask].float().mean().item())
            else:
                per_class_acc.append(float("nan"))

        results.append({
            "scan_id": scan_id,
            "multiview_acc": acc,
            "multiview_acc_per_class": per_class_acc,
            "n_blind_predictions": n_valid,
        })

    return results


# ── Combined entry point ─────────────────────────────────────────

def compute_multiview_metrics(
    net, dataset, latents, device, num_classes,
    max_scans: Optional[int] = None,
) -> Dict:
    """
    Compute all multi-view diagnostics.

    Args:
        net: trained MultiClassAutoDecoder
        dataset: OrbitalImplicitDataset with num_sparsify_offsets > 1
        latents: nn.Parameter [N, Z]
        device: torch device
        num_classes: total classes including BG
        max_scans: subsample for speed (None = all)

    Returns:
        dict with merged_dice and multiview_accuracy results
    """
    scan_groups = _select_multi_scans(dataset, max_scans)
    if not scan_groups:
        return {"n_scans": 0}

    merged = _compute_merged_dice(net, dataset, latents, device,
                                  num_classes, scan_groups)
    mva = _compute_multiview_accuracy(net, dataset, latents, device,
                                      num_classes, scan_groups)

    return {
        "n_scans": len(scan_groups),
        "merged_dice_results": merged,
        "multiview_acc_results": mva,
        "merged_dice_mean": float(np.mean([r["merged_dice_mean"] for r in merged])),
        "multiview_acc_mean": float(np.mean([r["multiview_acc"] for r in mva])) if mva else None,
    }


# ── Report printer ───────────────────────────────────────────────

DEFAULT_LABEL_NAMES = {0: "BG", 1: "ON", 2: "Globe", 3: "Fat", 4: "Recti"}


def print_multiview_report(metrics: Dict, label_names: Dict[int, str] = None):
    """Print multi-view diagnostic summary."""
    if metrics.get("n_scans", 0) == 0:
        print("[diag] No multi-offset scans to diagnose.")
        return

    if label_names is None:
        label_names = DEFAULT_LABEL_NAMES
    fg_names = [label_names[c] for c in sorted(label_names) if c > 0]

    n = metrics["n_scans"]
    md = metrics["merged_dice_mean"]
    ma = metrics["multiview_acc_mean"]

    print(f"\n{'─' * 60}")
    print(f"MULTI-VIEW DIAGNOSTICS ({n} scans)")
    print(f"{'─' * 60}")

    # Merged Dice
    per_scan = metrics["merged_dice_results"]
    per_class_all = np.array([r["merged_dice_per_class"] for r in per_scan])
    print(f"  Merged Dice (majority vote):  {md:.3f} ± "
          f"{np.std([r['merged_dice_mean'] for r in per_scan]):.3f}")
    for i, name in enumerate(fg_names):
        col = per_class_all[:, i]
        print(f"    {name:8s}: {np.mean(col):.3f} ± {np.std(col):.3f}")

    # Multi-view accuracy
    if ma is not None:
        mva_results = metrics["multiview_acc_results"]
        per_class_all = np.array([r["multiview_acc_per_class"] for r in mva_results])
        print(f"  Multi-view accuracy (blind):  {ma:.3f} ± "
              f"{np.std([r['multiview_acc'] for r in mva_results]):.3f}")
        for i, name in enumerate(fg_names):
            col = per_class_all[:, i]
            valid = col[~np.isnan(col)]
            if len(valid) > 0:
                print(f"    {name:8s}: {np.mean(valid):.3f} ± {np.std(valid):.3f}")

    print(f"{'─' * 60}")