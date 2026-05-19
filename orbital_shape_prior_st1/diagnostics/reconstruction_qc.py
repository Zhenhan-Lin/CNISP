"""
Reconstruction quality diagnostics.

Answers the key diagnostic questions:
    1. Is the reconstruction in the right PLACE? (centroid shift)
    2. Is the reconstruction the right SIZE? (volume ratio)
    3. How much error is due to position vs shape? (aligned dice gap)
    4. Which structures fail? (per-structure breakdown)

Usage:
    results = run_diagnostics(inference_results, canonical_label_names)
    print_diagnostic_report(results)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy import ndimage


# Label names matching canonical_align.py
DEFAULT_LABEL_NAMES = {0: "BG", 1: "ON", 2: "Globe", 3: "Fat", 4: "Recti"}


@dataclass
class CaseDiagnostics:
    casename: str

    # Per-structure metrics
    per_structure: Dict[str, dict] = field(default_factory=dict)
    # Each entry: {
    #   "dice_unaligned": float,
    #   "dice_aligned": float,    — after centroid alignment
    #   "position_contribution": float,  — aligned - unaligned
    #   "centroid_shift_mm": float,
    #   "volume_ratio": float,    — pred/gt
    #   "volume_pred_mm3": float,
    #   "volume_gt_mm3": float,
    # }

    # Aggregate
    mean_dice_unaligned: float = 0.0
    mean_dice_aligned: float = 0.0
    mean_position_contribution: float = 0.0


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Binary Dice coefficient."""
    intersection = np.sum(pred & gt)
    total = np.sum(pred) + np.sum(gt)
    if total == 0:
        return 1.0 if np.sum(pred) == 0 and np.sum(gt) == 0 else 0.0
    return 2.0 * intersection / total


def compute_centroid_mm(mask: np.ndarray, voxel_spacing: np.ndarray) -> Optional[np.ndarray]:
    """Centroid of a binary mask in mm."""
    if mask.sum() == 0:
        return None
    centroid_voxel = np.array(ndimage.center_of_mass(mask))
    return centroid_voxel * voxel_spacing


def shift_mask(
    mask: np.ndarray,
    shift_voxels: np.ndarray,
) -> np.ndarray:
    """
    Translate a binary mask by integer voxel shifts.
    Uses np.roll (wraps around, but for small shifts relative to
    volume size, the wrap-around region is negligible).
    """
    shifted = mask.copy()
    for ax in range(3):
        shifted = np.roll(shifted, int(shift_voxels[ax]), axis=ax)
    return shifted


def diagnose_single_case(
    pred_map: np.ndarray,
    gt_map: np.ndarray,
    spacing: np.ndarray,
    casename: str,
    label_names: Dict[int, str] = None,
) -> CaseDiagnostics:
    """
    Run all diagnostics on a single reconstruction vs GT pair.

    Args:
        pred_map: [D1,D2,D3] integer class map (predicted)
        gt_map:   [D1,D2,D3] integer class map (ground truth)
        spacing:  [3] voxel spacing in mm
        casename: identifier
        label_names: {int: str} mapping, defaults to canonical orbital labels

    Returns:
        CaseDiagnostics with per-structure and aggregate metrics
    """
    if label_names is None:
        label_names = DEFAULT_LABEL_NAMES

    voxel_vol = float(np.prod(spacing))
    diag = CaseDiagnostics(casename=casename)

    structure_dices_unaligned = []
    structure_dices_aligned = []

    for label, name in label_names.items():
        if name == "BG":
            continue

        pred_mask = (pred_map == label)
        gt_mask = (gt_map == label)

        vol_pred = float(pred_mask.sum()) * voxel_vol
        vol_gt = float(gt_mask.sum()) * voxel_vol

        # Dice (unaligned)
        dice_unaligned = compute_dice(pred_mask, gt_mask)

        # Centroid shift
        c_pred = compute_centroid_mm(pred_mask, spacing)
        c_gt = compute_centroid_mm(gt_mask, spacing)
        if c_pred is not None and c_gt is not None:
            centroid_shift = float(np.linalg.norm(c_pred - c_gt))

            # Centroid-aligned dice
            shift_voxels = np.round((c_gt - c_pred) / spacing).astype(int)
            pred_shifted = shift_mask(pred_mask, shift_voxels)
            dice_aligned = compute_dice(pred_shifted, gt_mask)
        else:
            centroid_shift = float("nan")
            dice_aligned = dice_unaligned

        # Volume ratio
        vol_ratio = vol_pred / vol_gt if vol_gt > 0 else float("inf")

        diag.per_structure[name] = {
            "dice_unaligned": dice_unaligned,
            "dice_aligned": dice_aligned,
            "position_contribution": dice_aligned - dice_unaligned,
            "centroid_shift_mm": centroid_shift,
            "volume_ratio": vol_ratio,
            "volume_pred_mm3": vol_pred,
            "volume_gt_mm3": vol_gt,
        }

        structure_dices_unaligned.append(dice_unaligned)
        structure_dices_aligned.append(dice_aligned)

    if structure_dices_unaligned:
        diag.mean_dice_unaligned = float(np.mean(structure_dices_unaligned))
        diag.mean_dice_aligned = float(np.mean(structure_dices_aligned))
        diag.mean_position_contribution = diag.mean_dice_aligned - diag.mean_dice_unaligned

    return diag


def run_diagnostics(
    inference_results: List[dict],
    label_names: Dict[int, str] = None,
) -> List[CaseDiagnostics]:
    """
    Run diagnostics on all inference results.

    Args:
        inference_results: list of dicts from infer.infer_single_case,
            each containing pred_class_map, gt_class_map, spacing, casename
    """
    all_diags = []
    for result in inference_results:
        if "gt_class_map" not in result:
            continue
        diag = diagnose_single_case(
            result["pred_class_map"],
            result["gt_class_map"],
            result["spacing"],
            result["casename"],
            label_names,
        )
        all_diags.append(diag)
    return all_diags


def print_diagnostic_report(diags: List[CaseDiagnostics]):
    """Print comprehensive diagnostic summary."""
    if not diags:
        print("No diagnostics to report.")
        return

    print("\n" + "=" * 80)
    print("RECONSTRUCTION DIAGNOSTIC REPORT")
    print("=" * 80)

    # Collect all structure names
    all_structures = set()
    for d in diags:
        all_structures.update(d.per_structure.keys())

    for struct in sorted(all_structures):
        vals = [d.per_structure[struct] for d in diags if struct in d.per_structure]
        if not vals:
            continue

        dice_u = [v["dice_unaligned"] for v in vals]
        dice_a = [v["dice_aligned"] for v in vals]
        pos_c = [v["position_contribution"] for v in vals]
        c_shift = [v["centroid_shift_mm"] for v in vals if not np.isnan(v["centroid_shift_mm"])]
        v_ratio = [v["volume_ratio"] for v in vals if np.isfinite(v["volume_ratio"])]

        print(f"\n── {struct} ({len(vals)} cases) ──")
        print(f"  Dice (unaligned):     {np.mean(dice_u):.3f} ± {np.std(dice_u):.3f}")
        print(f"  Dice (aligned):       {np.mean(dice_a):.3f} ± {np.std(dice_a):.3f}")
        print(f"  Position contribution:{np.mean(pos_c):+.3f} ± {np.std(pos_c):.3f}")
        if c_shift:
            print(f"  Centroid shift (mm):  {np.mean(c_shift):.1f} ± {np.std(c_shift):.1f}  "
                  f"[{np.min(c_shift):.1f}, {np.max(c_shift):.1f}]")
        if v_ratio:
            print(f"  Volume ratio (P/GT):  {np.mean(v_ratio):.2f} ± {np.std(v_ratio):.2f}")

    # Aggregate
    mean_u = [d.mean_dice_unaligned for d in diags]
    mean_a = [d.mean_dice_aligned for d in diags]
    mean_p = [d.mean_position_contribution for d in diags]

    print(f"\n{'─' * 80}")
    print(f"AGGREGATE (all structures)")
    print(f"  Mean Dice (unaligned): {np.mean(mean_u):.3f} ± {np.std(mean_u):.3f}")
    print(f"  Mean Dice (aligned):   {np.mean(mean_a):.3f} ± {np.std(mean_a):.3f}")
    print(f"  Position contribution: {np.mean(mean_p):+.3f} ± {np.std(mean_p):.3f}")

    print(f"\n{'─' * 80}")
    print("INTERPRETATION:")
    avg_pos = np.mean(mean_p)
    if avg_pos > 0.10:
        print("  ✗ Position is the dominant error source (contribution > 0.10).")
        print("    → Improve canonical alignment or add Jansen-style pose optimization.")
    elif avg_pos > 0.05:
        print("  ⚠ Position contributes moderately (0.05-0.10).")
        print("    → Consider pose optimization as a next step.")
    else:
        print("  ✓ Position is well-controlled (contribution < 0.05).")
        print("    → Focus on shape/capacity improvements if dice is still low.")

    avg_dice = np.mean(mean_u)
    if avg_dice > 0.85:
        print(f"  ✓ Reconstruction quality is good (mean dice {avg_dice:.3f}).")
    elif avg_dice > 0.70:
        print(f"  ⚠ Reconstruction quality is moderate (mean dice {avg_dice:.3f}).")
    else:
        print(f"  ✗ Reconstruction quality is poor (mean dice {avg_dice:.3f}).")
    print("=" * 80)
