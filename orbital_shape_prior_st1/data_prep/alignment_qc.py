"""
Alignment quality control.

After canonical alignment, this module answers the critical question:
    "Are the structures consistently located in the canonical patch space?"

Key outputs:
    1. Per-structure centroid statistics (mean, std in mm)
       — If ON centroid std > 10mm, alignment is insufficient
       — If Globe centroid std > 3mm, something is wrong with the pipeline
    2. Overlay visualizations (multiple cases superimposed)
    3. Structure presence/absence summary
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np
from dataclasses import dataclass

from .canonical_align import CANONICAL_LABELS


@dataclass
class StructureCentroidStats:
    """Per-structure centroid statistics across the dataset."""
    structure: str
    n_present: int          # number of cases where structure exists
    n_total: int            # total number of cases
    centroid_mean_mm: list   # [x, y, z] mean centroid in patch space
    centroid_std_mm: list    # [x, y, z] std of centroid positions
    centroid_range_mm: list  # [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
    volume_mean_mm3: float
    volume_std_mm3: float


def compute_structure_centroid(
    label_data: np.ndarray,
    voxel_sizes: np.ndarray,
    structure_label: int,
) -> Optional[np.ndarray]:
    """Compute centroid of a structure in mm (patch-local coordinates)."""
    mask = (label_data == structure_label)
    if mask.sum() == 0:
        return None
    centroid_voxel = np.array(
        [np.mean(np.where(mask)[ax]) for ax in range(3)]
    )
    return centroid_voxel * voxel_sizes


def compute_alignment_stats(
    aligned_dir: str,
    metadata_list: Optional[List[dict]] = None,
) -> Dict[str, StructureCentroidStats]:
    """
    Compute per-structure centroid statistics across all aligned patches.

    Args:
        aligned_dir: path to aligned_patches directory
        metadata_list: if provided, use these; otherwise load from metadata/ dir

    Returns:
        Dict mapping structure name → StructureCentroidStats
    """
    labels_dir = Path(aligned_dir) / "labels"
    meta_dir = Path(aligned_dir) / "metadata"

    # Load all metadata if not provided
    if metadata_list is None:
        metadata_list = []
        for json_path in sorted(meta_dir.glob("*.json")):
            with open(json_path) as f:
                metadata_list.append(json.load(f))

    # Collect centroids per structure
    centroids = {name: [] for name in CANONICAL_LABELS if name != "BG"}
    volumes = {name: [] for name in CANONICAL_LABELS if name != "BG"}

    n_total = 0
    for meta in metadata_list:
        casename = meta["casename"]
        nii_path = labels_dir / f"{casename}.nii.gz"
        if not nii_path.exists():
            continue
        n_total += 1

        img = nib.load(str(nii_path))
        data = np.asarray(img.dataobj, dtype=np.int32)
        voxel_sizes = np.sqrt(np.sum(img.affine[:3, :3] ** 2, axis=0))
        voxel_vol = float(np.prod(voxel_sizes))

        for name, label in CANONICAL_LABELS.items():
            if name == "BG":
                continue
            centroid = compute_structure_centroid(data, voxel_sizes, label)
            if centroid is not None:
                centroids[name].append(centroid)
                volumes[name].append(float(np.sum(data == label)) * voxel_vol)

    # Compute statistics
    stats = {}
    for name in centroids:
        pts = centroids[name]
        vols = volumes[name]
        if len(pts) == 0:
            continue

        pts_arr = np.array(pts)  # [N, 3]
        stats[name] = StructureCentroidStats(
            structure=name,
            n_present=len(pts),
            n_total=n_total,
            centroid_mean_mm=np.mean(pts_arr, axis=0).tolist(),
            centroid_std_mm=np.std(pts_arr, axis=0).tolist(),
            centroid_range_mm=[
                [float(pts_arr[:, ax].min()), float(pts_arr[:, ax].max())]
                for ax in range(3)
            ],
            volume_mean_mm3=float(np.mean(vols)),
            volume_std_mm3=float(np.std(vols)),
        )

    return stats


def print_alignment_report(stats: Dict[str, StructureCentroidStats]):
    """Print a human-readable summary of alignment quality."""
    print("\n" + "=" * 70)
    print("CANONICAL ALIGNMENT QUALITY REPORT")
    print("=" * 70)

    for name, s in stats.items():
        std = np.array(s.centroid_std_mm)
        max_std = float(np.max(std))
        status = "✓" if max_std < 5.0 else ("⚠" if max_std < 10.0 else "✗")
        present_pct = 100 * s.n_present / s.n_total if s.n_total else 0.0

        print(f"\n{status} {name}:")
        print(f"  Present: {s.n_present}/{s.n_total} cases "
              f"({present_pct:.0f}%)")
        print(f"  Centroid mean (mm): [{s.centroid_mean_mm[0]:.1f}, "
              f"{s.centroid_mean_mm[1]:.1f}, {s.centroid_mean_mm[2]:.1f}]")
        print(f"  Centroid std  (mm): [{s.centroid_std_mm[0]:.1f}, "
              f"{s.centroid_std_mm[1]:.1f}, {s.centroid_std_mm[2]:.1f}]")
        print(f"  Max centroid std:   {max_std:.1f} mm")
        print(f"  Volume: {s.volume_mean_mm3:.0f} ± {s.volume_std_mm3:.0f} mm³")

    print("\n" + "-" * 70)
    print("Interpretation:")
    print("  ✓ max_std < 5mm  — good alignment for implicit shape prior")
    print("  ⚠ max_std 5-10mm — marginal; consider rotation alignment")
    print("  ✗ max_std > 10mm — insufficient; need stronger preprocessing")
    print("=" * 70)


