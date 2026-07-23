#!/usr/bin/env python3
"""Driver: per-method cross-resolution Dice heatmaps (comparison figure).

Reads a MASK_INDEX (the same json build_metrics.py consumes -- a flat list of
per-(case, arm, step) mask entries) and renders, for every arm, the pairwise
cross-resolution Dice heatmaps + a by-method overview, alongside the other
evaluation figures. Reuses the CNISP engine/visualize heatmap core via
simulation.evaluation.cross_resolution.

Usage:
    python simulation/evaluation/cross_resolution_summary.py \
        --mask-index comparison/viz/evaluation__thick/mask_index.json \
        --out        comparison/viz/evaluation__thick

    # verify the plumbing on synthetic masks (no real data):
    python simulation/evaluation/cross_resolution_summary.py --self-test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simulation.evaluation import cross_resolution as cr


def run(args) -> int:
    with open(args.mask_index) as f:
        index = json.load(f)
    if not isinstance(index, list):
        index = index.get("index", index.get("masks", []))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = cr.render(index, out, min_steps=args.min_steps)
    if not results:
        print("[cross_resolution_summary] no arm had >= "
              f"{args.min_steps} steps; nothing rendered.", file=sys.stderr)
        return 1
    return 0


def _self_test() -> int:
    """Synthesize NIfTI masks + a MASK_INDEX and run the full render."""
    import tempfile
    import numpy as np
    import nibabel as nib
    from simulation.evaluation.metrics import SCHEMES, STRUCTURES

    td = Path(tempfile.mkdtemp())
    lut = SCHEMES["nnunet"]                 # {structure: label value}
    shape = (32, 32, 32)
    affine = np.eye(4)
    # four NON-overlapping structures (distinct corners) so all are present in every
    # mask -- real orbits always have all four; concentric balls would overwrite.
    CENTERS = {"Globe": (9, 9), "Optic nerve": (9, 23), "Recti": (23, 9), "Fat": (23, 23)}
    RADII = {"Globe": 5, "Optic nerve": 3, "Recti": 4, "Fat": 6}

    def ball(cy, cx, cz, r):
        z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
        return ((z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2) < r * r

    index = []
    # higher step -> more "erosion" of nnUNet's structures, so its cross-res Dice
    # DECREASES with the step gap; Proposed is resolution-stable (Dice stays 1).
    for arm in ("nnUNet", "Proposed"):
        for sid in ("srcA", "srcB"):
            dz = 16 if sid == "srcA" else 15
            for step in (1, 3, 6):
                vol = np.zeros(shape, np.int16)
                shrink = 0 if arm == "Proposed" else step
                for s in STRUCTURES:
                    cy, cx = CENTERS[s]
                    vol[ball(cy, cx, dz, max(2, RADII[s] - shrink))] = lut[s]
                p = td / f"{arm}_{sid}_step{step}.nii.gz"
                nib.save(nib.Nifti1Image(vol, affine), str(p))
                index.append({"case": sid, "arm": arm, "step": step, "mode": "thick",
                              "eff_res": float(step), "pred_path": str(p),
                              "gt_path": str(p), "pred_scheme": "nnunet",
                              "gt_scheme": "nnunet", "offset_pred": 0, "offset_gt": 0})

    results = cr.render(index, td, min_steps=2)
    assert set(results) == {"nnUNet", "Proposed"}, results
    for arm, r in results.items():
        mat = np.nanmean(r["mean_per_class"], axis=0)
        assert np.allclose(np.diag(mat), 1.0), (arm, np.diag(mat))   # self-Dice == 1
        # resolution-stable Proposed should be >= the eroding nnUNet off-diagonal
    m_nn = np.nanmean(results["nnUNet"]["mean_per_class"], axis=0)
    m_pr = np.nanmean(results["Proposed"]["mean_per_class"], axis=0)
    assert m_pr[0, -1] > m_nn[0, -1], (m_pr[0, -1], m_nn[0, -1])     # stable > eroding
    root = td / "cross_resolution"
    assert (root / "by_method_overview.png").stat().st_size > 0
    assert (root / "nnUNet" / "cross_res_dice_mean.png").stat().st_size > 0
    assert (root / "Proposed" / "cross_res_dice_Globe.png").stat().st_size > 0
    assert (root / "nnUNet" / "cross_res_dice_matrix.csv").stat().st_size > 0
    print(f"  nnUNet   step1-vs-step6 (mean) = {m_nn[0, -1]:.3f}  (eroding)")
    print(f"  Proposed step1-vs-step6 (mean) = {m_pr[0, -1]:.3f}  (resolution-stable)")
    print(f"  artifacts under {root}/")
    print("\nALL CROSS-RESOLUTION SELF-TESTS PASSED")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mask-index", help="MASK_INDEX json (list of per-mask entries).")
    ap.add_argument("--out", help="output dir (heatmaps go under <out>/cross_resolution/).")
    ap.add_argument("--min-steps", type=int, default=2,
                    help="minimum distinct steps an arm/source needs (default 2).")
    ap.add_argument("--self-test", action="store_true",
                    help="synthetic masks + index; verify the plumbing, no real data.")
    return ap


if __name__ == "__main__":
    a = build_parser().parse_args()
    if a.self_test:
        sys.exit(_self_test())
    if not (a.mask_index and a.out):
        build_parser().error("need --mask-index and --out (or --self-test)")
    sys.exit(run(a))
