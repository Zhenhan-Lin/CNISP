#!/usr/bin/env python3
"""Compare arm B (855) vs arm C (845) preprocessed cascade data to explain
identical val_dice curves.

By design B and C are IDENTICAL except for one thing: the per-case ``seg_prev``
(B = nnUNet stacked prior, C = CNISP prior). ch0 CT and GT are the same degraded
CT + same GT labels (fair B-vs-C). So if the two arms train identically, the
prior must not be differing between them. This script checks, per matched id:

  * ch0 CT data   (main ``{id}.b2nd`` channel 0)   -> EXPECT identical
  * GT seg        (main ``{id}_seg.b2nd``)          -> EXPECT identical
  * seg_prev      (relocated ``predicted_next_stage/<cfg>/{id}.b2nd``)
        - nonzero voxel fraction per arm             -> catch all-zero priors
        - B-vs-C exact equality + foreground Dice    -> catch identical priors
        - prior-vs-GT Dice within each arm           -> how trivial the prior is

Diagnosis:
  * seg_prev(B) == seg_prev(C) for most cases      -> priors mis-sourced (THE bug)
  * seg_prev all-zero in an arm                    -> relocate wrote empty priors
  * seg_prev differ (B-vs-C Dice < 1) but curves   -> data OK; look at aug /
    still identical                                    first-conv (model ignores prior)

Needs the box env: $nnUNet_preprocessed, $nnUNet_results, + importable blosc2/numpy.

Usage (defaults match the corrector pipeline):
  python nnunet-c/diagnostics/compare_arms_data.py
  python nnunet-c/diagnostics/compare_arms_data.py --max-cases 8   # faster sample
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path


def _find_dataset_dir(root: str, dataset_id: int) -> Path:
    hits = sorted(glob.glob(os.path.join(root, f"Dataset{dataset_id:03d}_*")))
    if not hits:
        raise FileNotFoundError(
            f"no Dataset{dataset_id:03d}_* under {root} (is $nnUNet_* right?)")
    return Path(hits[0])


def _load(path: Path):
    import blosc2  # lazy (box-only)
    import numpy as np
    return np.asarray(blosc2.open(urlpath=str(path), mode="r")[:])


def _squeeze_ch(arr):
    # nnUNet stores seg as (1, D, H, W); seg_prev is (D, H, W). Normalise to (D,H,W).
    if arr.ndim == 4 and arr.shape[0] == 1:
        return arr[0]
    return arr


def _fg_dice(a, b) -> float:
    """Foreground (label>0) Dice between two integer label maps of equal shape."""
    import numpy as np
    fa, fb = a > 0, b > 0
    denom = fa.sum() + fb.sum()
    if denom == 0:
        return 1.0  # both empty -> identical
    return float(2.0 * (fa & fb).sum() / denom)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--b-id", type=int, default=855, help="arm B main dataset id")
    ap.add_argument("--c-id", type=int, default=845, help="arm C main dataset id")
    ap.add_argument("--plan-name", default="nnUNetPlansFinetune")
    ap.add_argument("--configuration", default="3d_fullres")
    ap.add_argument("--trainer", default="nnUNetTrainer_OrbitalCascade",
                    help="training trainer class (keys the seg_prev folder).")
    ap.add_argument("--previous-stage", default="cnisp_prior")
    ap.add_argument("--max-cases", type=int, default=0,
                    help="cap cases compared (0 = all matched).")
    args = ap.parse_args()

    pp = os.environ.get("nnUNet_preprocessed")
    res = os.environ.get("nnUNet_results")
    if not pp or not res:
        print("ERROR: export $nnUNet_preprocessed and $nnUNet_results first.",
              file=sys.stderr)
        return 2

    b_main = _find_dataset_dir(pp, args.b_id)
    c_main = _find_dataset_dir(pp, args.c_id)
    tag = f"{args.plan_name}_{args.configuration}"
    b_data = b_main / tag
    c_data = c_main / tag

    def _prevdir(res_ds: Path) -> Path:
        return (Path(res) / res_ds.name
                / f"{args.trainer}__{args.plan_name}__{args.previous_stage}"
                / "predicted_next_stage" / args.configuration)

    b_prev = _prevdir(_find_dataset_dir(res, args.b_id))
    c_prev = _prevdir(_find_dataset_dir(res, args.c_id))

    print("── arm B vs arm C data comparison ─────────────────────────────")
    print(f"  B main data : {b_data}")
    print(f"  C main data : {c_data}")
    print(f"  B seg_prev  : {b_prev}")
    print(f"  C seg_prev  : {c_prev}")
    for d in (b_data, c_data, b_prev, c_prev):
        if not d.is_dir():
            print(f"  !! missing dir: {d}", file=sys.stderr)
    print("───────────────────────────────────────────────────────────────")

    b_ids = {p.name[:-4] for p in b_data.glob("*.pkl")}
    c_ids = {p.name[:-4] for p in c_data.glob("*.pkl")}
    common = sorted(b_ids & c_ids)
    only_b, only_c = sorted(b_ids - c_ids), sorted(c_ids - b_ids)
    print(f"  ids: B={len(b_ids)} C={len(c_ids)} common={len(common)} "
          f"only_B={len(only_b)} only_C={len(only_c)}")
    if only_b[:3] or only_c[:3]:
        print(f"       only_B e.g. {only_b[:3]}   only_C e.g. {only_c[:3]}")
    if not common:
        print("  no common ids -> different case cohorts; can't compare per-case.",
              file=sys.stderr)
        return 1
    if args.max_cases > 0:
        common = common[: args.max_cases]

    import numpy as np
    n = 0
    n_ct_same = n_gt_same = n_prev_same = 0
    prevBC_dices, prevB_gt, prevC_gt = [], [], []
    b_prev_nz, c_prev_nz = [], []
    print(f"\n  comparing {len(common)} case(s):")
    print(f"  {'id':<38} {'ct=':>4} {'gt=':>4} {'prevBC=':>7} "
          f"{'BC_dice':>7} {'Bnz%':>6} {'Cnz%':>6} {'B~GT':>5} {'C~GT':>5}")
    for cid in common:
        try:
            bd = _load(b_data / f"{cid}.b2nd")          # (C,D,H,W)
            cd = _load(c_data / f"{cid}.b2nd")
            bg = _squeeze_ch(_load(b_data / f"{cid}_seg.b2nd"))
            cg = _squeeze_ch(_load(c_data / f"{cid}_seg.b2nd"))
            bp = _squeeze_ch(_load(b_prev / f"{cid}.b2nd"))
            cp = _squeeze_ch(_load(c_prev / f"{cid}.b2nd"))
        except Exception as e:  # noqa: BLE001
            print(f"  {cid:<38} LOAD-ERR {type(e).__name__}: {e}")
            continue
        n += 1
        ct_same = (bd.shape == cd.shape and np.array_equal(bd[0], cd[0]))
        gt_same = (bg.shape == cg.shape and np.array_equal(bg, cg))
        prev_same = (bp.shape == cp.shape and np.array_equal(bp, cp))
        n_ct_same += ct_same
        n_gt_same += gt_same
        n_prev_same += prev_same
        bc_dice = _fg_dice(bp, cp) if bp.shape == cp.shape else float("nan")
        prevBC_dices.append(bc_dice)
        bnz = 100.0 * float((bp > 0).mean())
        cnz = 100.0 * float((cp > 0).mean())
        b_prev_nz.append(bnz); c_prev_nz.append(cnz)
        b_gt_d = _fg_dice(bp, bg) if bp.shape == bg.shape else float("nan")
        c_gt_d = _fg_dice(cp, cg) if cp.shape == cg.shape else float("nan")
        prevB_gt.append(b_gt_d); prevC_gt.append(c_gt_d)
        print(f"  {cid:<38} {str(ct_same):>4} {str(gt_same):>4} "
              f"{str(prev_same):>7} {bc_dice:7.4f} {bnz:6.2f} {cnz:6.2f} "
              f"{b_gt_d:5.2f} {c_gt_d:5.2f}")

    if n == 0:
        print("  nothing loaded.", file=sys.stderr)
        return 1

    def _mean(x):
        xs = [v for v in x if v == v]  # drop NaN
        return float(np.mean(xs)) if xs else float("nan")

    print("\n── summary ────────────────────────────────────────────────────")
    print(f"  cases compared           : {n}")
    print(f"  ch0 CT identical B==C    : {n_ct_same}/{n}  (EXPECT {n}/{n})")
    print(f"  GT seg identical B==C    : {n_gt_same}/{n}  (EXPECT {n}/{n})")
    print(f"  seg_prev identical B==C  : {n_prev_same}/{n}  (EXPECT 0/{n} !!)")
    print(f"  mean seg_prev B-vs-C fg Dice : {_mean(prevBC_dices):.4f}  "
          f"(1.000 => priors are the SAME map)")
    print(f"  mean seg_prev nonzero%   : B={_mean(b_prev_nz):.2f}  "
          f"C={_mean(c_prev_nz):.2f}  (0.00 => empty prior !!)")
    print(f"  mean prior-vs-GT Dice    : B={_mean(prevB_gt):.3f}  "
          f"C={_mean(prevC_gt):.3f}")
    print("───────────────────────────────────────────────────────────────")
    if n_prev_same == n:
        print("  VERDICT: seg_prev is IDENTICAL across B and C -> priors were "
              "mis-sourced. The two arms are training on the same input. FIX the "
              "prior build/relocate (each arm must relocate ITS OWN prior dataset).")
    elif _mean(b_prev_nz) == 0 or _mean(c_prev_nz) == 0:
        print("  VERDICT: a seg_prev is ALL ZERO -> relocate wrote empty priors "
              "for that arm; the one-hot channels carry no signal.")
    elif _mean(prevBC_dices) > 0.98:
        print("  VERDICT: priors nearly identical (Dice>0.98) though not byte-equal "
              "-> effectively the same input; check the prior sources.")
    else:
        print("  VERDICT: priors genuinely DIFFER (Dice<0.98) and are non-empty -> "
              "the DATA is fine. Identical curves then point at the model ignoring "
              "the prior (aug dropout too strong / first-conv ch1-4 not learning) "
              "or shared RNG making val patches coincide -- NOT a data-sharing bug.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
