#!/usr/bin/env python3
"""Prior-health diagnostic for the nnUNet-C corrector (Arm B vs Arm C).

Answers three questions WITHOUT retraining, on the existing data/ tree:

  D1  Are Arm C's CNISP priors (data/cnisp_pred) degrading at the thick steps
      (9/12) relative to Arm B's nnUNet priors (data/nnunet_pred)?  -> candidate C
      (the 200->320 / +step12 dataset change fed the corrector garbage priors).

  D2  Is one eye (L/R half of the volume) systematically worse -> the OS-mirror
      / misplacement asymmetry?  -> candidate B.

  D3  Inventory: #kept samples by step, per-step slice thickness, the detected
      label-scheme of the CNISP priors, and prior-file presence -> what the
      200->320 switch actually changed in the training population.

Metric: per-structure Dice between each prior and the *pseudo-GT training label*
(manifest ``gt_candidate_pred`` = the full-res Dataset835 prediction that Arm C/B
are trained to reproduce). Priors and GT are remapped to nnUNet {1,2,3,4} BY NAME
(auto scheme detection, same convention as lib/labels.py) so a scheme mismatch
cannot silently zero a channel; raw unique values are printed in the inventory.

IMPORTANT interpretation note
-----------------------------
Arm C's CNISP prior is *designed* to diverge from the 835 pseudo-GT somewhat
(that's the point of the shape prior), so a moderate C < B Dice gap is EXPECTED
and not itself a bug. The red flags this script is looking for are:
  (a) CNISP Dice COLLAPSING at step 9/12 while nnUNet holds up,
  (b) a high fraction of EMPTY / degenerate CNISP priors at the thick steps,
  (c) a large L/R (eye) asymmetry,
  (d) CNISP priors in an unexpected label scheme (labelfusion / offset) that the
      pre-HEAD train builder value-split as {1,2,3,4} -> scrambled channels.
Any of those implicates candidate C (a/b/d) or candidate B (c).

Deps: numpy, nibabel.  Usage:
    python nnunet-c/diagnostics/prior_health.py --data-root nnunet-c/data
    python nnunet-c/diagnostics/prior_health.py --data-root nnunet-c/data \
        --out /tmp/prior_health.csv --max-cases 40
"""

from __future__ import annotations

import argparse
import csv as _csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to

STRUCTS = ["ON", "Recti", "Globe", "Fat"]
NNUNET_LABELS = {"ON": 1, "Recti": 2, "Globe": 3, "Fat": 4}
LABELFUSION_LABELS = {"ON": 1, "Recti": 3, "Globe": 5, "Fat": 7}


# ── label-scheme detection + remap-by-name (mirrors lib/labels.py) ───────────
def detect_scheme_and_offset(arr: np.ndarray):
    """Infer (scheme, offset) from a native label array's value set.

    Background = most frequent value. Foreground bases = fg values minus bg.
    labelfusion -> bases subset of {1,3,5,7} (and contains 5 or 7);
    nnunet -> bases subset of {1,2,3,4}. Offset = background value.
    """
    vals, counts = np.unique(arr, return_counts=True)
    bg = int(vals[int(np.argmax(counts))])
    bases = sorted({int(v) - bg for v in vals if int(v) != bg})
    bset = set(bases)
    if bset <= {1, 3, 5, 7} and (bset & {5, 7}):
        return "labelfusion", bg
    if bset <= {1, 2, 3, 4}:
        return "nnunet", bg
    return "unknown", bg


def build_struct_to_value(scheme: str, offset: int):
    base = LABELFUSION_LABELS if scheme == "labelfusion" else NNUNET_LABELS
    return {k: v + offset for k, v in base.items()}


def remap_by_name(arr: np.ndarray):
    """Remap a mask (any scheme/offset) to nnUNet {1,2,3,4} BY NAME.

    Returns (out_uint8, scheme, offset, raw_unique). 'unknown' scheme -> assume
    nnunet (best effort) but flag it via the returned scheme string.
    """
    raw_unique = sorted(int(v) for v in np.unique(arr))
    scheme, offset = detect_scheme_and_offset(arr)
    stv = build_struct_to_value("nnunet" if scheme == "unknown" else scheme, offset)
    out = np.zeros_like(arr, dtype=np.uint8)
    for name in STRUCTS:
        out[arr == stv[name]] = NNUNET_LABELS[name]
    return out, scheme, offset, raw_unique


# ── geometry + metric helpers ────────────────────────────────────────────────
def resample_to_ref(img, ref_img, order: int):
    """Nearest (order 0) resample of a discrete mask onto ref_img's grid."""
    return resample_from_to(
        img, (ref_img.shape[:3], np.asarray(ref_img.affine)), order=order
    )


def dice(a_bin: np.ndarray, b_bin: np.ndarray) -> float:
    a = float(a_bin.sum())
    b = float(b_bin.sum())
    denom = a + b
    if denom == 0.0:
        return 1.0  # both empty -> perfect by convention
    inter = float(np.logical_and(a_bin, b_bin).sum())
    return 2.0 * inter / denom


def lr_split_axis_and_index(gt_nn: np.ndarray, affine: np.ndarray):
    """Return (array_axis_for_LR, split_index) that separates the two eyes.

    LR array axis = the array axis most aligned with world x (the first RAS axis,
    Left<->Right). Split at the median LR-index of GT foreground (the midline
    between the two globes); fall back to the array midpoint if no foreground.
    """
    lr_axis = int(np.argmax(np.abs(np.asarray(affine)[0, :3])))
    fg = np.argwhere(gt_nn > 0)
    if fg.size == 0:
        return lr_axis, gt_nn.shape[lr_axis] // 2
    split = int(np.median(fg[:, lr_axis]))
    return lr_axis, split


# ── per-sample evaluation ────────────────────────────────────────────────────
def eval_prior(prior_path: Path, gt_img, gt_nn, lr_axis, split):
    """Dice(prior, pseudo-GT) per structure + per L/R side + degeneracy stats."""
    p_img = nib.load(str(prior_path))
    p_raw = np.asanyarray(p_img.dataobj)
    p_nn, scheme, offset, raw_unique = remap_by_name(p_raw)
    # bring the prior onto the GT grid (nearest; both are discrete)
    if p_img.shape[:3] != gt_img.shape[:3] or not np.allclose(
        p_img.affine, gt_img.affine, atol=1e-4
    ):
        p_rs = resample_to_ref(
            nib.Nifti1Image(p_nn, p_img.affine), gt_img, order=0
        )
        p_nn = np.asanyarray(p_rs.dataobj).astype(np.uint8)

    per_struct = {}
    for name in STRUCTS:
        v = NNUNET_LABELS[name]
        per_struct[name] = dice(p_nn == v, gt_nn == v)
    mean_dice = float(np.mean([per_struct[n] for n in STRUCTS]))

    # L/R side dice: average only over structures with GT present on that side,
    # so an absent-on-a-side structure doesn't inflate the score to 1.0.
    idx = np.arange(gt_nn.shape[lr_axis])
    lo_mask = (idx < split)
    side_dice = {}
    for tag, keep in (("lo", lo_mask), ("hi", ~lo_mask)):
        sl = [slice(None)] * gt_nn.ndim
        sl[lr_axis] = keep
        sl = tuple(sl)
        gt_s, p_s = gt_nn[sl], p_nn[sl]
        ds = [dice(p_s == NNUNET_LABELS[n], gt_s == NNUNET_LABELS[n])
              for n in STRUCTS if (gt_s == NNUNET_LABELS[n]).any()]
        side_dice[tag] = float(np.mean(ds)) if ds else float("nan")

    fg_vox = int((p_nn > 0).sum())
    return {
        "scheme": scheme, "offset": offset, "raw_unique": raw_unique,
        "empty": fg_vox == 0, "fg_vox": fg_vox,
        "mean": mean_dice, "side_lo": side_dice["lo"], "side_hi": side_dice["hi"],
        **{f"dice_{n}": per_struct[n] for n in STRUCTS},
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-root", required=True,
                    help="the corrector data/ dir (holds cnisp_pred/, nnunet_pred/, "
                         "images/, corrector_data_manifest.json)")
    ap.add_argument("--manifest", default=None,
                    help="override manifest path (default: <data-root>/corrector_data_manifest.json)")
    ap.add_argument("--cnisp-dir", default=None, help="override data/cnisp_pred")
    ap.add_argument("--nnunet-dir", default=None, help="override data/nnunet_pred")
    ap.add_argument("--max-cases", type=int, default=0,
                    help="only evaluate the first N cases (0=all; use for a quick look)")
    ap.add_argument("--out", default=None, help="per-sample CSV output path")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    manifest_path = Path(args.manifest) if args.manifest else data_root / "corrector_data_manifest.json"
    cnisp_dir = Path(args.cnisp_dir) if args.cnisp_dir else data_root / "cnisp_pred"
    nnunet_dir = Path(args.nnunet_dir) if args.nnunet_dir else data_root / "nnunet_pred"
    if not manifest_path.is_file():
        print(f"[prior_health] manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    manifest = json.load(open(manifest_path))
    cases = manifest.get("cases", {})
    cfg_steps = manifest.get("steps")
    print("=" * 78)
    print(f"[prior_health] manifest: {manifest_path}")
    print(f"  configured steps={cfg_steps} thick_threshold_mm={manifest.get('thick_threshold_mm')} "
          f"thick_threshold_steps={manifest.get('thick_threshold_steps')} "
          f"target_samples={manifest.get('target_samples')}")
    print(f"  cnisp_pred={cnisp_dir}  nnunet_pred={nnunet_dir}")

    rows = []                         # per (case, step, control)
    # inventory accumulators
    kept_by_step = defaultdict(int)
    dropped_by_step = defaultdict(int)
    thickness_by_step = defaultdict(list)
    present = defaultdict(lambda: defaultdict(int))     # control -> step -> count
    scheme_count = defaultdict(int)                     # detected CNISP scheme -> n
    scheme_examples = []                                # a few (case_step, raw_unique)

    case_items = sorted(cases.items())
    if args.max_cases > 0:
        case_items = case_items[: args.max_cases]

    for case_id, entry in case_items:
        gt_path = (entry.get("gt_candidate_pred") or "").strip()
        gt_ok = bool(gt_path) and Path(gt_path).exists()
        if gt_ok:
            gt_img = nib.load(gt_path)
            gt_nn, _s, _o, _u = remap_by_name(np.asanyarray(gt_img.dataobj))
            lr_axis, split = lr_split_axis_and_index(gt_nn, gt_img.affine)

        for step_s, sinfo in sorted(entry.get("steps", {}).items(), key=lambda kv: int(kv[0])):
            step = int(step_s)
            if not sinfo.get("kept"):
                dropped_by_step[step] += 1
                continue
            kept_by_step[step] += 1
            if sinfo.get("thickness_mm") is not None:
                thickness_by_step[step].append(float(sinfo["thickness_mm"]))
            stem = f"{case_id}_step{step:02d}"

            for control, pdir in (("CNISP", cnisp_dir), ("nnunet", nnunet_dir)):
                pf = pdir / f"{stem}.nii.gz"
                if not pf.exists():
                    continue
                present[control][step] += 1
                if not gt_ok:
                    continue
                try:
                    r = eval_prior(pf, gt_img, gt_nn, lr_axis, split)
                except Exception as e:  # noqa: BLE001
                    print(f"  [warn] {control} {stem}: {e}", file=sys.stderr)
                    continue
                if control == "CNISP":
                    scheme_count[r["scheme"]] += 1
                    if len(scheme_examples) < 8:
                        scheme_examples.append((stem, r["scheme"], r["raw_unique"]))
                rows.append({"case_id": case_id, "step": step, "control": control,
                             "thickness_mm": sinfo.get("thickness_mm", ""), **r})

    # ── D3 inventory ─────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("D3  INVENTORY")
    print("-" * 78)
    all_steps = sorted(set(kept_by_step) | set(dropped_by_step))
    print(f"{'step':>4} | {'kept':>5} {'dropped':>7} | {'thick mm (min/med/max)':>26} | "
          f"{'CNISP files':>11} {'nnunet files':>12}")
    for s in all_steps:
        th = thickness_by_step.get(s, [])
        th_s = (f"{min(th):.1f}/{np.median(th):.1f}/{max(th):.1f}" if th else "-")
        print(f"{s:>4} | {kept_by_step[s]:>5} {dropped_by_step[s]:>7} | {th_s:>26} | "
              f"{present['CNISP'].get(s,0):>11} {present['nnunet'].get(s,0):>12}")
    print(f"\n  detected CNISP-prior label scheme: {dict(scheme_count)} "
          f"(expect all 'nnunet'; any 'labelfusion'/'unknown' => scheme/remap bug)")
    for stem, sch, uniq in scheme_examples:
        print(f"    e.g. {stem}: scheme={sch} raw_unique={uniq}")

    if not rows:
        print("\n[prior_health] no scored samples (missing GT and/or prior files). "
              "Check gt_candidate_pred paths + cnisp_pred/nnunet_pred dirs.", file=sys.stderr)
        return 1

    # ── D1 prior-vs-pseudoGT Dice by step x control ──────────────────────────
    def agg(pred):
        sub = [r for r in rows if pred(r)]
        if not sub:
            return None
        return {
            "n": len(sub),
            "mean": float(np.mean([r["mean"] for r in sub])),
            "empty_pct": 100.0 * np.mean([r["empty"] for r in sub]),
            **{n: float(np.mean([r[f"dice_{n}"] for r in sub])) for n in STRUCTS},
        }

    print("\n" + "=" * 78)
    print("D1  Dice(prior, pseudo-GT=835 full-res pred), by step x control")
    print("    (watch for CNISP COLLAPSING at step 9/12 while nnunet holds, and %empty)")
    print("-" * 78)
    print(f"{'step':>4} {'ctrl':>7} {'n':>4} | {'ON':>5} {'Recti':>6} {'Globe':>6} "
          f"{'Fat':>5} | {'MEAN':>5} | {'%empty':>6}")
    for s in sorted(set(r["step"] for r in rows)):
        for control in ("CNISP", "nnunet"):
            a = agg(lambda r, s=s, c=control: r["step"] == s and r["control"] == c)
            if a is None:
                continue
            print(f"{s:>4} {control:>7} {a['n']:>4} | {a['ON']:>5.3f} {a['Recti']:>6.3f} "
                  f"{a['Globe']:>6.3f} {a['Fat']:>5.3f} | {a['mean']:>5.3f} | {a['empty_pct']:>5.1f}%")

    # ── D2 L/R (eye) asymmetry, CNISP prior ──────────────────────────────────
    print("\n" + "=" * 78)
    print("D2  L/R half-volume asymmetry (OS-mirror check), CNISP prior")
    print("    (a large lo-vs-hi gap => one eye's prior is misplaced/mirrored)")
    print("-" * 78)
    print(f"{'step':>4} {'n':>4} | {'side_lo':>8} {'side_hi':>8} | {'|gap|':>6}")
    for s in sorted(set(r["step"] for r in rows if r["control"] == "CNISP")):
        sub = [r for r in rows if r["control"] == "CNISP" and r["step"] == s]
        lo = np.nanmean([r["side_lo"] for r in sub])
        hi = np.nanmean([r["side_hi"] for r in sub])
        print(f"{s:>4} {len(sub):>4} | {lo:>8.3f} {hi:>8.3f} | {abs(lo-hi):>6.3f}")

    # ── optional per-sample CSV ──────────────────────────────────────────────
    if args.out:
        fields = ["case_id", "step", "control", "thickness_mm", "scheme", "empty",
                  "fg_vox", "mean", "side_lo", "side_hi"] + [f"dice_{n}" for n in STRUCTS]
        with open(args.out, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\n[prior_health] per-sample CSV -> {args.out}")

    print("\n" + "=" * 78)
    print("READ-OUT GUIDE")
    print("  * D1 CNISP mean Dice fine at step 3/6 but tanks at 9/12 (or %empty high)")
    print("    while nnunet holds  -> candidate C (thick CNISP priors are garbage;")
    print("    the 200->320 / +step12 change fed them to the corrector).")
    print("  * D1 CNISP scheme != all 'nnunet'  -> a chunk of priors are mis-schemed;")
    print("    the pre-HEAD train builder value-split them wrong (scrambled channels).")
    print("  * D2 big lo-vs-hi gap  -> OS-mirror/misplacement (candidate B).")
    print("  * All roughly flat across steps & sides, no empties  -> priors are fine;")
    print("    the drift is elsewhere (representation A / train-vs-test) -> run the D4")
    print("    train-prior vs test-prior comparison next.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
