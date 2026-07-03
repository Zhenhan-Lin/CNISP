#!/usr/bin/env python3
"""Shared Dice eval for nnUNet-C controls A / B / C (one code path = fair B-vs-C).

Consumes a ``test_cases_map.json`` (written by build_corrector_testset.py) which
records, per case: the prediction file, the native GT path, and the GT's
struct->value map. The SAME logic is used for every control, so the only thing
that differs across A/B/C is the prediction itself -- the resample + Dice are
byte-identical, which is what makes the B-vs-C gap trustworthy.

RESAMPLE DIRECTION (pinned): predictions are resampled onto EACH source's own
NATIVE GT grid with order 0 (nearest -> stays discrete); Dice is then computed on
that native grid. We never move the GT. nnUNet's prediction is exported on the
imagesTs geometry (the source's original/dense grid), so for B/C this resample is
effectively a no-op; for A (stock 835 native pred) it is likewise the native grid.
Pinning the direction guarantees A/B/C are scored on identical voxel grids.

Usage:
    # B/C: predictions are the nnUNetv2_predict outputs (relative names in map)
    python nnunet-c/diagnostics/eval_corrector.py \
        --map nnunet-c/test_input/PHOTON_CT_CORR_C_cnisp/test_cases_map.json \
        --pred-dir nnunet-c/predictions/PHOTON_CT_CORR_C_cnisp/fold_0 \
        --out-csv nnunet-c/predictions/eval_C.csv

    # A: pred paths in the map are absolute -> --pred-dir not needed
    python nnunet-c/diagnostics/eval_corrector.py \
        --map nnunet-c/test_input/PHOTON_CT_QAfiltered/test_cases_map.json \
        --out-csv nnunet-c/predictions/eval_A.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.config import add_repo_to_syspath  # noqa: E402

add_repo_to_syspath(__file__)

import numpy as np  # noqa: E402
import nibabel as nib  # noqa: E402

from lib import resample as _rs  # noqa: E402
from lib.labels import remap_to_nnunet, NNUNET_LABELS  # noqa: E402


def _dice(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    p = float(pred_bin.sum())
    g = float(gt_bin.sum())
    denom = p + g
    if denom == 0.0:                       # both empty -> perfect by convention
        return 1.0
    inter = float(np.logical_and(pred_bin, gt_bin).sum())
    return 2.0 * inter / denom


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", required=True, help="test_cases_map.json")
    ap.add_argument("--pred-dir", default=None,
                    help="dir holding predictions for RELATIVE pred_file entries "
                         "(nnUNetv2_predict output for B/C). Not needed when the "
                         "map's pred_file paths are absolute (control A).")
    ap.add_argument("--out-csv", default=None, help="per-case Dice CSV")
    ap.add_argument("--source-id", action="append", default=None,
                    help="score only these source_id(s) (repeatable). Use to see "
                         "ONE source's per-step Dice + the 'by step' summary for it.")
    ap.add_argument("--intersect-with", action="append", default=None,
                    help="One or more OTHER test_cases_map.json files. Restrict "
                         "scoring to the (source_id, step) present in THIS map AND "
                         "all of them. Use it for B vs C so both controls are "
                         "scored on the IDENTICAL (source, step) population -- "
                         "otherwise their independent --steps auto discovery can "
                         "yield different case sets and the aggregate mean Dice is "
                         "not comparable. Repeatable.")
    args = ap.parse_args()

    mp = json.load(open(args.map))
    structures = mp.get("structures", ["ON", "Recti", "Globe", "Fat"])
    cases = mp["cases"]
    control = mp.get("control", "?")
    pred_dir = Path(args.pred_dir) if args.pred_dir else None

    # ── optional source filter: score only these source_id(s) ──
    # Use to inspect ONE source's per-step Dice (the "by step" summary then
    # covers just that source across its step_sizes).
    if args.source_id:
        keep_sids = set(args.source_id)
        before = len(cases)
        cases = {cid: c for cid, c in cases.items()
                 if c.get("source_id") in keep_sids}
        print(f"[eval] source filter {sorted(keep_sids)}: "
              f"{before} -> {len(cases)} case(s)")
        if not cases:
            print("[eval] no cases match --source-id; nothing to score.",
                  file=sys.stderr)
            return 1

    # ── B∩C fairness gate: keep only (source_id, step) common to every map ──
    if args.intersect_with:
        def _keys(m: dict) -> set:
            return {(c.get("source_id"), c.get("step"))
                    for c in m["cases"].values()}
        common = _keys(mp)
        for other in args.intersect_with:
            common &= _keys(json.load(open(other)))
        before = len(cases)
        cases = {cid: c for cid, c in cases.items()
                 if (c.get("source_id"), c.get("step")) in common}
        print(f"[eval] intersect-with {len(args.intersect_with)} map(s): "
              f"{before} -> {len(cases)} case(s) on the common (source,step) set")
        if not cases:
            print("[eval] empty intersection -- nothing to score.", file=sys.stderr)
            return 1

    print("=" * 64)
    print(f"[eval] control={control}  cases={len(cases)}  structures={structures}")
    print(f"[eval] resample: PREDICTION -> native GT grid (order 0); Dice on GT grid")
    print("=" * 64)

    rows = []
    per_struct = defaultdict(list)
    per_struct_by_step = defaultdict(lambda: defaultdict(list))
    missing = 0
    for cid, c in sorted(cases.items()):
        pf = c["pred_file"]
        pred_path = Path(pf) if Path(pf).is_absolute() else (
            (pred_dir / pf) if pred_dir else Path(pf))
        gt_path = Path(c["gt_label_path"])
        if not pred_path.exists() or not gt_path.exists():
            print(f"  {cid}: MISSING pred/gt ({pred_path.name}); skip")
            missing += 1
            continue

        gt_img = nib.load(str(gt_path))
        gt_arr = np.asanyarray(gt_img.dataobj)
        stv = {k: int(v) for k, v in c["gt_struct_to_value"].items()}
        gt_nn = remap_to_nnunet(gt_arr, stv, structures)        # -> {1,2,3,4}

        # PIN: resample the prediction onto the GT grid (nearest); never move GT.
        pred_img = nib.load(str(pred_path))
        pred_rs = _rs.resample_to_grid(pred_img, gt_img.shape[:3],
                                       gt_img.affine, order=0)
        pred_nn = np.asanyarray(pred_rs.dataobj).astype(np.int16)

        row = {"case_id": cid, "source_id": c.get("source_id", ""),
               "step": c.get("step", "")}
        dices = []
        for name in structures:
            lab = NNUNET_LABELS[name]
            d = _dice(pred_nn == lab, gt_nn == lab)
            row[f"dice_{name}"] = round(d, 5)
            dices.append(d)
            per_struct[name].append(d)
            per_struct_by_step[c.get("step", "")][name].append(d)
        row["dice_mean"] = round(float(np.mean(dices)), 5)
        rows.append(row)
        detail = " ".join(f"{n}={d:.3f}" for n, d in zip(structures, dices))
        print(f"  {cid}: {detail}  mean={row['dice_mean']:.3f}")

    if not rows:
        print("[eval] no scored cases.", file=sys.stderr)
        return 1

    # ── summary ──────────────────────────────────────────────────────
    print("-" * 64)
    print(f"[eval] control={control}  scored={len(rows)}  missing={missing}")
    overall = []
    for name in structures:
        m = float(np.mean(per_struct[name]))
        overall.append(m)
        print(f"  {name:6s}: mean Dice = {m:.4f}  (n={len(per_struct[name])})")
    print(f"  {'MEAN':6s}: mean Dice = {float(np.mean(overall)):.4f}")
    print("  by step:")
    for step in sorted(per_struct_by_step):
        ms = [float(np.mean(per_struct_by_step[step][n])) for n in structures]
        print(f"    step={step}: " + " ".join(f"{n}={m:.3f}" for n, m in zip(structures, ms))
              + f"  mean={float(np.mean(ms)):.3f}")

    if args.out_csv:
        out = Path(args.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        fields = ["case_id", "source_id", "step"] + \
                 [f"dice_{n}" for n in structures] + ["dice_mean"]
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"[eval] per-case CSV -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
