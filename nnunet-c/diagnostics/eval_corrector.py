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
    ap.add_argument("--out-csv", default=None,
                    help="per-case Dice CSV. Default (when unset): "
                         "<pred-dir>/eval_<control>[__<source>].csv (a companion "
                         "..._by_step.csv is always written too).")
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
    ap.add_argument("--full-metrics", action="store_true",
                    help="also compute surface metrics (ASSD/HD95/NSD) + per-structure "
                         "volume + signed volume bias %%, REUSING "
                         "simulation.evaluation.metrics.surface_metrics (no new metric "
                         "math). Adds columns to the per-case CSV; the by-step CSV "
                         "schema is left unchanged (select_checkpoint.py reads it). "
                         "Volume CoV across steps is left to "
                         "simulation.evaluation.aggregate.stability on the emitted volumes.")
    ap.add_argument("--tau-mm", type=float, default=1.0,
                    help="Surface-Dice (NSD) tolerance in mm (default: %(default)s), "
                         "matching simulation.evaluation.metrics.DEFAULT_TAU_MM.")
    ap.add_argument("--region", choices=["all", "visible", "truncated"], default="all",
                    help="FOV experiment: restrict scoring to the VISIBLE (imaged) or "
                         "TRUNCATED (blanked) part of each case via --trunc-manifest. "
                         "'all' (default) = whole volume. The region mask is applied to "
                         "the label arrays before every per-structure metric, so surface "
                         "metrics at the FOV cut include the cut face.")
    ap.add_argument("--trunc-manifest", default=None,
                    help="fov_truncation_manifest.json (build_fov_truncated_data.py); "
                         "keyed by source_id -> pseudo-step -> {trunc_axis, visible_range, "
                         "source_shape}. Required for --region visible|truncated.")
    args = ap.parse_args()

    trunc = None
    if args.region != "all":
        if not args.trunc_manifest:
            print("[eval] --region visible|truncated needs --trunc-manifest",
                  file=sys.stderr)
            return 2
        trunc = json.load(open(args.trunc_manifest))

    # Reuse the existing metrics module (no duplicate surface-distance code). Import
    # lazily + guarded so the default Dice-only path never depends on it.
    surface_metrics = None
    if args.full_metrics:
        try:
            from simulation.evaluation.metrics import surface_metrics  # noqa: reuse
        except Exception as e:  # noqa: BLE001
            print(f"[eval] --full-metrics needs simulation.evaluation.metrics "
                  f"({type(e).__name__}: {e})", file=sys.stderr)
            return 2

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
    if args.region != "all":
        print(f"[eval] region={args.region} (FOV-restricted via {args.trunc_manifest})")
    if surface_metrics is not None:
        print(f"[eval] full metrics: ASSD/HD95/NSD(tau={args.tau_mm}mm) + volume + bias")
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

        # ── FOV region restriction (visible vs truncated) ──
        # Zero both label arrays outside the chosen region so every per-structure
        # metric is computed on that region only. The visible_range is a slice
        # window on the SOURCE grid; it maps directly iff the GT grid matches it.
        if trunc is not None:
            info = (trunc.get(str(c.get("source_id")), {}) or {}).get(str(c.get("step")))
            if not info or tuple(int(s) for s in gt_nn.shape) != \
                    tuple(int(s) for s in info.get("source_shape", ())):
                print(f"  {cid}: no trunc info / grid mismatch for --region; skip")
                missing += 1
                continue
            ax = int(info["trunc_axis"])
            vlo, vhi = int(info["visible_range"][0]), int(info["visible_range"][1])
            visible = np.zeros(gt_nn.shape, dtype=bool)
            sl = [slice(None)] * gt_nn.ndim
            sl[ax] = slice(vlo, vhi)
            visible[tuple(sl)] = True
            keep = visible if args.region == "visible" else ~visible
            gt_nn = np.where(keep, gt_nn, 0)
            pred_nn = np.where(keep, pred_nn, 0)

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
        # ── optional refinement metrics (ASSD/HD95/NSD + volume + signed bias) ──
        # Computed on the SAME already-resampled (pred_nn, gt_nn) arrays on the GT
        # grid, so they share the pinned resample; surface_metrics is the existing
        # simulation.evaluation implementation (reused, not re-derived).
        if surface_metrics is not None:
            spacing = np.asarray(gt_img.header.get_zooms()[:3], dtype=float)
            vv = float(np.prod(spacing))            # voxel volume (mm^3)
            for name in structures:
                lab = NNUNET_LABELS[name]
                pm, gm = (pred_nn == lab), (gt_nn == lab)
                sm = surface_metrics(pm, gm, spacing, args.tau_mm)
                vp, vg = float(pm.sum()) * vv, float(gm.sum()) * vv
                row[f"assd_{name}"] = round(float(sm["assd"]), 5)
                row[f"hd95_{name}"] = round(float(sm["hd95"]), 5)
                row[f"nsd_{name}"] = round(float(sm["nsd"]), 5)
                row[f"vol_pred_{name}"] = round(vp, 3)
                row[f"vol_gt_{name}"] = round(vg, 3)
                row[f"signed_pct_{name}"] = (round(100.0 * (vp - vg) / vg, 3)
                                             if vg > 0 else float("nan"))
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
    if surface_metrics is not None:
        def _nanmean(key):
            vals = np.asarray([r.get(key, np.nan) for r in rows], dtype=float)
            return float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
        print("  refinement metrics (mean, NaN-skipped):")
        for name in structures:
            print(f"    {name:6s}: ASSD={_nanmean(f'assd_{name}'):.3f}mm "
                  f"HD95={_nanmean(f'hd95_{name}'):.3f}mm "
                  f"NSD={_nanmean(f'nsd_{name}'):.3f}  "
                  f"volBias={_nanmean(f'signed_pct_{name}'):+.1f}%")

    # ── Always persist results (default path when --out-csv is unset) ──
    # Default: alongside the predictions (or the map when pred_dir is unset),
    # named eval_<control>[__<source>].csv so a manual run leaves a file behind
    # instead of only scrolling past in the terminal.
    if args.out_csv:
        out = Path(args.out_csv)
    else:
        base = pred_dir if pred_dir else Path(args.map).parent
        tag = ""
        if args.source_id:
            safe = "_".join(s.replace("/", "_") for s in args.source_id)
            tag = f"__{safe}" if len(args.source_id) == 1 else "__filtered"
        out = base / f"eval_{control}{tag}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    # per-case long CSV
    fields = ["case_id", "source_id", "step"] + \
             [f"dice_{n}" for n in structures] + ["dice_mean"]
    if args.full_metrics:
        for m in ("assd", "hd95", "nsd", "vol_pred", "vol_gt", "signed_pct"):
            fields += [f"{m}_{n}" for n in structures]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # by-step aggregate CSV (mirrors the "by step" summary printed above)
    by_step_out = out.with_name(out.stem + "_by_step.csv")
    with open(by_step_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step"] + [f"dice_{n}" for n in structures] + ["dice_mean", "n"])
        for step in sorted(per_struct_by_step):
            ms = [float(np.mean(per_struct_by_step[step][n])) for n in structures]
            n_step = len(per_struct_by_step[step][structures[0]])
            w.writerow([step] + [f"{m:.5f}" for m in ms]
                       + [f"{float(np.mean(ms)):.5f}", n_step])

    print(f"[eval] per-case CSV -> {out}")
    print(f"[eval] by-step CSV  -> {by_step_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
