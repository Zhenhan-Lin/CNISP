#!/usr/bin/env python3
"""FOV-completion region-split evaluator — emits the LONG metrics table consumed by
diagnostics/select_fov_checkpoint_driver.py, for ONE snapshot (epoch). Revised-plan
§6.3/6.4/8: FOV-validity-mask region split + strict completeness + FP/hallucination
and whole-volume metrics.

Key changes vs the first version:
  * REGION SPLIT via a stored FOV VALIDITY MASK (P0-2): the acquired-FOV mask lives
    on the truncated-CT grid and is resampled (nearest) onto the GT grid alongside
    the prediction (eval_common.load_fov_mask_on_gt_grid). We NEVER slice the GT
    array with a source-grid visible_box — that could silently mis-index.
  * STRICT COMPLETENESS (P0-3): a missing prediction / GT / FOV mask, a manifest
    mismatch, a duplicate case, an unexpected extra prediction, or a geometry
    failure RAISES. rows = expected cases × expected structures; the unique key is
    (epoch, case_id, structure).
  * FP / hallucination + volume metrics (P1-5/6): missing-region FP/FN/recall/
    precision and per-structure whole-volume + volume metrics, so the selector can
    apply a hallucination guardrail and the final test can report completion vs
    preservation vs hallucination.

Driven by ``eval_cases_map.json`` (build_fov_completion_evalset.py) which carries,
per FOV case: subject_id, crop_type, severity, is_full_fov, gt_label_path,
gt_struct_to_value, pred_file, fov_mask_file. The pure region-split core
(``evaluate_case_arrays``) is unit-tested here; the file I/O + resample path needs
nibabel + the repo's lib (masi-55).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # nnunet-c
from diagnostics.eval_common import (compute_region_confusion,           # noqa: E402
                                     compute_volume_metrics, compute_whole_metrics)

STRUCT_VALUES: Dict[str, int] = {"ON": 1, "Recti": 2, "Globe": 3, "Fat": 4}

LONG_COLUMNS = [
    "epoch", "subject_id", "case_id", "crop_type", "severity", "structure",
    # driver-consumed core
    "missing_dice", "visible_dice", "missing_gt_voxels", "visible_gt_voxels",
    # hallucination / completion (missing region)
    "missing_fp_voxels", "missing_fn_voxels", "missing_recall", "missing_precision",
    # preservation (visible region)
    "visible_fp_voxels", "visible_fn_voxels",
    # whole-structure (surface + volume; revised-plan §8.1)
    "whole_dice", "vol_pred_mm3", "vol_gt_mm3", "vol_signed_bias_mm3",
    "vol_abs_error_mm3", "centroid_error_vox",
]


def evaluate_case_arrays(
    gt_nn: np.ndarray,
    pred_nn: np.ndarray,
    M_visible: np.ndarray,
    struct_values: Dict[str, int],
    voxel_mm3: float,
    is_full_fov: bool,
    spacing_zyx=None,
    whole_surface: bool = False,
) -> List[dict]:
    """Region-split per-structure metrics for one case, all on the SAME (GT) grid.

    ``M_visible`` (bool): acquired-FOV mask already resampled to the GT grid.
    ``missing`` = ~M_visible = the truncated-away region the corrector must complete;
    a false positive there is a HALLUCINATION. Full-FOV rows carry whole-volume
    metrics in the visible columns and NaN/0 in the missing columns.
    """
    gt = np.asarray(gt_nn)
    pred = np.asarray(pred_nn)
    M_vis = np.asarray(M_visible, dtype=bool)
    if gt.shape != pred.shape or gt.shape != M_vis.shape:
        raise ValueError(f"grid mismatch gt {gt.shape} pred {pred.shape} mask {M_vis.shape}.")
    M_miss = ~M_vis
    rows: List[dict] = []
    for name, lab in struct_values.items():
        gt_k = (gt == int(lab))
        pred_k = (pred == int(lab))
        vis = compute_region_confusion(pred_k, gt_k, M_vis)
        whole = compute_whole_metrics(pred_k, gt_k, spacing_zyx, surface=whole_surface)
        vol = compute_volume_metrics(pred_k, gt_k, voxel_mm3)          # whole-structure volume
        row = dict(structure=name,
                   visible_dice=vis["dice"], visible_gt_voxels=vis["gt_voxels"],
                   visible_fp_voxels=vis["fp_voxels"], visible_fn_voxels=vis["fn_voxels"],
                   whole_dice=whole["dice"], vol_pred_mm3=vol["vol_pred_mm3"],
                   vol_gt_mm3=vol["vol_gt_mm3"], vol_signed_bias_mm3=vol["vol_signed_bias_mm3"],
                   vol_abs_error_mm3=vol["vol_abs_error_mm3"],
                   centroid_error_vox=vol["centroid_error_vox"])
        if is_full_fov:
            row.update(missing_dice=float("nan"), missing_gt_voxels=0,
                       missing_fp_voxels=0, missing_fn_voxels=0,
                       missing_recall=float("nan"), missing_precision=float("nan"))
        else:
            miss = compute_region_confusion(pred_k, gt_k, M_miss)
            row.update(missing_dice=miss["dice"], missing_gt_voxels=miss["gt_voxels"],
                       missing_fp_voxels=miss["fp_voxels"], missing_fn_voxels=miss["fn_voxels"],
                       missing_recall=miss["recall"], missing_precision=miss["precision"])
        rows.append(row)
    return rows


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(f"[fovc-eval] {msg}")


def run_epoch(
    map_json: str,
    pred_dir: str,
    fov_mask_dir: Optional[str],
    epoch: int,
    structures: Dict[str, int] = None,
    completion_manifest: Optional[str] = None,
    whole_surface: bool = False,
) -> List[dict]:
    """STRICT (P0-3): every expected case must have a prediction, GT and FOV mask;
    any missing/mismatch/extra/geometry failure RAISES. Returns
    len(cases)×len(structures) rows.

    NOTE (masi-55): nibabel load + resample + label remap live in eval_common and
    mirror eval_corrector; reconcile only if those helpers change.
    """
    from diagnostics.eval_common import (load_and_remap_gt, load_fov_mask_on_gt_grid,
                                         load_prediction_on_gt_grid)

    structures = structures or STRUCT_VALUES
    mp = json.loads(Path(map_json).read_text())
    cases = mp["cases"]
    struct_names = list(structures.keys())
    pdir = Path(pred_dir)
    mdir = Path(fov_mask_dir) if fov_mask_dir else None

    # optional cross-check against the completion manifest (revised-plan §6.4).
    if completion_manifest:
        man = json.loads(Path(completion_manifest).read_text())
        recs = man["records"] if isinstance(man, dict) and "records" in man else man
        man_by_case = {r["case_id"]: r for r in recs}
        for cid, c in cases.items():
            _require(cid in man_by_case, f"map case {cid!r} not in completion manifest")
            r = man_by_case[cid]
            _require(str(r.get("crop_type", "full" if r.get("is_full_fov") else "")) ==
                     str(c.get("crop_type")), f"crop_type mismatch for {cid!r}")

    # STRICT: no unexpected extra prediction files in the predict dir.
    expected_pred = {str(c["pred_file"]) for c in cases.values()
                     if not Path(c["pred_file"]).is_absolute()}
    if pdir.exists():
        actual_pred = {p.name for p in pdir.glob("*.nii.gz")}
        extra = actual_pred - expected_pred
        _require(not extra, f"unexpected extra prediction(s) in {pdir}: {sorted(extra)[:5]}")

    out: List[dict] = []
    seen = set()
    for cid, c in sorted(cases.items()):
        _require(cid not in seen, f"duplicate case {cid!r}")
        seen.add(cid)
        pf = c["pred_file"]
        pred_path = Path(pf) if Path(pf).is_absolute() else (pdir / pf)
        gt_path = Path(c["gt_label_path"])
        _require(pred_path.exists(), f"missing prediction for {cid!r}: {pred_path}")
        _require(gt_path.exists(), f"missing GT for {cid!r}: {gt_path}")
        is_full = bool(c.get("is_full_fov"))

        gt_nn, gt_img = load_and_remap_gt(gt_path, c["gt_struct_to_value"], struct_names)
        pred_nn = load_prediction_on_gt_grid(pred_path, gt_img)

        if is_full:
            M_visible = np.ones(gt_nn.shape, dtype=bool)              # whole volume acquired
        else:
            mf = c.get("fov_mask_file")
            _require(mf is not None, f"map case {cid!r} has no fov_mask_file")
            mpth = Path(mf) if Path(mf).is_absolute() else ((mdir / mf) if mdir else Path(mf))
            _require(mpth.exists(), f"missing FOV mask for {cid!r}: {mpth}")
            M_visible = load_fov_mask_on_gt_grid(mpth, gt_img)

        _require(pred_nn.shape == gt_nn.shape == M_visible.shape,
                 f"geometry failure for {cid!r}: gt {gt_nn.shape} pred {pred_nn.shape} "
                 f"mask {M_visible.shape}")

        zooms = np.asarray(gt_img.header.get_zooms()[:3], dtype=float)
        voxel_mm3 = float(np.prod(zooms))
        crop_type = "full" if is_full else str(c["crop_type"])
        severity = 0 if is_full else int(c["severity"])
        for row in evaluate_case_arrays(gt_nn, pred_nn, M_visible, structures, voxel_mm3,
                                        is_full, spacing_zyx=zooms, whole_surface=whole_surface):
            out.append(dict(epoch=int(epoch), subject_id=str(c["subject_id"]), case_id=cid,
                            crop_type=crop_type, severity=severity, **row))

    # STRICT: exact expected row count.
    exp = len(cases) * len(structures)
    _require(len(out) == exp, f"row count {len(out)} != expected {exp} "
             f"({len(cases)} cases × {len(structures)} structures)")
    print(f"[fovc-eval] epoch {epoch}: {len(out)} rows ({len(cases)} cases × {len(structures)} structs).")
    return out


def write_long(rows: List[dict], out_csv: str, append: bool = False) -> None:
    p = Path(out_csv)
    p.parent.mkdir(parents=True, exist_ok=True)
    exists = p.exists()
    with open(p, "a" if append else "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LONG_COLUMNS)
        if not (append and exists):
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in LONG_COLUMNS})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", help="eval_cases_map.json (build_fov_completion_evalset.py)")
    ap.add_argument("--pred-dir", help="dir holding the epoch's predictions")
    ap.add_argument("--fov-mask-dir", default=None, help="fovMaskTs/ (acquired-FOV masks)")
    ap.add_argument("--completion-manifest", default=None, help="optional strict cross-check")
    ap.add_argument("--epoch", type=int)
    ap.add_argument("--out-csv")
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--whole-surface", action="store_true",
                    help="also ASSD/HD95/NSD (needs simulation.evaluation.metrics).")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return _selftest()
    for req in ("map", "pred_dir", "epoch", "out_csv"):
        if getattr(args, req) in (None, ""):
            ap.error(f"--{req.replace('_', '-')} is required (or use --self-test).")
    rows = run_epoch(args.map, args.pred_dir, args.fov_mask_dir, args.epoch,
                     completion_manifest=args.completion_manifest, whole_surface=args.whole_surface)
    write_long(rows, args.out_csv, append=args.append)
    print(f"[fovc-eval] wrote {len(rows)} rows -> {args.out_csv}")
    return 0


def _selftest() -> int:
    S = 40
    gt = np.zeros((S, S, S), np.int16)
    gt[5:35, 8:12, 8:12] = 1                    # ON: a bar crossing the FOV cut
    gt[15:25, 15:25, 15:25] = 3                 # Globe: a cube straddling the cut
    struct_values = {"ON": 1, "Globe": 3}
    M_visible = np.zeros((S, S, S), bool)
    M_visible[20:] = True                       # acquired = z>=20
    M_missing = ~M_visible

    # (a) perfect prediction -> all region Dice 1, no FP
    rows = evaluate_case_arrays(gt, gt.copy(), M_visible, struct_values, 1.0, False, (1, 1, 1))
    by = {r["structure"]: r for r in rows}
    for st, lab in struct_values.items():
        assert abs(by[st]["missing_dice"] - 1.0) < 1e-9 and abs(by[st]["visible_dice"] - 1.0) < 1e-9
        assert by[st]["missing_fp_voxels"] == 0 and by[st]["visible_fp_voxels"] == 0
        assert by[st]["missing_gt_voxels"] == int(((gt == lab) & M_missing).sum())
        assert abs(by[st]["vol_signed_bias_mm3"]) < 1e-6

    # (b) HALLUCINATE ON in the missing region -> missing_fp>0, missing_precision<1,
    #     visible untouched (the key hallucination signal)
    pred = gt.copy()
    pred[16:19, 20:23, 20:23] = 1               # 27 FP ON voxels, all in missing (z<20? no: z16:19<20)
    # z16:19 is in the MISSING region (z<20) -> correct: these are missing-side FPs
    rows = evaluate_case_arrays(gt, pred, M_visible, struct_values, 1.0, False, (1, 1, 1))
    by = {r["structure"]: r for r in rows}
    assert by["ON"]["missing_fp_voxels"] == 27, by["ON"]["missing_fp_voxels"]
    assert by["ON"]["visible_fp_voxels"] == 0
    assert by["ON"]["missing_precision"] < 1.0
    print("hallucination: ON missing_fp=27, missing_precision=%.3f visible_fp=0"
          % by["ON"]["missing_precision"])

    # (c) full-FOV row -> missing NaN/0, visible = whole
    rows = evaluate_case_arrays(gt, gt.copy(), np.ones_like(M_visible), struct_values, 1.0, True, (1, 1, 1))
    for r in rows:
        assert np.isnan(r["missing_dice"]) and r["missing_gt_voxels"] == 0 and r["missing_fp_voxels"] == 0
        assert abs(r["visible_dice"] - 1.0) < 1e-9

    # (d) long-schema round-trip
    import tempfile
    tagged = [dict(epoch=25, subject_id="000", case_id="fov_000_axial_rm35",
                   crop_type="axial", severity=35, **r)
              for r in evaluate_case_arrays(gt, pred, M_visible, struct_values, 1.0, False, (1, 1, 1))]
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "m.csv"
        write_long(tagged, str(out))
        import pandas as pd
        df = pd.read_csv(out)
        assert list(df.columns) == LONG_COLUMNS, list(df.columns)
        assert "subject_id" in df.columns and (df["epoch"] == 25).all()
    print("FOV-COMPLETION-EVAL SELF-TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
