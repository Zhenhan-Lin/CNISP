#!/usr/bin/env python3
"""FOV-completion region-split evaluator — emits the LONG metrics table consumed by
diagnostics/select_fov_checkpoint_driver.py, for ONE snapshot (epoch).

It is the FOV analogue of diagnostics/eval_corrector.py, but:
  * it emits the driver's exact long schema (one row per structure), not the wide
    per-case ``dice_<struct>`` CSV;
  * it emits the per-region GT voxel counts (``missing_gt_voxels`` /
    ``visible_gt_voxels``) that eval_corrector does not — these drive the driver's
    independent missing/visible validity filters;
  * the region split comes from the COMPLETION manifest's ``visible_box`` (records
    list keyed by case_id), not from eval_corrector's ``[source_id][step]`` manifest.

Long-table columns (appended per epoch; concatenated by run_fov_completion_sweep.sh):
    epoch, case_id, crop_type, severity, structure,
    missing_dice, visible_dice, missing_gt_voxels, visible_gt_voxels
``crop_type == "full"`` (is_full_fov) rows carry the whole-volume Dice in
visible_dice, missing_dice = NaN, missing_gt_voxels = 0, severity = 0.

RESAMPLE (pinned, identical to eval_corrector): the prediction is resampled onto
each case's native GT grid with order 0; Dice is computed on the GT grid; the GT is
never moved. The ``visible_box`` is a per-axis half-open window on that same
(source) grid, guarded by ``source_shape``.

The pure region-split core (``evaluate_case_arrays``) is unit-tested here with
``--self-test``; the file I/O + resample path needs nibabel + the repo's lib
(masi-55, on real predictions).
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
from lib.fov_region_masks import visible_box_to_mask                     # noqa: E402

# nnU-Net foreground labels (matches lib.labels.NNUNET_LABELS / eval_corrector).
STRUCT_VALUES: Dict[str, int] = {"ON": 1, "Recti": 2, "Globe": 3, "Fat": 4}

LONG_COLUMNS = ["epoch", "case_id", "crop_type", "severity", "structure",
                "missing_dice", "visible_dice", "missing_gt_voxels", "visible_gt_voxels"]


def _dice(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    p = float(pred_bin.sum())
    g = float(gt_bin.sum())
    denom = p + g
    if denom == 0.0:                       # both empty -> perfect by convention
        return 1.0
    return 2.0 * float(np.logical_and(pred_bin, gt_bin).sum()) / denom


def evaluate_case_arrays(
    gt_nn: np.ndarray,
    pred_nn: np.ndarray,
    visible_box,
    struct_values: Dict[str, int],
    is_full_fov: bool,
) -> List[dict]:
    """Region-split per-structure Dice + GT voxel counts for one case.

    ``gt_nn`` / ``pred_nn``: label arrays ({0,1,2,3,4}) on the SAME grid.
    ``visible_box``: [(z_lo,z_hi),(y_lo,y_hi),(x_lo,x_hi)] half-open on that grid.
    Returns rows WITHOUT epoch/case metadata (added by the caller).

    Semantics: ``missing`` = outside the acquired FOV (the truncated-away region the
    corrector must complete); ``visible`` = inside it (must be preserved). Full-FOV
    rows report the whole-volume Dice and carry no missing region.
    """
    gt = np.asarray(gt_nn)
    pred = np.asarray(pred_nn)
    if gt.shape != pred.shape:
        raise ValueError(f"gt {gt.shape} != pred {pred.shape} (resample first).")
    M_vis = visible_box_to_mask(gt.shape, visible_box)
    M_miss = ~M_vis
    rows: List[dict] = []
    for name, lab in struct_values.items():
        gt_k = (gt == int(lab))
        pred_k = (pred == int(lab))
        if is_full_fov:
            rows.append(dict(structure=name, missing_dice=float("nan"),
                             visible_dice=_dice(pred_k, gt_k),
                             missing_gt_voxels=0, visible_gt_voxels=int(gt_k.sum())))
        else:
            rows.append(dict(
                structure=name,
                missing_dice=_dice(pred_k & M_miss, gt_k & M_miss),
                visible_dice=_dice(pred_k & M_vis, gt_k & M_vis),
                missing_gt_voxels=int((gt_k & M_miss).sum()),
                visible_gt_voxels=int((gt_k & M_vis).sum())))
    return rows


def _records_by_case(manifest_path: str) -> Dict[str, dict]:
    man = json.loads(Path(manifest_path).read_text())
    recs = man["records"] if isinstance(man, dict) and "records" in man else man
    return {r["case_id"]: r for r in recs}


def run_epoch(
    map_json: str,
    pred_dir: Optional[str],
    completion_manifest: str,
    epoch: int,
    structures: Dict[str, int] = None,
) -> List[dict]:
    """Load GT + prediction per case (mirroring eval_corrector's resample), join with
    the completion manifest for region + condition metadata, and return long rows.

    NOTE (masi-55): the nibabel load + resample + label remap mirror
    eval_corrector.py exactly; reconcile only if that file's helpers changed.
    """
    import nibabel as nib                                     # noqa: WPS433
    from lib import resample as _rs                            # noqa: WPS433
    from lib.labels import remap_to_nnunet                     # noqa: WPS433

    structures = structures or STRUCT_VALUES
    recs = _records_by_case(completion_manifest)
    mp = json.loads(Path(map_json).read_text())
    cases = mp["cases"]
    struct_names = mp.get("structures", list(structures.keys()))
    pdir = Path(pred_dir) if pred_dir else None

    out: List[dict] = []
    skipped = 0
    for cid, c in sorted(cases.items()):
        rec = recs.get(cid)
        if rec is None:                                       # not a completion case
            continue
        pf = c["pred_file"]
        pred_path = Path(pf) if Path(pf).is_absolute() else ((pdir / pf) if pdir else Path(pf))
        gt_path = Path(c["gt_label_path"])
        if not pred_path.exists() or not gt_path.exists():
            print(f"  [fovc-eval] {cid}: missing pred/gt; skip"); skipped += 1; continue

        gt_img = nib.load(str(gt_path))
        gt_arr = np.asanyarray(gt_img.dataobj)
        stv = {k: int(v) for k, v in c["gt_struct_to_value"].items()}
        gt_nn = remap_to_nnunet(gt_arr, stv, struct_names)     # -> {1,2,3,4}

        # PIN: resample the prediction onto the GT grid (nearest); never move GT.
        pred_img = nib.load(str(pred_path))
        pred_rs = _rs.resample_to_grid(pred_img, gt_img.shape[:3], gt_img.affine, order=0)
        pred_nn = np.asanyarray(pred_rs.dataobj).astype(np.int16)

        is_full = bool(rec.get("is_full_fov"))
        src_shape = tuple(int(s) for s in rec.get("source_shape", gt_nn.shape))
        if tuple(int(s) for s in gt_nn.shape) != src_shape:
            print(f"  [fovc-eval] {cid}: GT grid {gt_nn.shape} != manifest source_shape "
                  f"{src_shape}; skip (axis-order/grid reconcile — see NOTE)."); skipped += 1; continue

        if is_full:
            visible_box = [[0, s] for s in gt_nn.shape]        # whole volume
            crop_type, severity = "full", 0
        else:
            visible_box = rec["visible_box"]
            crop_type, severity = str(rec["crop_type"]), int(rec["severity"])

        for row in evaluate_case_arrays(gt_nn, pred_nn, visible_box, structures, is_full):
            out.append(dict(epoch=int(epoch), case_id=cid, crop_type=crop_type,
                            severity=severity, **row))
    print(f"[fovc-eval] epoch {epoch}: {len(out)} rows from {len(out)//max(1,len(structures))} "
          f"case(s); {skipped} skipped.")
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
    ap.add_argument("--map", help="test_cases_map.json (build_corrector_testset.py)")
    ap.add_argument("--pred-dir", default=None, help="dir for relative pred_file entries")
    ap.add_argument("--completion-manifest", help="fov_completion_manifest.json (records list)")
    ap.add_argument("--epoch", type=int, help="snapshot epoch (from checkpoint_epoch_XXXX)")
    ap.add_argument("--out-csv", help="append these epoch rows to the long metrics CSV")
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return _selftest()
    for req in ("map", "completion_manifest", "epoch", "out_csv"):
        if getattr(args, req) in (None, ""):
            ap.error(f"--{req.replace('_', '-')} is required (or use --self-test).")
    rows = run_epoch(args.map, args.pred_dir, args.completion_manifest, args.epoch)
    write_long(rows, args.out_csv, append=args.append)
    print(f"[fovc-eval] wrote {len(rows)} rows -> {args.out_csv}")
    return 0


def _selftest() -> int:
    S = 40
    gt = np.zeros((S, S, S), np.int16)
    gt[5:35, 8:12, 8:12] = 1                    # ON: a bar crossing the FOV cut
    gt[15:25, 15:25, 15:25] = 3                 # Globe: a cube
    struct_values = {"ON": 1, "Globe": 3}
    # visible FOV = z in [20, 40): the lower half of the ON bar + none of the globe
    visible_box = [[20, S], [0, S], [0, S]]
    M_vis = visible_box_to_mask(gt.shape, visible_box)
    M_miss = ~M_vis

    # (a) perfect prediction -> all region Dice == 1, gt-voxels split correctly
    rows = evaluate_case_arrays(gt, gt.copy(), visible_box, struct_values, is_full_fov=False)
    by = {r["structure"]: r for r in rows}
    for st in ("ON", "Globe"):
        assert abs(by[st]["missing_dice"] - 1.0) < 1e-9 and abs(by[st]["visible_dice"] - 1.0) < 1e-9
    # voxel split matches the mask directly and partitions each structure's total
    for st, lab in struct_values.items():
        gk = (gt == lab)
        assert by[st]["missing_gt_voxels"] == int((gk & M_miss).sum())
        assert by[st]["visible_gt_voxels"] == int((gk & M_vis).sum())
        assert by[st]["missing_gt_voxels"] + by[st]["visible_gt_voxels"] == int(gk.sum())
    # the ON bar (z 5:35) straddles the cut -> both parts non-empty; the Globe cube
    # (z 15:25) also straddles z=20 -> both parts non-empty
    assert by["ON"]["missing_gt_voxels"] > 0 and by["ON"]["visible_gt_voxels"] > 0
    assert by["Globe"]["missing_gt_voxels"] > 0 and by["Globe"]["visible_gt_voxels"] > 0
    print("perfect:", {k: (round(v['missing_dice'], 3), round(v['visible_dice'], 3),
                           v['missing_gt_voxels'], v['visible_gt_voxels']) for k, v in by.items()})

    # (b) damage ONLY the missing region for ON -> missing_dice drops, visible stays 1
    pred = gt.copy()
    pred[(gt == 1) & M_miss] = 0                 # erase ON in the missing region
    rows = evaluate_case_arrays(gt, pred, visible_box, struct_values, is_full_fov=False)
    by = {r["structure"]: r for r in rows}
    assert by["ON"]["missing_dice"] == 0.0, by["ON"]["missing_dice"]
    assert abs(by["ON"]["visible_dice"] - 1.0) < 1e-9
    print("missing-damaged ON:", round(by["ON"]["missing_dice"], 3), round(by["ON"]["visible_dice"], 3))

    # (c) full-FOV row -> missing NaN / missing_gt 0 / visible = whole Dice
    rows = evaluate_case_arrays(gt, gt.copy(), [[0, S], [0, S], [0, S]], struct_values, is_full_fov=True)
    for r in rows:
        assert np.isnan(r["missing_dice"]) and r["missing_gt_voxels"] == 0
        assert abs(r["visible_dice"] - 1.0) < 1e-9
        assert r["visible_gt_voxels"] == int((gt == struct_values[r["structure"]]).sum())
    print("full-fov: missing_dice=NaN, visible=1.0 OK")

    # (d) long-schema round-trip through write_long
    import tempfile
    tagged = [dict(epoch=25, case_id="fov_000_axial_rm35", crop_type="axial", severity=35, **r)
              for r in evaluate_case_arrays(gt, pred, visible_box, struct_values, False)]
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "m.csv"
        write_long(tagged, str(out))
        import pandas as pd
        df = pd.read_csv(out)
        assert list(df.columns) == LONG_COLUMNS, list(df.columns)
        assert set(df["structure"]) == {"ON", "Globe"} and (df["epoch"] == 25).all()
    print("FOV-COMPLETION-EVAL SELF-TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
