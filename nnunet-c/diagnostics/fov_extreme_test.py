#!/usr/bin/env python3
"""Extreme-case test for the CNISP output under type-2 (box) FOV truncation.

Scouts how the CNISP-COMPLETED prior behaves when the box truncation is pushed to
extremes -- aggressive keep_fractions and each of the 4 corners -- BEFORE committing a
full training run. It quantifies the failure modes surfaced in the design discussion:

  * recovery_trunc   -- of the reference anatomy the box BLANKED, how much did the
                        CNISP-on-truncated prior reproduce? (does CNISP complete the FOV?)
  * centroid drift   -- mm between the CNISP prior's foreground / globe centroid and the
                        reference's (mislocation under truncation; TTO cannot fix large drift).
  * extent / clip    -- CNISP foreground bbox extent vs reference, per axis (< 1 => the
                        prior under-covers). If the CNISP foreground TOUCHES the array
                        boundary it may have been silently clipped by the fixed decode
                        patch (a large / drifted eye that ran off the 64 mm window).

The EXTREME INPUTS reuse the existing builder (no new degradation code):
    for C in SL SR IL IR; do
      python nnunet-c/scripts/build_fov_truncated_data.py --mode box --corner "$C" \
        --keep-fractions 0.25,0.35,0.5 --out-data-root "nnunet-c/data_fov_extreme_${C}"
    done
Then run the CNISP re-fit (RUNBOOK_FOV.md Step 2) on each and point --analyze at its output.

Modes
-----
  --analyze : reference seg (untruncated gt_candidate_pred / prior) + the CNISP output
              (completed iso prior) + the fov sidecar -> a diagnostics table (needs the
              model's outputs; run on the box that has them).
  --self-test : run the pure-numpy analysis on synthetic arrays (no model/data) to verify
              the logic. Runs anywhere.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import nibabel as nib  # noqa: E402

# nnUNet target scheme {ON:1, Recti:2, Globe:3, Fat:4}; kept local so --self-test needs
# no repo imports (the real labels are pulled in --analyze only).
_SELFTEST_LABELS = {"ON": 1, "Recti": 2, "Globe": 3, "Fat": 4}


# ── pure-numpy analysis (unit-testable, no torch/model) ───────────────────────
def _bbox(mask: np.ndarray):
    idx = np.argwhere(mask)
    if idx.size == 0:
        return None
    return idx.min(0), idx.max(0)


def _com(mask: np.ndarray):
    idx = np.argwhere(mask)
    return idx.mean(0) if idx.size else None


def _truncated_mask(shape, visible_box) -> np.ndarray:
    """Blanked-FOV mask = complement of the visible box (per-axis [lo, hi))."""
    vis = np.zeros(shape, dtype=bool)
    sl = tuple(slice(int(lo), int(hi)) for (lo, hi) in visible_box)
    vis[sl] = True
    return ~vis


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    """Dice of two boolean masks (1.0 when both are empty, by convention)."""
    sa, sb = int(a.sum()), int(b.sum())
    if sa + sb == 0:
        return 1.0
    return 2.0 * int(np.logical_and(a, b).sum()) / (sa + sb)


def analyze_case(cnisp_nn: np.ndarray, ref_nn: np.ndarray, visible_box,
                 spacing, labels) -> dict:
    """Diagnostics for one extreme case. ``cnisp_nn`` and ``ref_nn`` are same-shape
    nnUNet label arrays ({1..4}) on the SAME grid; the blanked-FOV region is the
    complement of ``visible_box`` (per-axis [lo, hi)); ``spacing`` is mm/voxel."""
    T = _truncated_mask(cnisp_nn.shape, visible_box)
    fg_ref, fg_c = ref_nn > 0, cnisp_nn > 0
    ref_T = fg_ref & T
    out = {
        # of the reference anatomy inside the blanked FOV, fraction CNISP reproduced
        "recovery_trunc": round(float((fg_c & ref_T).sum()) / max(1, int(ref_T.sum())), 3),
        # of CNISP's foreground inside the blanked FOV, fraction NOT matching the ref
        "spurious_trunc": round(float((fg_c & ~fg_ref & T).sum())
                                / max(1, int((fg_c & T).sum())), 3),
        "n_ref_trunc_vox": int(ref_T.sum()),
    }
    vis = ~T                                            # imaged (visible) region
    per = {}
    for name, L in labels.items():
        rm, cm = ref_nn == L, cnisp_nn == L
        rmT = rm & T
        per[name] = {
            "vol_ratio": round(float(cm.sum()) / max(1, int(rm.sum())), 3),
            "recovery_trunc": round(float((cm & rmT).sum()) / max(1, int(rmT.sum())), 3),
            # Dice of CNISP vs reference: whole volume, and split by FOV region.
            "dice": round(_dice(cm, rm), 4),
            "dice_visible": round(_dice(cm & vis, rm & vis), 4),
            "dice_truncated": round(_dice(cm & T, rm & T), 4),
        }
    out["per_structure"] = per
    out["dice_mean"] = round(float(np.mean([per[n]["dice"] for n in labels])), 4)
    out["dice_mean_visible"] = round(
        float(np.mean([per[n]["dice_visible"] for n in labels])), 4)
    out["dice_mean_truncated"] = round(
        float(np.mean([per[n]["dice_truncated"] for n in labels])), 4)

    sp = np.asarray(spacing, dtype=float)

    def _drift(mask_ref, mask_c):
        cr, cc = _com(mask_ref), _com(mask_c)
        if cr is None or cc is None:
            return None
        return round(float(np.linalg.norm((cr - cc) * sp)), 2)

    out["centroid_drift_mm"] = _drift(fg_ref, fg_c)
    gl = labels.get("Globe")
    out["globe_drift_mm"] = _drift(ref_nn == gl, cnisp_nn == gl) if gl is not None else None

    br, bc = _bbox(fg_ref), _bbox(fg_c)
    if br is not None and bc is not None:
        er = (br[1] - br[0] + 1).astype(float)
        ec = (bc[1] - bc[0] + 1).astype(float)
        out["extent_ratio"] = [round(float(x), 3) for x in (ec / np.maximum(1.0, er))]
        # CNISP foreground reaching the array edge -> possibly clipped by the patch.
        out["cnisp_touches_boundary"] = [
            bool(bc[0][a] <= 0 or bc[1][a] >= cnisp_nn.shape[a] - 1) for a in range(3)]
    return out


def _flag(d: dict) -> str:
    """One-line health flag for a case (heuristic thresholds; tune per cohort)."""
    bad = []
    if d.get("recovery_trunc", 1.0) < 0.5:
        bad.append("LOW-RECOVERY")
    if (d.get("globe_drift_mm") or 0) > 3.0:
        bad.append("GLOBE-DRIFT")
    if any(d.get("cnisp_touches_boundary", [])):
        bad.append("BOUNDARY-CLIP?")
    if min(d.get("extent_ratio", [1.0])) < 0.7:
        bad.append("UNDER-COVER")
    return ",".join(bad) if bad else "ok"


# ── --analyze: run on real model outputs ──────────────────────────────────────
def run_analyze(args) -> int:
    from lib import resample as _rs                      # noqa: E402  (repo import)

    trunc = json.load(open(args.trunc_manifest))
    ref_img = nib.load(args.ref)
    ref_nn = np.asanyarray(ref_img.dataobj).astype(np.int16)     # gt_candidate_pred: nnUNet {1..4}
    spacing = np.asarray(ref_img.header.get_zooms()[:3], dtype=float)

    info = (trunc.get(str(args.source_id), {}) or {}).get(str(args.step))
    if not info or "visible_box" not in info:
        print(f"[fov-test] no box sidecar entry (visible_box) for source={args.source_id} "
              f"step={args.step} (is this a --mode box / --min-retains build?)", file=sys.stderr)
        return 2
    if tuple(int(s) for s in ref_nn.shape) != tuple(int(s) for s in info.get("source_shape", ())):
        print(f"[fov-test] ref grid {ref_nn.shape} != source_shape {info.get('source_shape')}; "
              f"the box mask needs the ref on the truncation source grid.", file=sys.stderr)
        return 2

    # CNISP completed iso prior -> resample onto the ref grid (nearest; never move ref).
    cnisp_img = nib.load(args.cnisp)
    cnisp_rs = _rs.resample_to_grid(cnisp_img, ref_img.shape[:3], ref_img.affine, order=0)
    cnisp_nn = np.asanyarray(cnisp_rs.dataobj).astype(np.int16)

    from lib.labels import NNUNET_LABELS                 # noqa: E402
    d = analyze_case(cnisp_nn, ref_nn, info["visible_box"], spacing, dict(NNUNET_LABELS))
    d.update({"source_id": args.source_id, "step": args.step, "corner": info.get("corner"),
              "min_retain": info.get("min_retain"), "keep_fraction": info.get("keep_fraction"),
              "flag": _flag(d)})

    print(json.dumps(d, indent=2))
    print(f"\n[fov-test] source={args.source_id} step={args.step} corner={info.get('corner')} "
          f"->  {d['flag']}")
    print(f"  recovery(trunc)={d['recovery_trunc']}  globe_drift={d['globe_drift_mm']}mm  "
          f"extent_ratio={d.get('extent_ratio')}  boundary={d.get('cnisp_touches_boundary')}")
    print(f"  DICE(CNISP vs ref)  mean={d['dice_mean']}  "
          f"visible={d['dice_mean_visible']}  truncated={d['dice_mean_truncated']}")
    for name, s in d["per_structure"].items():
        print(f"    {name:6s}: dice={s['dice']:.4f}  visible={s['dice_visible']:.4f}  "
              f"truncated={s['dice_truncated']:.4f}")
    for name, pe in (info.get("per_eye") or {}).items():        # built per-eye retention (QC)
        print(f"  built eye {name}: ret_total={pe.get('ret_total')} "
              f"ret_ON={pe.get('ret_ON')} bind={pe.get('binding_constraint')}")
    if args.out_csv:
        flat = {k: v for k, v in d.items() if k != "per_structure"}
        for name, s in d["per_structure"].items():
            flat[f"vol_ratio_{name}"] = s["vol_ratio"]
            flat[f"recovery_trunc_{name}"] = s["recovery_trunc"]
            flat[f"dice_{name}"] = s["dice"]
            flat[f"dice_truncated_{name}"] = s["dice_truncated"]
        write_header = not Path(args.out_csv).exists()
        with open(args.out_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(flat))
            if write_header:
                w.writeheader()
            w.writerow(flat)
        print(f"[fov-test] appended -> {args.out_csv}")
    return 0


# ── --self-test: synthetic, no model/data ─────────────────────────────────────
def run_self_test() -> int:
    shape = (30, 30, 30)
    ref = np.zeros(shape, np.int16)
    ref[8:22, 8:22, 8:16] = _SELFTEST_LABELS["ON"]        # ON (posterior half on axis2)
    ref[8:22, 8:22, 16:22] = _SELFTEST_LABELS["Globe"]    # Globe (anterior half on axis2)
    # blank a corner: keep axis0 [12,30) (blanks low-x) and axis2 [0,17) (blanks high-z),
    # so part of BOTH structures lands in the blanked FOV.
    vbox = [[12, 30], [0, 30], [0, 17]]
    spacing = [1.0, 1.0, 1.0]

    good = ref.copy()                                     # CNISP recovers everything
    bad = np.where(_truncated_mask(shape, vbox), 0, ref)  # CNISP fills only the visible box

    dg = analyze_case(good, ref, vbox, spacing, _SELFTEST_LABELS)
    db = analyze_case(bad, ref, vbox, spacing, _SELFTEST_LABELS)
    print("good:", json.dumps(dg), "->", _flag(dg))
    print("bad :", json.dumps(db), "->", _flag(db))

    assert dg["recovery_trunc"] > 0.95, dg
    assert db["recovery_trunc"] < 0.05, db          # never fills the blanked FOV
    assert dg["centroid_drift_mm"] < 1e-6, dg
    assert db["centroid_drift_mm"] > dg["centroid_drift_mm"], (db, dg)   # visible-only shifts CoM
    assert min(dg["extent_ratio"]) > 0.95, dg
    assert min(db["extent_ratio"]) < min(dg["extent_ratio"]), (db, dg)   # bad under-covers
    assert _flag(dg) == "ok" and "LOW-RECOVERY" in _flag(db), (dg, db)
    # per-structure recovery separates the two
    assert dg["per_structure"]["Globe"]["recovery_trunc"] > 0.95
    assert db["per_structure"]["Globe"]["recovery_trunc"] < 0.05
    # Dice: perfect recovery -> ~1 everywhere. Visible-only fill -> the truncated-region
    # Dice of the CLIPPED structures is ~0 (absent structures Dice=1.0 by the both-empty
    # convention, so check present structures, not the mean).
    assert dg["dice_mean"] > 0.99 and dg["dice_mean_truncated"] > 0.99, dg
    assert db["per_structure"]["Globe"]["dice_truncated"] < 0.05, db
    assert db["per_structure"]["ON"]["dice_truncated"] < 0.05, db
    assert db["dice_mean_visible"] > 0.99, db        # visible region is perfect for both

    _self_test_min_retain_calibration()
    print("\nSELF-TEST PASSED")
    return 0


def _self_test_min_retain_calibration() -> None:
    """Verify the build-side GLOBAL-box min-retain calibration: one corner box, both eyes
    >= floor, and the box is a real boundary-connected truncation (imports the build
    helpers; SKIPPED with a note if the repo/deps aren't importable)."""
    import importlib.util
    bpath = Path(__file__).resolve().parents[1] / "scripts" / "build_fov_truncated_data.py"
    try:
        spec = importlib.util.spec_from_file_location("bfov_selftest", bpath)
        b = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(b)
    except Exception as e:                              # bare env without repo deps
        print(f"[min-retain calibration self-test SKIPPED: {type(e).__name__}: {e}]")
        return

    shape = (40, 40, 50)                                # RAS+: axis0=R/L, axis1=A/P(kept), axis2=S/I
    axcodes = ("R", "A", "S")
    T = 0.5
    eye_data, allfg = {}, np.zeros(shape, bool)
    for name, a0 in (("OD", (28, 38)), ("OS", (2, 12))):   # two orbits at different L-R
        fg = np.zeros(shape, bool); fg[a0[0]:a0[1], 10:30, 8:42] = True
        on = np.zeros(shape, bool); on[a0[0] + 2:a0[0] + 8, 10:30, 20:28] = True
        eye_data[name] = {"fg": fg, "on": on, "structs": {"ON": on, "Globe": fg & ~on,
                          "Recti": np.zeros(shape, bool), "Fat": np.zeros(shape, bool)}}
        allfg |= fg
    idx = np.argwhere(allfg); lo, hi = idx.min(0), idx.max(0)
    f, per_eye = b._calibrate_global_box(shape, axcodes, lo, hi, "SL", eye_data, T, T)
    print("min-retain calib f=", f, {k: (per_eye[k]["ret_total"], per_eye[k]["ret_ON"])
                                      for k in per_eye})
    for name, pe in per_eye.items():                    # BOTH eyes hold the floor
        assert pe["ret_total"] >= T - 0.03, (name, pe)
        assert pe["ret_ON"] >= T - 0.03, (name, pe)
    assert min(min(pe["ret_total"], pe["ret_ON"]) for pe in per_eye.values()) >= T - 0.03
    assert f < 0.999, ("must actually cut", f)
    # single global box = a REAL boundary-connected truncation (anterior kept, cut slabs
    # reach the image edge -- not an interior hole).
    win = b._box_full_windows(shape, axcodes, lo, hi, "SL", f)
    assert win[1] == (0, shape[1]), ("anterior axis must stay full", win)
    assert win[0][0] == 0 or win[0][1] == shape[0], ("L/R slab must reach an edge", win)
    assert win[2][0] == 0 or win[2][1] == shape[2], ("S/I slab must reach an edge", win)
    print("min-retain calibration self-test: single global box, both eyes hold "
          "ret_total & ret_ON >= T, cut reaches the image edge")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true",
                    help="run the synthetic analyzer checks (no model/data).")
    ap.add_argument("--analyze", action="store_true",
                    help="analyze a CNISP output for one extreme (case, step).")
    ap.add_argument("--ref", help="reference seg NIfTI (untruncated gt_candidate_pred / prior, "
                                  "nnUNet {1..4}) on the truncation source grid.")
    ap.add_argument("--cnisp", help="CNISP completed iso-prior NIfTI for the truncated case.")
    ap.add_argument("--trunc-manifest", help="fov_truncation_manifest.json (box build).")
    ap.add_argument("--source-id", help="source_id key in the sidecar.")
    ap.add_argument("--step", help="pseudo-step (e.g. 35) key in the sidecar.")
    ap.add_argument("--out-csv", default=None, help="append the flat row to this CSV.")
    args = ap.parse_args()

    if args.self_test:
        return run_self_test()
    if args.analyze:
        missing = [k for k in ("ref", "cnisp", "trunc_manifest", "source_id", "step")
                   if getattr(args, k.replace("-", "_")) is None]
        if missing:
            print(f"[fov-test] --analyze needs {missing}", file=sys.stderr)
            return 2
        return run_analyze(args)
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
