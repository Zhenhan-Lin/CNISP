#!/usr/bin/env python3
"""Diagnose the obs-vs-GT frame mismatch in the FOV patch pipeline (no CNISP fit).

The corrector aligns the DENSE target (from the full-res gt_candidate_pred) and each
per-step OBSERVATION (from the truncated nnUNet pred) SEPARATELY -- each canonically
aligned on its OWN globe centroid (see engine/infer.py::_observed_meta_path_for:
"a different crop than the dense target patch"). Under FOV truncation the observed
globe centroid drifts, so the two patches are NOT the same physical frame. Comparing
sub_obs vs sub_gt by voxel index (as the first-cut harness did) therefore mixes the
shape error with a frame offset, and the 64 mm inner crop -- re-centered on the
truncated globe's visible centroid -- can clip part of the still-visible eye.

This tool quantifies both, per (case, step), WITHOUT running CNISP:

  * offset_world_mm      : || obs crop_centroid_world - GT crop_centroid_world ||
                           (the drift between the two patch frames)
  * visible_retained     : fraction of the OBS foreground that survives the 64 mm
                           inner crop (1.0 = the visible eye is fully kept)
  * sub_fg_offset_vox    : distance between the sub_obs and sub_dense foreground
                           centroids in the 64 mm sub-patch (0 = co-registered)

Usage:
  python nnunet-c/diagnostics/fov_frame_diag.py \
      --config nnunet-c/configs/corrector_fov.yaml \
      -p configs/paths.yaml -t configs/train_v6_5_gt.yaml -c configs/test_corrector.yaml \
      -m orbital_ad_v6_5_gt --experiment fov --test-label-source nnunet_pred \
      --steps 50,65,80 --test-casefile corrector_train_cases_fov.txt \
      --aligned-dir nnunet-c/data_fov_pereye_test/aligned_patch \
      --out-csv nnunet-c/data_fov_pereye_test/frame_diag.csv
  python nnunet-c/diagnostics/fov_frame_diag.py --self-test
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CNISP_DIR = REPO / "orbital_shape_prior_st1"


def _fg_centroid_vox(vol):
    idx = np.argwhere(np.asarray(vol) > 0)
    return idx.mean(axis=0) if idx.size else None


def run(args) -> int:
    import yaml
    for p in (str(CNISP_DIR), str(REPO)):
        if p not in sys.path:
            sys.path.insert(0, p)
    from engine.dataset import load_casenames, inner_crop_64mm
    from engine.infer import (_load_labels_dense_per_case, _build_label_obs_loader,
                              _meta_path_for_case, _observed_meta_path_for)
    from engine.test_label_sources import build_run_layout

    def _cnisp(p):
        p = Path(p)
        return p if p.is_absolute() else (CNISP_DIR / p)

    params = {}
    for key in (args.paths, args.train_config, args.config_cnisp):
        if key:
            with open(_cnisp(key)) as f:
                params.update(yaml.safe_load(f) or {})
    params["model_name"] = args.model_name or params.get("model_name", "m")
    params["test_label_source"] = args.test_label_source
    params["experiment"] = args.experiment
    params["run_tag"] = params.get("run_tag", "corrector_gt")
    if args.test_casefile:
        params["test_casefile"] = args.test_casefile
    if args.aligned_dir:
        params["aligned_dir"] = args.aligned_dir

    layout = build_run_layout(params)
    casefiles_dir = Path(params["casefiles_dir"])
    casenames_all = load_casenames(casefiles_dir / params["test_casefile"])
    labels_dense, spacings_dense, casenames = _load_labels_dense_per_case(layout, casenames_all)
    label_obs_loader = _build_label_obs_loader(layout)
    gt_meta_for = _meta_path_for_case(layout)
    obs_meta_for = _observed_meta_path_for(layout)
    if label_obs_loader is None or obs_meta_for is None:
        print("[frame-diag] need test_label_source=nnunet_pred (obs override + obs metadata).",
              file=sys.stderr)
        return 2

    steps_list = [int(s) for s in args.steps.split(",") if s.strip()]
    rows = []
    for ci, cn in enumerate(casenames):
        gmp = Path(gt_meta_for(cn))
        gt_meta = json.load(open(gmp)) if gmp.is_file() else {}
        C_gt = np.asarray(gt_meta.get("crop_centroid_world", [np.nan] * 3), float)
        spacing_dense = spacings_dense[ci]
        offset_dense = spacing_dense / 2.0
        for step in steps_list:
            ov = label_obs_loader(cn, step, 0)
            omp = obs_meta_for(cn, step, 0)
            if ov is None:
                continue
            label_obs, spacing_obs, offset_obs = ov
            obs_meta = json.load(open(omp)) if Path(omp).is_file() else {}
            C_obs = np.asarray(obs_meta.get("crop_centroid_world", [np.nan] * 3), float)
            offset_world = float(np.linalg.norm(C_obs - C_gt)) if np.isfinite(C_obs).all() \
                and np.isfinite(C_gt).all() else float("nan")

            inner = inner_crop_64mm(label_obs, spacing_obs, offset_obs,
                                    labels_dense[ci], spacing_dense, offset_dense)
            sub_obs = np.asarray(inner["sub_sparse"])
            sub_gt = np.asarray(inner["sub_dense"])
            obs_fg = int((np.asarray(label_obs) > 0).sum())
            sub_obs_fg = int((sub_obs > 0).sum())
            visible_retained = (sub_obs_fg / obs_fg) if obs_fg else float("nan")
            c_obs = _fg_centroid_vox(sub_obs)
            c_gt = _fg_centroid_vox(sub_gt)
            sub_fg_offset = (float(np.linalg.norm(c_obs - c_gt))
                             if c_obs is not None and c_gt is not None else float("nan"))
            row = {"case": cn, "step": step,
                   "offset_world_mm": round(offset_world, 3),
                   "visible_retained": round(visible_retained, 4),
                   "sub_fg_offset_vox": round(sub_fg_offset, 3),
                   "obs_fg_vox": obs_fg, "sub_obs_fg_vox": sub_obs_fg,
                   "sub_shape": list(int(s) for s in sub_gt.shape)}
            rows.append(row)
            print(f"  {cn} step{step:02d}: obs-GT centroid offset={offset_world:.2f}mm  "
                  f"visible_eye_kept_in_64mm={visible_retained:.3f}  "
                  f"sub_obs-vs-sub_gt fg offset={sub_fg_offset:.2f}vox")

    if not rows:
        print("[frame-diag] no cases produced output", file=sys.stderr)
        return 1
    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow({k: (json.dumps(v) if isinstance(v, list) else v)
                            for k, v in r.items()})
        print(f"[frame-diag] wrote {args.out_csv}")
    off = [r["offset_world_mm"] for r in rows if r["offset_world_mm"] == r["offset_world_mm"]]
    vis = [r["visible_retained"] for r in rows if r["visible_retained"] == r["visible_retained"]]
    if off:
        print(f"[frame-diag] mean obs-GT centroid offset = {np.mean(off):.2f} mm "
              f"(max {np.max(off):.2f}) -> nonzero == obs/GT are different frames")
    if vis:
        print(f"[frame-diag] mean visible-eye retained in 64mm crop = {np.mean(vis):.3f} "
              f"(min {np.min(vis):.3f}) -> <1 == the crop clips the still-visible eye")
    return 0


def _self_test() -> int:
    # offset math + centroid helpers on synthetic arrays.
    a = np.zeros((10, 10, 10), np.int16); a[2:5, 2:5, 2:5] = 1
    b = np.zeros((10, 10, 10), np.int16); b[5:8, 5:8, 5:8] = 1
    ca, cb = _fg_centroid_vox(a), _fg_centroid_vox(b)
    assert np.allclose(ca, [3, 3, 3]) and np.allclose(cb, [6, 6, 6])
    assert abs(float(np.linalg.norm(ca - cb)) - np.sqrt(27)) < 1e-6
    C_obs, C_gt = np.array([12.0, 4.0, 7.0]), np.array([9.0, 6.0, 7.0])
    assert abs(float(np.linalg.norm(C_obs - C_gt)) - np.sqrt(13)) < 1e-6
    assert _fg_centroid_vox(np.zeros((4, 4, 4))) is None
    print("centroid + offset math OK")
    print("\nALL FOV-FRAME-DIAG SELF-TESTS PASSED")
    return 0


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="nnunet-c/configs/corrector_fov.yaml")
    ap.add_argument("-m", "--model-name", default=None)
    ap.add_argument("-p", "--paths", default="configs/paths.yaml")
    ap.add_argument("-t", "--train-config", default=None)
    ap.add_argument("-c", "--config-cnisp", default=None)
    ap.add_argument("--experiment", default="fov")
    ap.add_argument("--test-label-source", default="nnunet_pred")
    ap.add_argument("--test-casefile", default=None)
    ap.add_argument("--aligned-dir", default=None)
    ap.add_argument("--steps", default="50,65,80")
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--self-test", action="store_true")
    return ap


if __name__ == "__main__":
    a = build_parser().parse_args()
    sys.exit(_self_test() if a.self_test else run(a))
