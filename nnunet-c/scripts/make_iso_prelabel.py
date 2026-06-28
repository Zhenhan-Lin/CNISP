#!/usr/bin/env python3
"""Build CNISP iso-0.5 corrector prelabels from ALREADY-SAVED latents (post-hoc).

Use this when you don't want to wait for a full 03_infer run to finish (the
in-run iso emit only fires after the whole sweep). It reads, per finished case,
the incrementally-saved artifacts a CNISP run already wrote:

    runs/<exp>/<run_tag>/step_XX/latents/<casename>.npy        (fitted latent)
    runs/<exp>/<run_tag>/step_XX/pred/<casename>_sub_crop.json (sub-patch placement)
    <aligned_dir>/metadata[_dataset835]/<casename>.json        (canonical-align meta)

and decodes each latent at iso-mm, places it into the full-head iso volume via
``engine.native_mapping.map_iso_results_to_native`` (iso_sp pinned), and writes
the SAME layout the in-run emit / build_corrector_testset expect:

    <out>/native_space_step_XX/<stem>_cnisp_iso_stepXX.nii.gz + manifest.json

It ONLY READS the CNISP run; it does not touch the running process. Run it any
time for whatever sources have finished (or after the whole run).

Usage:
    python nnunet-c/scripts/make_iso_prelabel.py \
        -m orbital_ad_v6_5_gt --experiment thick --run-tag corrector_gt \
        --iso-mm 0.5 [--sources chk_14455 ...]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CNISP_DIR = REPO_ROOT / "orbital_shape_prior_st1"
for _p in (str(CNISP_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from engine.train import create_model                       # noqa: E402
from engine.infer import load_model_checkpoint, _AUTOCAST_DTYPE  # noqa: E402
from engine.native_mapping import map_iso_results_to_native  # noqa: E402

_STEP_RE = re.compile(r"^step_(\d+)$")


def _load_yaml(p):
    with open(p) as f:
        return yaml.safe_load(f) or {}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-p", "--paths", default=str(CNISP_DIR / "configs" / "paths.yaml"))
    ap.add_argument("-t", "--train-config",
                    default=str(CNISP_DIR / "configs" / "train_v6_5_gt.yaml"))
    ap.add_argument("-m", "--model-name", default="orbital_ad_v6_5_gt")
    ap.add_argument("--checkpoint", default="latest", choices=["best", "latest"])
    ap.add_argument("--experiment", default="thick")
    ap.add_argument("--run-tag", default="corrector_gt")
    ap.add_argument("--run-dir", default=None,
                    help="override the CNISP run dir (else output_basedir/<model>/"
                         "runs/<experiment>/<run-tag>)")
    ap.add_argument("--aligned-dir", default=None,
                    help="override aligned_dir (for metadata); else paths.yaml")
    ap.add_argument("--meta-subdirs", default="metadata_dataset835,metadata")
    ap.add_argument("--out-dir", default=None,
                    help="iso output root (default nnunet-c/data/cnisp_pred_test_iso)")
    ap.add_argument("--iso-mm", type=float, default=0.5)
    ap.add_argument("--sources", nargs="*", default=None,
                    help="only build these source_ids (default: all finished)")
    args = ap.parse_args()

    params = {**_load_yaml(args.paths), **_load_yaml(args.train_config)}
    params["model_name"] = args.model_name

    run_dir = (Path(args.run_dir) if args.run_dir else
               Path(params["output_basedir"]) / args.model_name / "runs"
               / args.experiment / args.run_tag)
    aligned_dir = Path(args.aligned_dir or params["aligned_dir"])
    meta_dirs = [aligned_dir / s.strip() for s in args.meta_subdirs.split(",") if s.strip()]
    out_root = Path(args.out_dir) if args.out_dir else (
        REPO_ROOT / "nnunet-c" / "data" / "cnisp_pred_test_iso")
    if not run_dir.is_dir():
        print(f"[make-iso] run dir not found: {run_dir}", file=sys.stderr)
        return 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = Path(params["model_basedir"]) / args.model_name
    model_state, _ = load_model_checkpoint(model_dir, args.checkpoint, verbose=True)
    net = create_model(params, torch.ones(3)).to(device).eval()
    net.load_state_dict(model_state["net"], strict=True)
    print(f"[make-iso] run={run_dir}\n[make-iso] out={out_root}  iso_mm={args.iso_mm}  device={device}")

    def _meta_path(casename: str):
        for d in meta_dirs:
            p = d / f"{casename}.json"
            if p.is_file():
                return p
        return None

    want = set(args.sources) if args.sources else None

    # Gather (step -> [result dicts]) from saved latents + sub_crop sidecars.
    by_step = defaultdict(list)
    n_seen = 0
    for step_dir in sorted(run_dir.glob("step_*")):
        m = _STEP_RE.match(step_dir.name)
        if not m:
            continue
        step = int(m.group(1))
        lat_dir = step_dir / "latents"
        pred_dir = step_dir / "pred"
        if not lat_dir.is_dir():
            continue
        for lf in sorted(lat_dir.glob("*.npy")):
            casename = lf.stem
            mp = _meta_path(casename)
            if mp is None:
                continue
            if want is not None:
                meta_sid = json.load(open(mp)).get("source_id", casename[:-3])
                if str(meta_sid) not in want and casename[:-3] not in want:
                    continue
            sc = pred_dir / f"{casename}_sub_crop.json"
            if not sc.is_file():
                print(f"  skip {casename} step{step:02d}: no sub_crop sidecar")
                continue
            sub = json.load(open(sc))
            latent = np.load(str(lf)).astype(np.float32).reshape(1, -1)
            lat_t = torch.from_numpy(latent).to(device)
            tgt = torch.round(net.image_size.detach().cpu().float() / args.iso_mm).long()
            with torch.no_grad():
                iso_map = net.predict_dense(
                    lat_t, tgt.to(device),
                    torch.full((3,), args.iso_mm, dtype=torch.float32).to(device),
                    autocast_dtype=_AUTOCAST_DTYPE,
                )
            by_step[step].append({
                "casename": casename,
                "pred_class_map_iso": iso_map.numpy().astype(np.int16),
                "sub_crop_lo_vox_dense": sub["sub_crop_lo_vox_dense"],
                "sub_crop_shape_vox_dense": sub["sub_crop_shape_vox_dense"],
            })
            n_seen += 1

    if not by_step:
        print("[make-iso] no (latent + sub_crop) pairs found; nothing to build.",
              file=sys.stderr)
        return 1

    n_masks = 0
    for step, results in sorted(by_step.items()):
        step_out = out_root / f"native_space_step_{step:02d}"
        suffix = f"_cnisp_iso_step{step:02d}"
        paths = map_iso_results_to_native(
            results, meta_dirs[0], step_out, suffix=suffix,
            iso_mm=args.iso_mm,
            meta_path_for_casename=lambda cn: (_meta_path(cn) or meta_dirs[0] / f"{cn}.json"),
        )
        n_masks += len(paths)
        # accumulate by_source_id manifest
        mf_path = step_out / "manifest.json"
        mf = json.load(open(mf_path)) if mf_path.is_file() else {}
        by_sid = mf.get("by_source_id", {})
        for r in results:
            mp = _meta_path(r["casename"])
            if mp is None:
                continue
            meta = json.load(open(mp))
            sid = str(meta["source_id"])
            stem = (Path(meta["original_nifti_path"]).name
                    .replace(".nii.gz", "").replace(".nii", ""))
            by_sid[sid] = f"{stem}{suffix}.nii.gz"
        json.dump({"iso_mm": args.iso_mm, "step_size": step, "suffix": suffix,
                   "run_tag": args.run_tag, "experiment": args.experiment,
                   "by_source_id": by_sid}, open(mf_path, "w"), indent=2)
        print(f"  step{step:02d}: {len(paths)} mask(s) -> {step_out}")

    print(f"[make-iso] done: {n_masks} iso mask(s) from {n_seen} (case,step) latents "
          f"-> {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
