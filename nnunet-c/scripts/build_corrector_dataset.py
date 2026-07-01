#!/usr/bin/env python3
"""Build the corrector's 5-channel nnUNet raw dataset from the data/ tree.

Consumes the self-contained corrector data layout (NOT the work_dir sweep):
    ch0  = data/images/{case}_step{XX}_0000.nii.gz      (degraded CT, pinned)
    ch1..ch4 = control prelabel split into per-class binaries:
        control B -> data/nnunet_pred/{case}_step{XX}.nii.gz   (835 pred, {1,2,3,4})
        control C -> data/cnisp_pred/{case}_step{XX}.nii.gz    (CNISP,   {1,2,3,4})
    label = full-res Dataset835 prediction (manifest gt_candidate_pred), the
            pseudo-GT target (keep=False images have no manual GT), {0..4}.
Everything is resampled to the 835 plan-spacing grid (pothole-2 a-ii) so nnUNet's
preprocess resample is a no-op and the binary channels stay {0,1}.

Each (case_id, step) -> one nnUNet case `corr_{case_id}_step{XX}`. Only samples
whose ch0 + prelabel (+ optionally cnisp, to mirror C) + gt all exist are built,
so a capped CNISP run (e.g. --max-samples 300) yields a matching dataset.

Usage:
    python nnunet-c/scripts/build_corrector_dataset.py --control C
    python nnunet-c/scripts/build_corrector_dataset.py --control B --require-cnisp
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.config import add_repo_to_syspath, load_corrector_config, get_control  # noqa: E402

add_repo_to_syspath(__file__)

import numpy as np  # noqa: E402
import nibabel as nib  # noqa: E402

from engine.convert import convert_case  # noqa: E402  (the SINGLE converter)
from engine.build_dataset import _raw_root, _dataset_dir, _write_dataset_json  # noqa: E402

_DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "corrector.yaml"


def _assemble_one(task):
    """Assemble ONE (case, step) into a 5ch nnUNet case.

    Top-level (picklable) so it can run in a ProcessPoolExecutor worker. Each
    (case, step) is independent -- it only reads its own ct/prelabel/gt and
    writes its own imagesTr/labelsTr files -- so the assembly parallelises
    cleanly across cases. Returns (cid, summary).
    """
    (cid, ct_path, prelabel_path, ref_grid, experiment,
     images_out, gt, labels_out, degraded_marker) = task
    summary = convert_case(
        case_id=cid,
        ct_path=ct_path,
        prelabel_path=prelabel_path,
        ref_grid=ref_grid, experiment=experiment,
        images_dir=images_out, gt_path=gt, labels_dir=labels_out,
        degraded_marker=degraded_marker,
    )
    return cid, summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(_DEFAULT_CONFIG))
    ap.add_argument("--control", required=True, choices=["B", "C", "b", "c"])
    ap.add_argument("--require-cnisp", action="store_true",
                    help="also require data/cnisp_pred to exist for each sample "
                         "(use for control B so it matches C's capped case set).")
    ap.add_argument("--max-samples", type=int, default=0,
                    help="cap the built dataset to the first N (case,step) in "
                         "sorted (case_id, step) order (0=all). When >0, candidacy "
                         "requires BOTH nnunet_pred AND cnisp_pred (a control-"
                         "INDEPENDENT basis) so --control B and --control C select "
                         "the IDENTICAL N samples -- the only difference being which "
                         "prelabel fills ch1..4.")
    ap.add_argument("--raw-root", default=None, help="override $nnUNet_raw")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel worker processes for the 5ch assembly. Each "
                         "(case,step) is independent, so this scales the (slow) "
                         "order-3 CT resampling across cores. Default 1 "
                         "(sequential; unchanged behaviour). Try 8-16 on a big "
                         "box; watch RAM (each worker holds a full-res volume).")
    args = ap.parse_args()

    cfg = load_corrector_config(args.config, caller_file=__file__)
    control = get_control(cfg, args.control)
    if control.get("external"):
        raise RuntimeError(f"control {args.control.upper()} is external (Dataset"
                           f"{control['dataset_id']}); nothing to build.")
    if int(control["n_channels"]) != 5:
        raise RuntimeError("this builder is for the 5-channel controls (B/C).")

    cd = cfg["corrector_data"]
    res = cfg["_resolved"]
    data_root = Path(cd["data_root"])
    data_root = data_root if data_root.is_absolute() else (res["repo_root"] / data_root)
    images_dirname = cd.get("images_dirname", "images")
    images_dir = data_root / images_dirname
    pre_dirname = (cd.get("nnunet_pred_dirname", "nnunet_pred")
                   if control["prelabel_source"] == "nnunet"
                   else cd.get("cnisp_pred_dirname", "cnisp_pred"))
    prelabel_dir = data_root / pre_dirname
    cnisp_dir = data_root / cd.get("cnisp_pred_dirname", "cnisp_pred")
    manifest_path = data_root / "corrector_data_manifest.json"
    if not manifest_path.is_file():
        print(f"[build] {manifest_path} missing -- run build_corrector_data.py first.",
              file=sys.stderr)
        return 2
    manifest = json.load(open(manifest_path))

    raw = _raw_root(args.raw_root)
    ds_dir = _dataset_dir(raw, control)
    images_out = ds_dir / "imagesTr"
    labels_out = ds_dir / "labelsTr"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    print(f"[build] control={args.control.upper()} -> {ds_dir}")
    print(f"[build] ch0={images_dir}  prelabel={prelabel_dir}  grid=GT original (per source)")
    print(f"[build]   (nnUNet resamples this original grid -> iso 0.5 plan at preprocess)")

    # ── Phase 1: deterministic candidate selection ───────────────────
    # Build the ordered list of buildable (case_id, step) FIRST, then optionally
    # cap it, then assemble only the selected ones (so a capped run never does
    # convert_case work for samples it will not keep). When --max-samples is set
    # we use a control-INDEPENDENT candidacy (ct + nnunet_pred + cnisp_pred + gt)
    # so --control B and --control C pick the IDENTICAL first-N set; otherwise we
    # keep the per-control predicate (ct + this control's prelabel [+ cnisp when
    # --require-cnisp] + gt).
    nnunet_dir = data_root / cd.get("nnunet_pred_dirname", "nnunet_pred")
    cap = max(0, int(args.max_samples))
    need_cnisp = bool(args.require_cnisp) or cap > 0
    need_nnunet = cap > 0

    candidates, skipped = [], 0   # candidates: list of (case_id, step, gt_path)
    for case_id, entry in sorted(manifest["cases"].items()):
        gt = entry.get("gt_candidate_pred", "")
        if not gt or not Path(gt).exists():
            skipped += 1
            continue
        for step in sorted(int(s) for s in entry.get("steps", {})):
            sinfo = entry["steps"][str(step)]
            if not sinfo.get("kept"):
                continue
            stem = f"{case_id}_step{step:02d}"
            ct = images_dir / f"{stem}_0000.nii.gz"
            pre = prelabel_dir / f"{stem}.nii.gz"
            if not ct.exists() or not pre.exists():
                skipped += 1
                continue
            if need_cnisp and not (cnisp_dir / f"{stem}.nii.gz").exists():
                skipped += 1
                continue
            if need_nnunet and not (nnunet_dir / f"{stem}.nii.gz").exists():
                skipped += 1
                continue
            candidates.append((case_id, step, Path(gt)))

    # candidates are already globally sorted by (case_id, step): the outer loop
    # iterates cases in sorted order and the inner loop iterates steps ascending.
    if cap > 0:
        if len(candidates) < cap:
            print(f"[build] WARNING: only {len(candidates)} candidate(s) available "
                  f"< --max-samples {cap}; building all of them.", file=sys.stderr)
        candidates = candidates[:cap]

    # ── Phase 2: assemble only the selected (case, step) ─────────────
    # Common grid = the GT's ORIGINAL native grid: label = GT (shared across
    # steps), ch1-4 = prelabel mask (already on this grid), ch0 = degraded
    # upsampled to it. nnUNet then resamples original -> iso 0.5 at preprocess.
    # Build the ref grid per case (lazy: nib.load reads only shape+affine, no
    # data) and the per-(case,step) task list, then assemble sequentially or in
    # a process pool. ``_assemble_one`` is identical work either way, so
    # --workers 1 reproduces the old sequential behaviour byte-for-byte.
    assembled, ref_cache = [], {}
    tasks = []
    for case_id, step, gt in candidates:
        if case_id not in ref_cache:
            gt_img = nib.load(str(gt))
            ref_cache[case_id] = (gt_img.shape[:3], np.asarray(gt_img.affine))
        ref_grid = ref_cache[case_id]
        stem = f"{case_id}_step{step:02d}"
        cid = f"corr_{stem}"
        tasks.append((
            cid,
            images_dir / f"{stem}_0000.nii.gz",
            prelabel_dir / f"{stem}.nii.gz",
            ref_grid, cfg["experiment"],
            images_out, gt, labels_out, f"/{images_dirname}/",
        ))

    workers = max(1, int(args.workers))
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        print(f"[build] assembling {len(tasks)} case(s) with {workers} workers")
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_assemble_one, t) for t in tasks]
            for fut in as_completed(futs):
                cid, summary = fut.result()
                assembled.append(summary)
                print(f"  {cid}: shape={summary['shape']} "
                      f"labels={summary['label_values']}")
    else:
        for t in tasks:
            cid, summary = _assemble_one(t)
            assembled.append(summary)
            print(f"  {cid}: shape={summary['shape']} "
                  f"labels={summary['label_values']}")

    _write_dataset_json(ds_dir, control, cfg, num_training=len(assembled))
    with open(ds_dir / "corrector_build_manifest.json", "w") as f:
        json.dump({"control": args.control.upper(), "n": len(assembled),
                   "cases": assembled}, f, indent=2)
    print(f"[build] wrote {len(assembled)} case(s); skipped {skipped} (missing files).")
    print(f"[build] dataset -> {ds_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
