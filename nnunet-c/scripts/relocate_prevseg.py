#!/usr/bin/env python3
"""Relocate the parallel prior dataset's preprocessed segs into the main dataset's
cascade ``seg_prev`` slot (Route A, Phase 0.5 final step).

Pipeline recap (see build_corrector_dataset.py --layout cascade):
  * MAIN dataset  (control C, e.g. 845): 1-ch CT + GT  -> preprocessed to
      ``<pp>/Dataset845_.../<plan>_<cfg>/{id}.b2nd`` (+ ``{id}_seg.b2nd`` = GT)
  * PRIOR dataset (parallel, e.g. 846): same CT + the CNISP prior AS the label ->
      preprocessed to ``<pp>/Dataset846_.../<plan>_<cfg>/{id}_seg.b2nd`` = the prior
      on the SAME voxel grid (identical CT ⇒ identical nonzero-crop + resample).

nnUNet's cascade loads ``seg_prev`` from ``folder_with_segs_from_previous_stage``
as ``{id}.b2nd`` (a plain blosc2 seg, produced identically to a ``_seg.b2nd``;
confirmed by inspect_cascade_route.py: ``load_case`` reads it via ``blosc2.open``
and ``resample_and_save`` writes next-stage segs with ``nnUNetDatasetBlosc2.save_seg``).
So this script simply **copies (or moves) each prior ``{id}_seg.b2nd`` to
``<dest>/{id}.b2nd``**, where ``<dest>`` is read from a live ``nnUNetTrainer``
instance (ground truth) rather than guessed.

Run AFTER preprocessing BOTH datasets with the finetune plan and BEFORE training.
Idempotent with --overwrite. Depends on the stdlib + an importable ``nnunetv2``
(for the dest-folder read; a computed fallback covers the rare import failure).

Usage:
  python nnunet-c/scripts/relocate_prevseg.py --control C \
      --plan-name nnUNetPlansFinetune            # copy (safe, default)
  python nnunet-c/scripts/relocate_prevseg.py --control C --move --overwrite
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def _resolve(config, control, plan_name, configuration, prior_id, prior_name):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from lib.config import load_corrector_config, get_control  # lazy

    cfg = load_corrector_config(config, caller_file=__file__)
    ctrl = get_control(cfg, control)
    configuration = configuration or cfg["configuration"]
    pp = os.environ.get("nnUNet_preprocessed")
    if not pp:
        raise RuntimeError("$nnUNet_preprocessed unset (need it on the GPU box).")

    main_name = f"Dataset{int(ctrl['dataset_id']):03d}_{ctrl['dataset_name']}"
    pid = int(prior_id) if prior_id is not None else int(ctrl["dataset_id"]) + 1
    pname = prior_name or f"{ctrl['dataset_name']}_prior"
    prior_dsname = f"Dataset{pid:03d}_{pname}"

    main_dir = Path(pp) / main_name
    prior_dir = Path(pp) / prior_dsname
    main_data = main_dir / f"{plan_name}_{configuration}"
    prior_data = prior_dir / f"{plan_name}_{configuration}"
    return cfg, ctrl, configuration, plan_name, main_dir, main_data, prior_data


def _dest_folder(main_dir: Path, plan_name: str, configuration: str, trainer: str):
    """Return (dest_folder, prev, how) for the cascade seg_prev.

    nnUNet builds ``folder_with_segs_from_previous_stage`` as
        nnUNet_results/<dataset>/<TRAINER>__<plans_name>__<previous_stage>/
            predicted_next_stage/<configuration>
    -- crucially keyed by the TRAINING trainer's class name (not the base class).
    We compute it with the ACTUAL training trainer (default nnUNetTrainer_Orbital-
    Cascade), so relocate, the gate, and training all agree. (Instantiating a base
    ``nnUNetTrainer`` here gave the WRONG class name -> a folder training never reads.)
    """
    import json
    plan_json = main_dir / f"{plan_name}.json"
    if not plan_json.is_file():
        raise FileNotFoundError(f"missing {plan_json} (preprocess the main dataset first)")
    plans = json.load(open(plan_json))
    prev = plans.get("configurations", {}).get(configuration, {}).get("previous_stage")
    if not prev:
        raise RuntimeError(
            f"plan {plan_json} has no configurations.{configuration}.previous_stage "
            f"-- rebuild it with build_finetune_plan.py --cascade.")
    plans_name = plans.get("plans_name", plan_name)
    results = os.environ.get("nnUNet_results")
    if not results:
        raise RuntimeError("$nnUNet_results unset (needed to locate the seg_prev folder).")
    dest = (Path(results) / main_dir.name
            / f"{trainer}__{plans_name}__{prev}" / "predicted_next_stage" / configuration)
    return dest, prev, "results"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config",
                    default=str(Path(__file__).resolve().parents[1] / "configs" / "corrector.yaml"))
    ap.add_argument("--control", required=True, choices=["B", "C", "b", "c"])
    ap.add_argument("--plan-name", default="nnUNetPlansFinetune")
    ap.add_argument("--configuration", default=None)
    ap.add_argument("--trainer", default="nnUNetTrainer_OrbitalCascade",
                    help="the trainer the model will be TRAINED with (default "
                         "%(default)s). The seg_prev folder is keyed by this class "
                         "name, so it MUST match run_train.sh's CORRECTOR_TRAINER.")
    ap.add_argument("--fold", default=0, type=int,
                    help="(unused; the seg_prev folder is fold-independent).")
    ap.add_argument("--prior-dataset-id", type=int, default=None,
                    help="parallel prior dataset id (default: control id + 1).")
    ap.add_argument("--prior-dataset-name", default=None,
                    help="parallel prior dataset name (default: <name>_prior).")
    ap.add_argument("--move", action="store_true",
                    help="move instead of copy (frees the prior dataset's disk).")
    ap.add_argument("--overwrite", action="store_true",
                    help="overwrite existing {id}.b2nd in the dest (idempotent re-run).")
    ap.add_argument("--dest", default=None,
                    help="override the seg_prev dest folder (else read from trainer).")
    args = ap.parse_args()

    cfg, ctrl, configuration, plan_name, main_dir, main_data, prior_data = _resolve(
        args.config, args.control, args.plan_name, args.configuration,
        args.prior_dataset_id, args.prior_dataset_name)

    if not prior_data.is_dir():
        print(f"[relocate] prior preprocessed dir not found: {prior_data}\n"
              f"           preprocess the prior dataset with the finetune plan first.",
              file=sys.stderr)
        return 2
    if not main_data.is_dir():
        print(f"[relocate] main preprocessed dir not found: {main_data}", file=sys.stderr)
        return 2

    if args.dest:
        dest, prev, how = Path(args.dest), "(override)", "override"
    else:
        dest, prev, how = _dest_folder(main_dir, plan_name, configuration, args.trainer)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[relocate] previous_stage={prev}  dest ({how}) = {dest}")
    print(f"[relocate] prior segs from = {prior_data}")

    # main ids (each must get a seg_prev); prior segs available.
    main_ids = sorted(p.name[:-4] for p in main_data.glob("*.pkl"))
    prior_segs = {p.name[: -len("_seg.b2nd")]: p
                  for p in prior_data.glob("*_seg.b2nd")}
    if not main_ids:
        print(f"[relocate] no main cases (*.pkl) in {main_data}", file=sys.stderr)
        return 2
    print(f"[relocate] main cases={len(main_ids)}  prior segs={len(prior_segs)}")

    missing = [i for i in main_ids if i not in prior_segs]
    if missing:
        print(f"[relocate] ERROR: {len(missing)} main case(s) have NO prior seg, e.g. "
              f"{missing[:5]}. The prior dataset must cover every main case "
              f"(same build). Aborting.", file=sys.stderr)
        return 1

    # The prior dataset stores its label as a preprocessed SEG -> shape (1, D, H, W)
    # (leading channel dim). But nnUNet's cascade dataloader expects seg_prev WITHOUT
    # the channel dim -> (D, H, W): it does crop_and_pad_nd(seg_prev, bbox, -1)[None]
    # to re-add the channel before vstacking onto the GT seg. A raw copy of the (1,*)
    # file makes that [None] produce a 5-D array -> vstack ValueError at train time.
    # So we LOAD each prior seg, squeeze the channel dim, and re-write it as (D,H,W).
    import numpy as np  # noqa: E402  (heavy; only needed here)
    import blosc2  # noqa: E402
    n_done, n_skip = 0, 0
    for i in main_ids:
        src = prior_segs[i]
        out = dest / f"{i}.b2nd"
        if out.exists() and not args.overwrite:
            n_skip += 1
            continue
        arr = np.asarray(blosc2.open(str(src), mode="r")[:])
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]                       # (1, D, H, W) -> (D, H, W)
        elif arr.ndim != 3:
            raise RuntimeError(
                f"{src}: seg_prev has unexpected shape {arr.shape}; want (D,H,W) or (1,D,H,W).")
        if out.exists():
            out.unlink()
        blosc2.asarray(np.ascontiguousarray(arr.astype(np.uint8)),
                       urlpath=str(out), mode="w")
        if args.move:
            Path(src).unlink()
        n_done += 1

    verb = "moved" if args.move else "wrote"
    print(f"[relocate] {verb} {n_done} seg_prev file(s) as (D,H,W); skipped {n_skip} "
          f"existing (use --overwrite to replace).")
    # sanity: every main id now has a seg_prev on disk
    have = sum(1 for i in main_ids if (dest / f"{i}.b2nd").is_file())
    print(f"[relocate] dest now has seg_prev for {have}/{len(main_ids)} main cases.")
    if have != len(main_ids):
        print("[relocate] WARNING: coverage incomplete -- training will fail on the "
              "missing cases (load_case can't find their seg_prev).", file=sys.stderr)
        return 1
    print(f"[relocate] OK -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
