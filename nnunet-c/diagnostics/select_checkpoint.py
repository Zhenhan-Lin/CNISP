#!/usr/bin/env python3
"""Whole-volume STRATIFIED checkpoint selection for the corrector (Part 1 §1.4).

nnUNet's per-epoch pseudo-Dice is a noisy, biased estimator under the hidden
``step_size`` stratum, so ``checkpoint_best.pth`` (its argmax) selects on sampling
luck. This script instead selects on a *whole-volume, step-stratified* metric:

    for each candidate checkpoint:
        nnUNetv2_predict (sliding window) on a FIXED, step-balanced val set
        -> diagnostics/eval_corrector.py (whole-vol per-structure Dice, by step)
        -> read the companion ``*_by_step.csv`` it always writes
        -> stratified_mean = mean_z[ per-step dice_mean ]
           worst_stratum  = min_z[ per-step dice_mean ]
    pick the checkpoint maximising the chosen criterion.

This is deliberately EXTERNAL (no training-loop surgery). It reuses
``eval_corrector.py`` verbatim (same resample->native-GT, same by-step
aggregation), so selection is scored identically to final evaluation.

Prerequisites (GPU box):
  * The corrector trainer + per-channel resampler are already installed into
    nnunetv2 (run_train.sh / run_corrector_predict.sh do this once).
  * A fixed validation set with balanced step_size exists, produced by
    build_corrector_testset.py: its inputs under ``--images-ts`` (cascade: 1-ch CT
    + a sibling ``prevsegTs`` of CNISP prior masks, auto-detected here; legacy: 5-ch
    image) and its map (``test_cases_map.json``, each case carrying a ``step``
    field) under ``--map``.
  * Periodic checkpoint snapshots exist. nnUNet overwrites checkpoint_latest.pth,
    so either (a) a save-every-N hook on the trainer writes checkpoint_epochXXX.pth,
    or (b) you copy checkpoint_latest.pth aside at intervals. Point --checkpoints
    at whatever snapshots you have (explicit paths or a glob).

Only depends on the Python stdlib (argparse/subprocess/csv/glob/pathlib/statistics).

Usage:
  python nnunet-c/diagnostics/select_checkpoint.py \
      --map      nnunet-c/test_input/PHOTON_CT_CORR_C_cnisp/test_cases_map.json \
      --images-ts nnunet-c/test_input/PHOTON_CT_CORR_C_cnisp/imagesTs \
      --dataset-id 845 --plan-name nnUNetPlansFinetune --configuration 3d_fullres \
      --trainer nnUNetTrainer_OrbitalCascade --fold 0 \
      --checkpoints "$nnUNet_results/Dataset845_.../fold_0/checkpoint_epoch*.pth" \
      --work-dir nnunet-c/predictions/_select_C --criterion stratified_mean \
      --out-csv nnunet-c/predictions/select_C.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import statistics
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_EVAL = _HERE / "eval_corrector.py"


def _resolve_checkpoints(spec: str) -> list[Path]:
    """--checkpoints is a comma list of paths and/or globs; expand + sort."""
    out: list[Path] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        hits = [Path(p) for p in glob.glob(tok)] or ([Path(tok)] if Path(tok).exists() else [])
        out.extend(hits)
    # de-dup, stable sort by name (checkpoint_epoch0050 < 0100 < ... when zero-padded)
    seen, uniq = set(), []
    for p in sorted(out, key=lambda p: p.name):
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return uniq


def _predict(images_ts: Path, out_dir: Path, args, chk: Path,
             prev_stage: str | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "nnUNetv2_predict",
        "-i", str(images_ts), "-o", str(out_dir),
        "-d", str(args.dataset_id), "-c", args.configuration,
        "-tr", args.trainer, "-p", args.plan_name, "-f", str(args.fold),
        "-chk", chk.name,
    ]
    # Cascade (Route A): the config has previous_stage=cnisp_prior, so nnUNet
    # asserts a prev-stage seg folder is supplied. Pass the prevsegTs dir (the
    # per-case CNISP prior masks) exactly as run_corrector_predict.sh does.
    if prev_stage:
        cmd += ["-prev_stage_predictions", str(prev_stage)]
    print(f"    [predict] -chk {chk.name} -> {out_dir}"
          f"{'  (+prev_stage)' if prev_stage else ''}")
    subprocess.run(cmd, check=True)


def _eval(map_json: Path, pred_dir: Path, eval_csv: Path) -> Path:
    """Run eval_corrector.py; return the path to the companion *_by_step.csv."""
    cmd = [sys.executable, str(_EVAL),
           "--map", str(map_json), "--pred-dir", str(pred_dir),
           "--out-csv", str(eval_csv)]
    subprocess.run(cmd, check=True)
    return eval_csv.with_name(eval_csv.stem + "_by_step.csv")


def _read_by_step(by_step_csv: Path) -> dict[str, float]:
    """step -> dice_mean, from eval_corrector's by-step CSV."""
    per_step: dict[str, float] = {}
    with open(by_step_csv, newline="") as f:
        for row in csv.DictReader(f):
            per_step[str(row["step"])] = float(row["dice_mean"])
    return per_step


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", required=True, help="fixed val set test_cases_map.json (step per case)")
    ap.add_argument("--images-ts", required=True,
                    help="the val set's imagesTs dir (cascade: 1-ch CT; legacy: 5-ch)")
    ap.add_argument("--prev-stage-predictions", default=None,
                    help="cascade (Route A): folder of {caseid}.nii.gz CNISP prior masks "
                         "passed to nnUNetv2_predict -prev_stage_predictions. DEFAULT: "
                         "auto-detect the sibling 'prevsegTs' next to --images-ts (present "
                         "for a cascade testset); omit/absent -> legacy stacked path.")
    ap.add_argument("--checkpoints", required=True,
                    help="comma list of .pth paths and/or globs (periodic snapshots)")
    ap.add_argument("--dataset-id", required=True)
    ap.add_argument("--plan-name", default="nnUNetPlansFinetune")
    ap.add_argument("--configuration", default="3d_fullres")
    ap.add_argument("--trainer", default="nnUNetTrainer_OrbitalCascade")
    ap.add_argument("--fold", default="0")
    ap.add_argument("--work-dir", required=True, help="scratch dir for per-ckpt preds + eval CSVs")
    ap.add_argument("--criterion", choices=["stratified_mean", "worst_stratum"],
                    default="stratified_mean")
    ap.add_argument("--out-csv", default=None, help="summary CSV (default: <work-dir>/select_summary.csv)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="reuse a checkpoint's prediction dir if it already exists")
    args = ap.parse_args()

    map_json = Path(args.map)
    images_ts = Path(args.images_ts)
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    if not _EVAL.is_file():
        print(f"[select] eval script not found: {_EVAL}", file=sys.stderr)
        return 2
    ckpts = _resolve_checkpoints(args.checkpoints)
    if not ckpts:
        print(f"[select] no checkpoints matched: {args.checkpoints!r}", file=sys.stderr)
        return 2

    # Cascade prev-stage folder: explicit --prev-stage-predictions wins; else
    # auto-detect the sibling prevsegTs the cascade testset build writes next to
    # imagesTs. None -> legacy stacked layout (prior baked into the 5-ch image).
    prev_stage = args.prev_stage_predictions
    if prev_stage is None:
        cand = images_ts.parent / "prevsegTs"
        prev_stage = str(cand) if cand.is_dir() else None
    if prev_stage and not Path(prev_stage).is_dir():
        print(f"[select] --prev-stage-predictions not a dir: {prev_stage}", file=sys.stderr)
        return 2
    print(f"[select] {len(ckpts)} checkpoint(s); criterion={args.criterion}; "
          f"prev_stage={'(cascade) ' + prev_stage if prev_stage else '(none/stacked)'}")

    rows = []                       # per-checkpoint summary
    all_steps: set[str] = set()
    for chk in ckpts:
        stem = chk.stem
        pred_dir = work / f"pred_{stem}"
        if not (args.skip_existing and pred_dir.is_dir() and any(pred_dir.glob("*.nii.gz"))):
            _predict(images_ts, pred_dir, args, chk, prev_stage)
        by_step = _eval(map_json, pred_dir, work / f"eval_{stem}.csv")
        per_step = _read_by_step(by_step)
        if not per_step:
            print(f"    [warn] no by-step rows for {stem}; skipping")
            continue
        all_steps |= set(per_step)
        strat_mean = statistics.fmean(per_step.values())
        worst = min(per_step.values())
        score = strat_mean if args.criterion == "stratified_mean" else worst
        rows.append({"checkpoint": stem, "per_step": per_step,
                     "stratified_mean": strat_mean, "worst_stratum": worst,
                     "score": score, "path": str(chk)})
        detail = " ".join(f"s{k}={v:.3f}" for k, v in sorted(per_step.items()))
        print(f"  {stem}: {detail}  strat_mean={strat_mean:.4f} worst={worst:.4f}")

    if not rows:
        print("[select] nothing scored.", file=sys.stderr)
        return 1

    best = max(rows, key=lambda r: r["score"])
    print("-" * 64)
    print(f"[select] BEST ({args.criterion}={best['score']:.4f}): {best['checkpoint']}")
    print(f"         {best['path']}")

    out_csv = Path(args.out_csv) if args.out_csv else work / "select_summary.csv"
    step_cols = [f"dice_step{s}" for s in sorted(all_steps, key=lambda x: (len(x), x))]
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["checkpoint"] + step_cols + ["stratified_mean", "worst_stratum", "is_best", "path"])
        for r in sorted(rows, key=lambda r: r["checkpoint"]):
            steps = [f"{r['per_step'].get(s.replace('dice_step',''), float('nan')):.5f}" for s in step_cols]
            w.writerow([r["checkpoint"]] + steps
                       + [f"{r['stratified_mean']:.5f}", f"{r['worst_stratum']:.5f}",
                          int(r is best), r["path"]])
    print(f"[select] summary -> {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
