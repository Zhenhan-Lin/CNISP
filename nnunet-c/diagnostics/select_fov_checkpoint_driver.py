#!/usr/bin/env python3
"""
FOV-completion checkpoint-selection driver (implementation-plan §15-18).

Two layers:
  * SELECTION (self-testable here): read a long validation metrics table, build
    per-epoch scores and pick the checkpoint via lib/fov_checkpoint_select
    (missing macro Dice primary; visible + full-FOV guardrails; worst-condition
    tie-break; full-FOV completeness enforced).
  * SWEEP (masi-55): for each periodic snapshot, run whole-volume nnUNetv2_predict
    on the FIXED 7-condition val set, eval_corrector --region {truncated,visible}
    + the full-FOV cases, and assemble the table. This mirrors the existing
    diagnostics/select_checkpoint.py orchestration; only the FOV aggregation +
    scoring is new.

Long metrics table columns (one row per epoch x case x structure):
    epoch, case_id, crop_type, severity, structure,
    missing_dice, visible_dice, missing_gt_voxels, visible_gt_voxels
crop_type == "full" rows are the full-FOV validation condition (visible_dice ==
whole-volume Dice; no missing region).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # nnunet-c
from lib.fov_checkpoint_select import (build_checkpoint_scores,          # noqa: E402
                                       scores_from_frame, select_fov_checkpoint)
from lib.plan_spacing import mm3_to_voxels, resolve_target_spacing_from_plan  # noqa: E402


def select_from_table(metrics: pd.DataFrame, min_missing_mm3: float = 0.0,
                      min_visible_mm3: float = 0.0, plans_file: str = None,
                      configuration: str = "3d_fullres", final: bool = False,
                      expected_structures=None):
    """Build per-epoch scores and select. Physical-volume floors are converted to
    voxels with the exact plan spacing (review §5.5) when a plan is given.

    ``final`` (the paper selection, review §12 + brief-recs items 5/6): a plan file,
    BOTH physical floors AND the expected structure list are REQUIRED (no silent
    32-voxel floor, absolute coverage enforced). The guardrail is FIXED at a single
    0.005 step and strict — visible/full Dice may drop at most 0.5pp below their best,
    else the selector RAISES rather than falling back to raw missing Dice."""
    if final and not (plans_file and min_missing_mm3 > 0 and min_visible_mm3 > 0
                      and expected_structures):
        raise SystemExit("--final requires --plans-file, --min-missing-mm3, "
                         "--min-visible-mm3 and --expect-structures (no silent 32-voxel "
                         "floor and no unverified coverage for the paper run).")
    if plans_file and (min_missing_mm3 > 0 or min_visible_mm3 > 0):
        sp = resolve_target_spacing_from_plan(plans_file, configuration)
        min_missing_vox = mm3_to_voxels(min_missing_mm3, sp) if min_missing_mm3 > 0 else 32
        min_visible_vox = mm3_to_voxels(min_visible_mm3, sp) if min_visible_mm3 > 0 else 32
    else:
        min_missing_vox = min_visible_vox = 32
    result = build_checkpoint_scores(metrics, min_missing_volume_voxels=min_missing_vox,
                                     min_visible_volume_voxels=min_visible_vox,
                                     expected_structures=expected_structures)
    # brief-recs item 5: final selection pins the guardrail to a single 0.005 step.
    steps = (0.005,) if final else (0.005, 0.010, 0.020)
    selection = select_fov_checkpoint(scores_from_frame(result), relaxation_steps=steps,
                                      strict_guardrail=final)
    return result, selection


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics-csv", help="pre-built long metrics table")
    ap.add_argument("--out-scores-csv", default=None)
    ap.add_argument("--plans-file", default=None)
    ap.add_argument("--configuration", default="3d_fullres")
    ap.add_argument("--min-missing-mm3", type=float, default=0.0)
    ap.add_argument("--min-visible-mm3", type=float, default=0.0)
    ap.add_argument("--expect-structures", default=None,
                    help="comma-separated structure names the eval MUST cover every epoch "
                         "(brief-recs item 6), e.g. ON,Recti,Globe,Fat. Required with --final.")
    ap.add_argument("--final", action="store_true",
                    help="paper selection: require plan + both physical floors + "
                         "--expect-structures; fixed 0.005 strict guardrail (review §12, "
                         "brief-recs 5/6); no silent fallback.")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return _selftest()
    if not args.metrics_csv:
        ap.error("--metrics-csv is required (the sweep that builds it is "
                 "run_fov_completion_sweep.sh; see the module docstring). Use --self-test "
                 "to exercise the selector.")

    expect = ([s.strip() for s in args.expect_structures.split(",") if s.strip()]
              if args.expect_structures else None)
    metrics = pd.read_csv(args.metrics_csv)
    result, selection = select_from_table(
        metrics, args.min_missing_mm3, args.min_visible_mm3, args.plans_file,
        args.configuration, final=args.final, expected_structures=expect)
    show = [c for c in ["epoch", "missing_macro", "visible_macro", "full_fov_macro",
                        "missing_precision_macro", "worst_condition"] if c in result.columns]
    print(result[show].round(4).to_string(index=False))
    c = selection.checkpoint
    print(f"\nCHOSEN epoch={c.epoch}  missing={c.missing_macro:.4f} visible={c.visible_macro:.4f} "
          f"full={c.full_fov_macro} precision={c.missing_precision_macro} worst={c.worst_condition:.4f}")
    print(f"  guardrail relaxed: visible={selection.visible_guardrail_relaxed} "
          f"full={selection.full_fov_guardrail_relaxed} "
          f"hallucination={selection.hallucination_guardrail_relaxed} "
          f"(vis_tol={selection.applied_visible_tolerance}, full_tol={selection.applied_full_tolerance}, "
          f"prec_tol={selection.applied_precision_tolerance})")
    if args.out_scores_csv:
        result.to_csv(args.out_scores_csv, index=False)
        print(f"  wrote {args.out_scores_csv}")
    return 0


def _selftest() -> int:
    import numpy as np
    import tempfile

    structs = ["ON", "Recti", "Globe", "Fat"]
    conds = [("axial", s) for s in (20, 35, 50)] + [("corner", s) for s in (20, 35, 50)]
    rng = np.random.default_rng(0)
    rows = []
    profile = {100: (0.60, 0.90, 0.93), 125: (0.685, 0.90, 0.93),
               150: (0.69, 0.90, 0.93), 175: (0.69, 0.85, 0.93), 200: (0.688, 0.90, 0.86)}
    for ep, (miss, vis, full) in profile.items():
        for ci, (ct, sev) in enumerate(conds):
            base = miss - (0.08 if ci == 0 else 0.0) + (0.05 if (ci == 0 and ep == 125) else 0.0)
            for st in structs:
                rows.append(dict(epoch=ep, case_id=f"v{ci}", crop_type=ct, severity=sev, structure=st,
                                 missing_dice=float(np.clip(base + rng.normal(0, 0.003), 0, 1)),
                                 visible_dice=float(np.clip(vis + rng.normal(0, 0.003), 0, 1)),
                                 missing_gt_voxels=500, visible_gt_voxels=800))
        for st in structs:
            rows.append(dict(epoch=ep, case_id="vfull", crop_type="full", severity=0, structure=st,
                             missing_dice=float("nan"),
                             visible_dice=float(np.clip(full + rng.normal(0, 0.003), 0, 1)),
                             missing_gt_voxels=0, visible_gt_voxels=1200))
    df = pd.DataFrame(rows)
    with tempfile.TemporaryDirectory() as d:
        csv = Path(d) / "metrics.csv"
        df.to_csv(csv, index=False)
        metrics = pd.read_csv(csv)
        result, selection = select_from_table(metrics)
    print(result[["epoch", "missing_macro", "visible_macro", "full_fov_macro"]].round(4).to_string(index=False))
    print("chosen epoch:", selection.checkpoint.epoch)
    assert selection.checkpoint.epoch == 125, selection.checkpoint.epoch   # 175/200 guarded out
    print("FOV CHECKPOINT-DRIVER SELF-TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
