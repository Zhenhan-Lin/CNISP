"""
FOV-completion checkpoint selection (implementation-plan §16-18) + the review's
required modifications:

  * missing and visible macro Dice come from INDEPENDENT validity filters
    (missing_gt_voxels vs visible_gt_voxels) so a fully-visible small structure
    that's absent from the missing region still counts toward the visible
    guardrail (review §5.2);
  * a full-FOV guardrail protects normal (uncropped) performance since full-FOV
    anchors are trained on (review §5.3);
  * guardrails are never silently relaxed: an escalating tolerance sequence with
    an explicit relaxed-flag status is returned (review §5.4);
  * the missing-volume floor is a PHYSICAL volume converted to voxels with the
    exact plan spacing, not a hard-coded voxel count (review §5.5).

Pure scoring over a per-(epoch, crop_type, severity, structure) validation table
produced by the existing whole-volume region-split eval (eval_corrector.py
--region truncated == missing, --region visible; crop_type == "full" rows carry
the full-FOV whole-volume Dice in visible_dice). No nnU-Net / no I/O.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Optional, Sequence

import pandas as pd


@dataclass
class CheckpointScore:
    epoch: int
    missing_macro: float
    visible_macro: float
    worst_condition: float
    full_fov_macro: Optional[float] = None
    smoothed_missing: Optional[float] = None


@dataclass
class SelectionResult:
    checkpoint: CheckpointScore
    visible_guardrail_relaxed: bool
    full_fov_guardrail_relaxed: bool
    applied_visible_tolerance: float
    applied_full_tolerance: Optional[float]


def build_checkpoint_scores(
    metrics: pd.DataFrame,
    min_missing_volume_voxels: int = 32,
    min_visible_volume_voxels: int = 32,
    smooth_window: int = 3,
    full_crop_type: str = "full",
) -> pd.DataFrame:
    """Aggregate a long metrics table into per-epoch scores (review §5.2/5.3, §18).

    Required cols: epoch, crop_type, severity, structure, missing_dice,
    visible_dice, missing_gt_voxels, visible_gt_voxels. Rows with
    crop_type == ``full_crop_type`` are the full-FOV validation condition (their
    visible_dice == whole-volume Dice; they have no missing region).
    Returns one row per epoch: missing_macro, visible_macro, full_fov_macro,
    worst_condition, smoothed_missing.
    """
    trunc = metrics[metrics["crop_type"] != full_crop_type]
    full = metrics[metrics["crop_type"] == full_crop_type]

    # independent validity filters
    missing_valid = trunc[trunc["missing_gt_voxels"] > min_missing_volume_voxels]
    visible_valid = trunc[trunc["visible_gt_voxels"] > min_visible_volume_voxels]
    if missing_valid.empty:
        raise ValueError("no truncated rows exceed the missing-volume floor.")

    def _cond_struct(df, col):
        return (df.groupby(["epoch", "crop_type", "severity", "structure"], as_index=False)
                .agg(val=(col, "mean")))

    miss_cs = _cond_struct(missing_valid, "missing_dice")
    vis_cs = _cond_struct(visible_valid, "visible_dice")

    epoch_missing = miss_cs.groupby("epoch", as_index=False).agg(missing_macro=("val", "mean"))
    epoch_visible = vis_cs.groupby("epoch", as_index=False).agg(visible_macro=("val", "mean"))

    # per-condition missing (mean over structures) -> worst condition per epoch
    cond_macro = (miss_cs.groupby(["epoch", "crop_type", "severity"], as_index=False)
                  .agg(cond_missing=("val", "mean")))
    worst = cond_macro.groupby("epoch", as_index=False).agg(worst_condition=("cond_missing", "min"))

    result = (epoch_missing.merge(epoch_visible, on="epoch", how="left")
              .merge(worst, on="epoch", how="left"))

    # full-FOV macro (optional): mean full-FOV whole Dice over structures
    full_valid = full[full["visible_gt_voxels"] > min_visible_volume_voxels]
    if not full_valid.empty:
        full_cs = _cond_struct(full_valid, "visible_dice")
        epoch_full = full_cs.groupby("epoch", as_index=False).agg(full_fov_macro=("val", "mean"))
        result = result.merge(epoch_full, on="epoch", how="left")
    else:
        result["full_fov_macro"] = pd.NA

    result = result.sort_values("epoch").reset_index(drop=True)
    result["smoothed_missing"] = (result["missing_macro"]
                                  .rolling(window=smooth_window, center=True, min_periods=2).mean())
    return result


def scores_from_frame(result: pd.DataFrame) -> List[CheckpointScore]:
    out = []
    for r in result.itertuples(index=False):
        full = getattr(r, "full_fov_macro", None)
        out.append(CheckpointScore(
            int(r.epoch), float(r.missing_macro), float(r.visible_macro),
            float(r.worst_condition),
            None if (full is None or pd.isna(full)) else float(full),
            None if pd.isna(r.smoothed_missing) else float(r.smoothed_missing)))
    return out


def select_fov_checkpoint(
    scores: Sequence[CheckpointScore],
    missing_tolerance: float = 0.005,
    relaxation_steps: Sequence[float] = (0.005, 0.010, 0.020),
    warn: bool = True,
) -> SelectionResult:
    """Select a checkpoint (review §5.3/5.4). Primary = missing macro Dice; near-best
    set within ``missing_tolerance``; visible + full-FOV guardrails with an
    escalating tolerance; explicit relaxed flags in the result."""
    if not scores:
        raise ValueError("no checkpoint scores.")
    scores = list(scores)

    best_missing = max(s.missing_macro for s in scores)
    best_visible = max(s.visible_macro for s in scores)
    full_vals = [s.full_fov_macro for s in scores if s.full_fov_macro is not None]
    best_full = max(full_vals) if full_vals else None

    near_best = [s for s in scores if s.missing_macro >= best_missing - missing_tolerance]

    def _ranking_key(s: CheckpointScore):
        smoothed = s.smoothed_missing if s.smoothed_missing is not None else s.missing_macro
        return (s.worst_condition, smoothed, s.missing_macro, -s.epoch)   # earlier epoch wins ties

    steps = list(relaxation_steps)
    for tol in steps:
        guarded = [s for s in near_best
                   if s.visible_macro >= best_visible - tol
                   and (best_full is None or s.full_fov_macro is None
                        or s.full_fov_macro >= best_full - tol)]
        if guarded:
            chosen = max(guarded, key=_ranking_key)
            relaxed = tol > steps[0]
            if relaxed and warn:
                warnings.warn(f"[fov-ckpt] guardrails relaxed to {tol:.3f} to admit a "
                              f"candidate (epoch {chosen.epoch}).", stacklevel=2)
            return SelectionResult(chosen, visible_guardrail_relaxed=relaxed,
                                   full_fov_guardrail_relaxed=(relaxed and best_full is not None),
                                   applied_visible_tolerance=tol,
                                   applied_full_tolerance=(None if best_full is None else tol))

    # all guardrails empty even at the loosest tolerance -> explicit fallback
    if warn:
        warnings.warn("[fov-ckpt] visible/full guardrails removed EVERY near-best "
                      "candidate even at the loosest tolerance; falling back to raw "
                      "near-best missing Dice.", stacklevel=2)
    chosen = max(near_best, key=_ranking_key)
    return SelectionResult(chosen, visible_guardrail_relaxed=True,
                           full_fov_guardrail_relaxed=(best_full is not None),
                           applied_visible_tolerance=float("inf"),
                           applied_full_tolerance=(None if best_full is None else float("inf")))


# ── self-test ────────────────────────────────────────────────────────────────
def _selftest() -> int:
    import numpy as np

    from plan_spacing import mm3_to_voxels  # sibling module

    structs = ["ON", "Recti", "Globe", "Fat"]
    conds = [("axial", s) for s in (20, 35, 50)] + [("corner", s) for s in (20, 35, 50)]
    rng = np.random.default_rng(0)
    rows = []
    profile = {
        100: dict(miss=0.60, vis=0.90, full=0.93, worst_lift=0.00),
        125: dict(miss=0.685, vis=0.90, full=0.93, worst_lift=0.05),   # near-best + best worst-cond
        150: dict(miss=0.69, vis=0.90, full=0.93, worst_lift=0.00),    # best raw missing
        175: dict(miss=0.69, vis=0.85, full=0.93, worst_lift=0.00),    # damages VISIBLE
        200: dict(miss=0.688, vis=0.90, full=0.86, worst_lift=0.00),   # damages FULL-FOV
    }
    for ep, p in profile.items():
        for ci, (ct, sev) in enumerate(conds):
            base = p["miss"] - (0.08 if ci == 0 else 0.0) + (p["worst_lift"] if ci == 0 else 0.0)
            for st in structs:
                rows.append(dict(epoch=ep, crop_type=ct, severity=sev, structure=st,
                                 missing_dice=float(np.clip(base + rng.normal(0, 0.003), 0, 1)),
                                 visible_dice=float(np.clip(p["vis"] + rng.normal(0, 0.003), 0, 1)),
                                 missing_gt_voxels=500, visible_gt_voxels=800))
        # full-FOV condition rows (no missing region)
        for st in structs:
            rows.append(dict(epoch=ep, crop_type="full", severity=0, structure=st,
                             missing_dice=float("nan"),
                             visible_dice=float(np.clip(p["full"] + rng.normal(0, 0.003), 0, 1)),
                             missing_gt_voxels=0, visible_gt_voxels=1200))
    df = pd.DataFrame(rows)

    sp = (0.5, 0.4765625, 0.4765625)
    nvox = mm3_to_voxels(3.0, sp)                       # 3 mm^3 physical floor -> voxels
    result = build_checkpoint_scores(df, min_missing_volume_voxels=nvox,
                                     min_visible_volume_voxels=nvox)
    res = select_fov_checkpoint(scores_from_frame(result))
    print(result[["epoch", "missing_macro", "visible_macro", "full_fov_macro",
                  "worst_condition"]].round(4).to_string(index=False))
    print(f"chosen epoch {res.checkpoint.epoch}  "
          f"(vis_relaxed={res.visible_guardrail_relaxed}, full_relaxed={res.full_fov_guardrail_relaxed})")

    # 175 (bad visible) and 200 (bad full-FOV) must both be rejected.
    assert res.checkpoint.epoch not in (175, 200), res.checkpoint.epoch
    # 125 wins: near-best missing + best worst-condition, both guardrails satisfied.
    assert res.checkpoint.epoch == 125, res.checkpoint.epoch
    assert not res.visible_guardrail_relaxed and not res.full_fov_guardrail_relaxed

    # relaxation path: the unique best-missing checkpoint also damages visible, so
    # the near-best set's only member fails the guardrail at every tolerance ->
    # explicit fallback with the relaxed flag set.
    forced = [
        CheckpointScore(epoch=1, missing_macro=0.70, visible_macro=0.80,
                        worst_condition=0.6, full_fov_macro=0.93, smoothed_missing=0.70),
        CheckpointScore(epoch=2, missing_macro=0.69, visible_macro=0.90,
                        worst_condition=0.6, full_fov_macro=0.93, smoothed_missing=0.69),
    ]
    res2 = select_fov_checkpoint(forced, missing_tolerance=0.005, warn=False)
    print(f"forced-relax chosen epoch {res2.checkpoint.epoch}  "
          f"relaxed(vis={res2.visible_guardrail_relaxed}, tol={res2.applied_visible_tolerance})")
    assert res2.visible_guardrail_relaxed, "guardrail should report it was relaxed"
    assert res2.checkpoint.epoch == 1, "fallback keeps the best-missing near-best member"

    print("FOV CHECKPOINT-SELECT SELF-TEST PASSED")
    return 0


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))   # find sibling plan_spacing
    raise SystemExit(_selftest())
