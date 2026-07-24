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
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


def _epoch_coverage(df: pd.DataFrame) -> Dict[int, frozenset]:
    """epoch -> the set of (crop_type, severity, structure) cells present (validity-
    filtered). Used to prove every epoch is averaged over the SAME cells."""
    cov: Dict[int, frozenset] = {}
    for ep, g in df.groupby("epoch"):
        cov[int(ep)] = frozenset(
            (str(ct), int(sv), str(st))
            for ct, sv, st in g[["crop_type", "severity", "structure"]].to_numpy().tolist())
    return cov


def _check_finite(df: pd.DataFrame, col: str, what: str) -> None:
    v = df[col].to_numpy(dtype=float)
    if v.size and not np.isfinite(v).all():
        bad = sorted(set(df.loc[~np.isfinite(df[col].to_numpy(dtype=float)), "epoch"].tolist()))
        raise ValueError(f"{what}: non-finite '{col}' at epoch(s) {bad}.")


def _check_coverage(df: pd.DataFrame, what: str, strict: bool) -> None:
    """Every epoch must cover the same cells, else a macro average silently spans a
    different denominator per epoch (review §12). Raise (strict) or warn."""
    cov = _epoch_coverage(df)
    if len(cov) <= 1:
        return
    union = frozenset().union(*cov.values())
    incomplete = {ep: sorted(union - cells) for ep, cells in cov.items() if cells != union}
    if not incomplete:
        return
    ex_ep = next(iter(incomplete))
    msg = (f"{what}: inconsistent per-epoch coverage — e.g. epoch {ex_ep} is missing "
           f"{len(incomplete[ex_ep])} cell(s) like {incomplete[ex_ep][:3]}. A missing row "
           f"must not silently change an epoch's macro denominator.")
    if strict:
        raise ValueError(msg)
    warnings.warn("[fov-ckpt] " + msg, stacklevel=2)


def _subject_unit(df: pd.DataFrame) -> pd.Series:
    """The equal-weight statistical unit (revised-plan §10): subject_id if present,
    else case_id, else a single unit (reduces to a flat cell mean for legacy tables)."""
    if "subject_id" in df.columns:
        return df["subject_id"].astype(str)
    if "case_id" in df.columns:
        return df["case_id"].astype(str)
    return pd.Series(["_all"] * len(df), index=df.index)


def _subject_major(df: pd.DataFrame, value_col: str, out_name: str) -> pd.DataFrame:
    """Per (epoch, subject) mean over the subject's cells, then per epoch mean over
    subjects. Equal weight per subject regardless of how many valid cells it has."""
    d = df.assign(_subj=_subject_unit(df))
    per_subj = d.groupby(["epoch", "_subj"], as_index=False).agg(_v=(value_col, "mean"))
    return per_subj.groupby("epoch", as_index=False).agg(**{out_name: ("_v", "mean")})


def _worst_condition(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Worst (min) per-condition subject-major missing Dice per epoch (tie-break)."""
    d = df.assign(_subj=_subject_unit(df))
    per = (d.groupby(["epoch", "crop_type", "severity", "_subj"], as_index=False)
           .agg(_v=(value_col, "mean")))
    cond = (per.groupby(["epoch", "crop_type", "severity"], as_index=False)
            .agg(_cv=("_v", "mean")))
    return cond.groupby("epoch", as_index=False).agg(worst_condition=("_cv", "min"))


def _check_absolute_coverage(metrics: pd.DataFrame, expected_structures: Sequence[str],
                             crop_types: Sequence[str], severities: Sequence[int],
                             full_crop_type: str, strict: bool) -> None:
    """brief-recs item 6: check the RAW table against the FIXED expected grid, so a
    condition×structure that is absent in EVERY epoch (which the per-epoch consistency
    check cannot see — it only compares epochs to each other) is still caught.

    Expected per epoch: crop_types × severities × structures (truncated) PLUS
    (full_crop_type, structure) for each structure. Runs on the raw metrics BEFORE
    validity filtering — the sweep must EMIT a row for every cell (Dice may be NaN /
    empty region); validity filtering happens afterwards."""
    structs = [str(s) for s in expected_structures]
    expected = {(str(ct), int(sv), st) for ct in crop_types for sv in severities for st in structs}
    expected |= {(str(full_crop_type), 0, st) for st in structs}  # full-FOV severity encoded 0
    present_by_epoch: Dict[int, set] = {}
    for ep, g in metrics.groupby("epoch"):
        present_by_epoch[int(ep)] = {
            (str(ct), int(sv), str(st))
            for ct, sv, st in g[["crop_type", "severity", "structure"]].to_numpy().tolist()}
    problems = {ep: sorted(expected - present) for ep, present in present_by_epoch.items()
                if expected - present}
    if not problems:
        return
    ex_ep = next(iter(problems))
    msg = (f"absolute coverage: epoch {ex_ep} is missing {len(problems[ex_ep])} expected "
           f"cell(s) like {problems[ex_ep][:3]} (expected {len(expected)} = "
           f"{len(crop_types)}crop×{len(severities)}sev×{len(structs)}struct + full×{len(structs)}). "
           f"The eval sweep must emit a row for every condition×structure.")
    if strict:
        raise ValueError(msg)
    warnings.warn("[fov-ckpt] " + msg, stacklevel=2)


@dataclass
class CheckpointScore:
    epoch: int
    missing_macro: float
    visible_macro: float
    worst_condition: float
    full_fov_macro: Optional[float] = None
    smoothed_missing: Optional[float] = None
    missing_precision_macro: Optional[float] = None   # hallucination guardrail metric


@dataclass
class SelectionResult:
    checkpoint: CheckpointScore
    visible_guardrail_relaxed: bool
    full_fov_guardrail_relaxed: bool
    applied_visible_tolerance: float
    applied_full_tolerance: Optional[float]
    hallucination_guardrail_relaxed: bool = False
    applied_precision_tolerance: Optional[float] = None


def build_checkpoint_scores(
    metrics: pd.DataFrame,
    min_missing_volume_voxels: int = 32,
    min_visible_volume_voxels: int = 32,
    smooth_window: int = 3,
    full_crop_type: str = "full",
    require_consistent_coverage: bool = True,
    expected_structures: Optional[Sequence[str]] = None,
    expected_crop_types: Sequence[str] = ("axial", "corner"),
    expected_severities: Sequence[int] = (20, 35, 50),
) -> pd.DataFrame:
    """Aggregate a long metrics table into per-epoch scores (review §5.2/5.3, §18).

    Required cols: epoch, crop_type, severity, structure, missing_dice,
    visible_dice, missing_gt_voxels, visible_gt_voxels. Rows with
    crop_type == ``full_crop_type`` are the full-FOV validation condition (their
    visible_dice == whole-volume Dice; they have no missing region).
    Returns one row per epoch: missing_macro, visible_macro, full_fov_macro,
    worst_condition, smoothed_missing.
    """
    # brief-recs item 6: absolute coverage against the fixed grid, on the RAW table.
    if expected_structures is not None:
        _check_absolute_coverage(metrics, expected_structures, expected_crop_types,
                                 expected_severities, full_crop_type,
                                 strict=require_consistent_coverage)

    trunc = metrics[metrics["crop_type"] != full_crop_type]
    full = metrics[metrics["crop_type"] == full_crop_type]

    # independent validity filters
    missing_valid = trunc[trunc["missing_gt_voxels"] > min_missing_volume_voxels]
    visible_valid = trunc[trunc["visible_gt_voxels"] > min_visible_volume_voxels]
    if missing_valid.empty:
        raise ValueError("no truncated rows exceed the missing-volume floor.")
    if visible_valid.empty:                                   # review §12
        raise ValueError("no truncated rows exceed the visible-volume floor.")
    # finite-metric + per-epoch coverage guards (review §12): a NaN Dice or a
    # dropped cell must not silently inflate/deflate an epoch's macro average.
    _check_finite(missing_valid, "missing_dice", "missing-valid rows")
    _check_finite(visible_valid, "visible_dice", "visible-valid rows")
    _check_coverage(missing_valid, "missing-valid", require_consistent_coverage)
    _check_coverage(visible_valid, "visible-valid", require_consistent_coverage)

    # revised-plan §10 / P1-8: the statistical unit is the SUBJECT, not the FOV case.
    # macro = mean over subjects of (each subject's mean over its cells), so a subject
    # with more valid cells does not get more weight.
    epoch_missing = _subject_major(missing_valid, "missing_dice", "missing_macro")
    epoch_visible = _subject_major(visible_valid, "visible_dice", "visible_macro")
    worst = _worst_condition(missing_valid, "missing_dice")

    result = (epoch_missing.merge(epoch_visible, on="epoch", how="left")
              .merge(worst, on="epoch", how="left"))

    # revised-plan §6.5 / P1-5: hallucination guardrail metric = subject-major
    # missing-region PRECISION (finite only). Low precision == anatomy invented where
    # the FOV did not acquire it; higher is better. Reported alongside mean FP volume.
    if "missing_precision" in missing_valid.columns:
        prec = missing_valid[np.isfinite(missing_valid["missing_precision"].astype(float))]
        if not prec.empty:
            result = result.merge(_subject_major(prec, "missing_precision",
                                                 "missing_precision_macro"), on="epoch", how="left")
    if "missing_precision_macro" not in result.columns:
        result["missing_precision_macro"] = pd.NA
    if "missing_fp_voxels" in missing_valid.columns:
        fp = (missing_valid.groupby("epoch", as_index=False)
              .agg(missing_fp_voxels_mean=("missing_fp_voxels", "mean")))
        result = result.merge(fp, on="epoch", how="left")

    # full-FOV macro (optional): subject-major full-FOV whole Dice
    full_valid = full[full["visible_gt_voxels"] > min_visible_volume_voxels]
    if not full_valid.empty:
        _check_finite(full_valid, "visible_dice", "full-FOV rows")
        _check_coverage(full_valid, "full-FOV", require_consistent_coverage)
        result = result.merge(_subject_major(full_valid, "visible_dice", "full_fov_macro"),
                              on="epoch", how="left")
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
        prec = getattr(r, "missing_precision_macro", None)
        out.append(CheckpointScore(
            int(r.epoch), float(r.missing_macro), float(r.visible_macro),
            float(r.worst_condition),
            None if (full is None or pd.isna(full)) else float(full),
            None if pd.isna(r.smoothed_missing) else float(r.smoothed_missing),
            None if (prec is None or pd.isna(prec)) else float(prec)))
    return out


def select_fov_checkpoint(
    scores: Sequence[CheckpointScore],
    missing_tolerance: float = 0.005,
    relaxation_steps: Sequence[float] = (0.005, 0.010, 0.020),
    warn: bool = True,
    strict_guardrail: bool = False,
) -> SelectionResult:
    """Select a checkpoint (review §5.3/5.4 + re-audit §10.3). Primary = missing
    macro Dice; near-best within ``missing_tolerance``; visible + full-FOV
    guardrails with an escalating tolerance; INDEPENDENT per-guardrail relaxation
    reported in the result. If full-FOV validation exists it must be COMPLETE —
    a missing full_fov_macro is an error, never a silent guardrail bypass.

    ``strict_guardrail`` (review §12, for the FINAL paper selection): if even the
    loosest tolerance removes every near-best candidate, RAISE instead of silently
    falling back to raw near-best missing Dice."""
    if not scores:
        raise ValueError("no checkpoint scores.")
    scores = list(scores)

    best_missing = max(s.missing_macro for s in scores)
    best_visible = max(s.visible_macro for s in scores)

    # full-FOV completeness (re-audit §10.3 / P0-5): all-or-nothing, no bypass.
    full_present = any(s.full_fov_macro is not None for s in scores)
    if full_present:
        incomplete = sorted(s.epoch for s in scores if s.full_fov_macro is None)
        if incomplete:
            raise ValueError(f"full-FOV validation is present but epochs {incomplete} "
                             "lack full_fov_macro; refuse to select on incomplete "
                             "metrics (a None must not bypass the full-FOV guardrail).")
        best_full = max(s.full_fov_macro for s in scores)  # type: ignore[arg-type]
    else:
        best_full = None

    # hallucination guardrail (revised-plan §6.5): missing-region precision — must be
    # COMPLETE if present (a None must not bypass it, same rule as full-FOV).
    prec_present = any(s.missing_precision_macro is not None for s in scores)
    if prec_present:
        incomplete = sorted(s.epoch for s in scores if s.missing_precision_macro is None)
        if incomplete:
            raise ValueError(f"hallucination (missing-precision) guardrail is present but "
                             f"epochs {incomplete} lack it; refuse to select on incomplete "
                             f"metrics (a None must not bypass the hallucination guardrail).")
        best_prec = max(s.missing_precision_macro for s in scores)  # type: ignore[arg-type]
    else:
        best_prec = None

    near_best = [s for s in scores if s.missing_macro >= best_missing - missing_tolerance]
    base = float(relaxation_steps[0])

    def _ranking_key(s: CheckpointScore):
        smoothed = s.smoothed_missing if s.smoothed_missing is not None else s.missing_macro
        return (s.worst_condition, smoothed, s.missing_macro, -s.epoch)   # earlier epoch wins ties

    def _result(chosen: CheckpointScore) -> SelectionResult:
        vis_margin = best_visible - chosen.visible_macro
        vis_relaxed = vis_margin > base + 1e-12
        if best_full is None:
            full_margin, full_relaxed, full_tol = None, False, None
        else:
            full_margin = best_full - (chosen.full_fov_macro or 0.0)
            full_relaxed = full_margin > base + 1e-12
            full_tol = max(base, full_margin)
        if best_prec is None:
            prec_margin, prec_relaxed, prec_tol = None, False, None
        else:
            prec_margin = best_prec - (chosen.missing_precision_macro or 0.0)
            prec_relaxed = prec_margin > base + 1e-12
            prec_tol = max(base, prec_margin)
        if (vis_relaxed or full_relaxed or prec_relaxed) and warn:
            warnings.warn(f"[fov-ckpt] guardrail relaxed for epoch {chosen.epoch} "
                          f"(visible margin {vis_margin:.4f}"
                          + ("" if full_margin is None else f", full margin {full_margin:.4f}")
                          + ("" if prec_margin is None else f", precision margin {prec_margin:.4f}")
                          + ").", stacklevel=2)
        return SelectionResult(chosen, visible_guardrail_relaxed=vis_relaxed,
                               full_fov_guardrail_relaxed=full_relaxed,
                               applied_visible_tolerance=max(base, vis_margin),
                               applied_full_tolerance=full_tol,
                               hallucination_guardrail_relaxed=prec_relaxed,
                               applied_precision_tolerance=prec_tol)

    for tol in relaxation_steps:
        guarded = [s for s in near_best
                   if s.visible_macro >= best_visible - tol
                   and (best_full is None or s.full_fov_macro >= best_full - tol)
                   and (best_prec is None or s.missing_precision_macro >= best_prec - tol)]
        if guarded:
            return _result(max(guarded, key=_ranking_key))

    # all guardrails empty even at the loosest tolerance
    if strict_guardrail:                                       # review §12 (final)
        raise ValueError(
            "[fov-ckpt] visible/full guardrails removed EVERY near-best candidate even "
            "at the loosest tolerance, and strict_guardrail=True: refusing to fall back "
            "to raw near-best missing Dice. Inspect the metrics or relax explicitly.")
    if warn:
        warnings.warn("[fov-ckpt] visible/full guardrails removed EVERY near-best "
                      "candidate even at the loosest tolerance; falling back to raw "
                      "near-best missing Dice.", stacklevel=2)
    return _result(max(near_best, key=_ranking_key))


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
          f"relaxed(vis={res2.visible_guardrail_relaxed}, full={res2.full_fov_guardrail_relaxed}, "
          f"vis_tol={res2.applied_visible_tolerance:.3f})")
    assert res2.visible_guardrail_relaxed and not res2.full_fov_guardrail_relaxed, \
        "only the visible guardrail should report relaxed here"
    assert res2.checkpoint.epoch == 1

    # full-FOV completeness (§10.3): a None among present full metrics must raise.
    mixed = [CheckpointScore(1, 0.70, 0.90, 0.6, full_fov_macro=0.93, smoothed_missing=0.70),
             CheckpointScore(2, 0.69, 0.90, 0.6, full_fov_macro=None, smoothed_missing=0.69)]
    try:
        select_fov_checkpoint(mixed, warn=False)
        raise AssertionError("incomplete full-FOV validation should raise")
    except ValueError:
        pass

    # ── §12 new guards ──────────────────────────────────────────────────────
    def _row(ep, st, mdice=0.5, vdice=0.9, mvox=500, vvox=800, ct="axial", sev=20):
        return dict(epoch=ep, crop_type=ct, severity=sev, structure=st, missing_dice=mdice,
                    visible_dice=vdice, missing_gt_voxels=mvox, visible_gt_voxels=vvox)

    # (a) empty visible-valid -> raise
    try:
        build_checkpoint_scores(pd.DataFrame([_row(1, "ON", vvox=0)]),
                                min_missing_volume_voxels=1, min_visible_volume_voxels=1)
        raise AssertionError("empty visible-valid should raise")
    except ValueError:
        pass
    # (b) non-finite missing dice -> raise
    try:
        build_checkpoint_scores(pd.DataFrame([_row(1, "ON", mdice=float("nan")), _row(1, "Globe")]),
                                min_missing_volume_voxels=1, min_visible_volume_voxels=1)
        raise AssertionError("non-finite missing dice should raise")
    except ValueError:
        pass
    # (c) inconsistent per-epoch coverage: strict raises, non-strict only warns
    cov = pd.DataFrame([_row(1, "ON"), _row(1, "Globe"), _row(2, "ON")])
    try:
        build_checkpoint_scores(cov, min_missing_volume_voxels=1, min_visible_volume_voxels=1)
        raise AssertionError("inconsistent coverage should raise (strict)")
    except ValueError:
        pass
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        build_checkpoint_scores(cov, min_missing_volume_voxels=1, min_visible_volume_voxels=1,
                                require_consistent_coverage=False)
    # (d) strict_guardrail: every near-best fails the guardrail -> raise (no raw fallback)
    try:
        select_fov_checkpoint(forced, missing_tolerance=0.005, warn=False, strict_guardrail=True)
        raise AssertionError("strict_guardrail should raise when all near-best fail")
    except ValueError:
        pass
    # (e) brief-recs item 6: absolute coverage. The full synthetic df covers the whole
    # 2x3x4 + full x4 grid -> passes; dropping a structure from EVERY epoch -> raises.
    build_checkpoint_scores(df, min_missing_volume_voxels=nvox, min_visible_volume_voxels=nvox,
                            expected_structures=structs)
    df_gap = df[df["structure"] != "Fat"]        # Fat absent in ALL epochs
    try:
        build_checkpoint_scores(df_gap, min_missing_volume_voxels=nvox,
                                min_visible_volume_voxels=nvox, expected_structures=structs)
        raise AssertionError("absolute coverage should raise when a structure is never present")
    except ValueError:
        pass
    print("§12 guards OK: empty-visible / non-finite / coverage / strict-guardrail / absolute all raise")

    # ── revised-plan §10: subject-major aggregation weights subjects EQUALLY ──
    sub = pd.DataFrame([
        dict(epoch=1, subject_id="A", crop_type="axial", severity=20, structure="ON",
             missing_dice=0.9, visible_dice=0.9, missing_gt_voxels=500, visible_gt_voxels=500),
        dict(epoch=1, subject_id="A", crop_type="axial", severity=35, structure="ON",
             missing_dice=0.9, visible_dice=0.9, missing_gt_voxels=500, visible_gt_voxels=500),
        dict(epoch=1, subject_id="B", crop_type="axial", severity=20, structure="ON",
             missing_dice=0.5, visible_dice=0.9, missing_gt_voxels=500, visible_gt_voxels=500),
    ])
    rs = build_checkpoint_scores(sub, min_missing_volume_voxels=1, min_visible_volume_voxels=1,
                                 require_consistent_coverage=False)
    # subject-major = (A:0.9 + B:0.5)/2 = 0.70; a flat cell mean over 3 rows would be 0.767
    assert abs(float(rs["missing_macro"].iloc[0]) - 0.70) < 1e-6, rs["missing_macro"].iloc[0]
    print("subject-major macro:", round(float(rs["missing_macro"].iloc[0]), 4), "(flat would be 0.7667)")

    # ── revised-plan §6.5: hallucination (missing-precision) guardrail ──
    # epoch 1 has the best missing Dice but LOW missing precision (invents anatomy in
    # the missing region) -> rejected for the high-precision epoch 2.
    halluc = [
        CheckpointScore(1, 0.700, 0.90, 0.6, full_fov_macro=0.93, smoothed_missing=0.700,
                        missing_precision_macro=0.60),
        CheckpointScore(2, 0.697, 0.90, 0.6, full_fov_macro=0.93, smoothed_missing=0.697,
                        missing_precision_macro=0.92),
    ]
    resh = select_fov_checkpoint(halluc, missing_tolerance=0.005, warn=False)
    assert resh.checkpoint.epoch == 2, resh.checkpoint.epoch
    assert not resh.hallucination_guardrail_relaxed
    # completeness: a None precision among present metrics must raise (no bypass)
    mixp = [CheckpointScore(1, 0.70, 0.90, 0.6, full_fov_macro=0.93, smoothed_missing=0.70,
                            missing_precision_macro=0.9),
            CheckpointScore(2, 0.69, 0.90, 0.6, full_fov_macro=0.93, smoothed_missing=0.69,
                            missing_precision_macro=None)]
    try:
        select_fov_checkpoint(mixp, warn=False)
        raise AssertionError("incomplete precision guardrail should raise")
    except ValueError:
        pass
    print("hallucination guardrail OK: low-precision best-Dice rejected; incomplete raises")

    print("FOV CHECKPOINT-SELECT SELF-TEST PASSED")
    return 0


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))   # find sibling plan_spacing
    raise SystemExit(_selftest())
