#!/usr/bin/env python3
"""Subject-level statistics + comparison arms for the FOV-completion FINAL test
(revised-plan §9/§10, P1-7/8).

The statistical unit is the SUBJECT, never the FOV case: each subject contributes
correlated variants, so six truncations of one subject are NOT six independent
samples. Aggregation:

    case-level metric
    → condition-specific subject metric
    → subject-level summary
    → population summary

and paired inference (corrector vs CNISP / stage-1) is done BY SUBJECT via paired
bootstrap + sign-flip permutation (+ Wilcoxon signed-rank when scipy is available).

Inputs are the per-arm long metric CSVs emitted by fov_completion_eval.py (columns
include subject_id, case_id, crop_type, severity, structure, and the metric of
interest, e.g. missing_dice / visible_dice / missing_fp_voxels / whole_dice).

All logic is pure pandas/numpy and unit-tested (``--self-test``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


def subject_level(df: pd.DataFrame, value_col: str,
                  within: Sequence[str] = ("crop_type", "severity", "structure")) -> pd.DataFrame:
    """Per-subject value = mean over the subject's cells (subject-major, §10). Returns
    columns [subject_id, value]. Non-finite cells are dropped before averaging."""
    d = df[np.isfinite(df[value_col].astype(float))].copy()
    # cell = one row per (subject, within...); average up to the subject.
    cell = d.groupby(["subject_id", *within], as_index=False).agg(_v=(value_col, "mean"))
    return cell.groupby("subject_id", as_index=False).agg(value=("_v", "mean"))


def paired_by_subject(df_a: pd.DataFrame, df_b: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Per-subject paired values (arm A vs arm B) on the shared subject set. Returns
    [subject_id, a, b, delta] with delta = a - b."""
    a = subject_level(df_a, value_col).rename(columns={"value": "a"})
    b = subject_level(df_b, value_col).rename(columns={"value": "b"})
    m = a.merge(b, on="subject_id", how="inner")
    m["delta"] = m["a"] - m["b"]
    return m


def paired_test(delta: Sequence[float], n_boot: int = 10000, seed: int = 0) -> Dict[str, float]:
    """Paired inference on per-subject deltas: mean, 95% bootstrap CI (resample
    subjects), sign-flip permutation p (two-sided), and Wilcoxon p if scipy is
    present. Deterministic given ``seed``."""
    d = np.asarray([x for x in delta if np.isfinite(x)], dtype=float)
    n = len(d)
    out: Dict[str, float] = {"n": n, "mean": float(np.mean(d)) if n else float("nan")}
    if n < 2:
        out.update(ci_lo=float("nan"), ci_hi=float("nan"), perm_p=float("nan"), wilcoxon_p=float("nan"))
        return out
    rng = np.random.default_rng(seed)
    boot = np.array([np.mean(d[rng.integers(0, n, n)]) for _ in range(n_boot)])
    out["ci_lo"], out["ci_hi"] = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    # sign-flip permutation: null = symmetric about 0
    obs = abs(float(np.mean(d)))
    signs = rng.choice([-1.0, 1.0], size=(n_boot, n))
    null = np.abs((signs * d).mean(axis=1))
    out["perm_p"] = float((np.sum(null >= obs - 1e-12) + 1) / (n_boot + 1))
    try:
        from scipy.stats import wilcoxon                        # noqa: WPS433
        if np.any(d != 0):
            out["wilcoxon_p"] = float(wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue)
        else:
            out["wilcoxon_p"] = 1.0
    except Exception:                                            # noqa: BLE001
        out["wilcoxon_p"] = float("nan")
    return out


def compare_arms(arms: Dict[str, pd.DataFrame], value_col: str, baseline: str,
                 target: str = "corrector", n_boot: int = 10000, seed: int = 0) -> Dict[str, object]:
    """Paired ``target - baseline`` on ``value_col`` by subject (revised-plan §9)."""
    pair = paired_by_subject(arms[target], arms[baseline], value_col)
    stats = paired_test(pair["delta"].to_numpy(), n_boot=n_boot, seed=seed)
    return {"value_col": value_col, "target": target, "baseline": baseline,
            "per_subject": pair, "stats": stats}


def stability_across_truncations(df: pd.DataFrame, value_col: str = "vol_pred_mm3",
                                 dice_col: str = "missing_dice") -> pd.DataFrame:
    """Per (subject, structure) across the six truncated conditions (revised-plan
    §8.5): volume CoV, normalized range, mean signed bias vs full, worst-condition
    Dice. ``full`` rows (crop_type=='full') are excluded from the truncated spread."""
    t = df[df["crop_type"] != "full"]
    rows = []
    for (subj, st), g in t.groupby(["subject_id", "structure"]):
        v = g[value_col].astype(float).to_numpy()
        v = v[np.isfinite(v)]
        mu = float(np.mean(v)) if len(v) else float("nan")
        cov = float(np.std(v) / mu) if len(v) and mu > 0 else float("nan")
        rng_norm = float((v.max() - v.min()) / mu) if len(v) and mu > 0 else float("nan")
        dice = g[dice_col].astype(float).to_numpy()
        dice = dice[np.isfinite(dice)]
        rows.append(dict(subject_id=subj, structure=st, vol_cov=cov, vol_norm_range=rng_norm,
                         vol_mean=mu, worst_dice=float(dice.min()) if len(dice) else float("nan")))
    return pd.DataFrame(rows)


def _selftest() -> int:
    # 6 subjects, corrector systematically better than CNISP on missing_dice.
    rng = np.random.default_rng(0)
    rows_c, rows_p = [], []
    for s in range(6):
        sid = f"{s:03d}"
        for ct in ("axial", "corner"):
            for sev in (20, 35, 50):
                for st in ("ON", "Globe"):
                    base = 0.6 + 0.02 * s
                    rows_c.append(dict(subject_id=sid, case_id=f"{sid}_{ct}_{sev}", crop_type=ct,
                                       severity=sev, structure=st,
                                       missing_dice=base + 0.08 + rng.normal(0, 0.005),
                                       vol_pred_mm3=100 + rng.normal(0, 3)))
                    rows_p.append(dict(subject_id=sid, case_id=f"{sid}_{ct}_{sev}", crop_type=ct,
                                       severity=sev, structure=st,
                                       missing_dice=base + rng.normal(0, 0.005),
                                       vol_pred_mm3=100 + rng.normal(0, 3)))
    corr, cnisp = pd.DataFrame(rows_c), pd.DataFrame(rows_p)

    # subject-level: 6 subjects, corrector mean ~0.08 above CNISP
    sc = subject_level(corr, "missing_dice")
    assert len(sc) == 6 and 0.65 < sc["value"].mean() < 0.80

    cmp = compare_arms({"corrector": corr, "cnisp": cnisp}, "missing_dice", baseline="cnisp")
    st = cmp["stats"]
    print("delta missing_dice:", {k: round(v, 4) if isinstance(v, float) else v for k, v in st.items()})
    assert st["n"] == 6 and st["mean"] > 0.05
    assert st["ci_lo"] > 0.0                       # CI excludes 0 -> corrector > CNISP
    assert st["perm_p"] < 0.10                     # 6 subjects all positive -> ~1/64 two-sided

    # a null comparison (same arm vs itself) -> delta 0, non-significant
    null = compare_arms({"corrector": corr, "same": corr.copy()}, "missing_dice", baseline="same")
    assert abs(null["stats"]["mean"]) < 1e-9 and null["stats"]["perm_p"] > 0.5

    # stability across truncations: 6 conditions per (subject, structure)
    stab = stability_across_truncations(corr, "vol_pred_mm3")
    assert len(stab) == 6 * 2 and (stab["vol_cov"] >= 0).all()
    print("stability rows:", len(stab), "| example CoV:", round(float(stab['vol_cov'].iloc[0]), 4))
    print("FOV-COMPLETION-STATS SELF-TEST PASSED")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corrector-csv", help="corrector arm long metrics")
    ap.add_argument("--baseline-csv", help="baseline arm (CNISP prior / stage-1) long metrics")
    ap.add_argument("--baseline-name", default="cnisp")
    ap.add_argument("--metric", default="missing_dice")
    ap.add_argument("--out-csv", default=None, help="per-subject paired deltas")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return _selftest()
    if not (args.corrector_csv and args.baseline_csv):
        ap.error("--corrector-csv and --baseline-csv required (or --self-test).")
    corr = pd.read_csv(args.corrector_csv)
    base = pd.read_csv(args.baseline_csv)
    cmp = compare_arms({"corrector": corr, args.baseline_name: base}, args.metric,
                       baseline=args.baseline_name)
    print(f"[fov-stats] {args.metric}: corrector - {args.baseline_name}")
    print("  ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in cmp["stats"].items()})
    if args.out_csv:
        cmp["per_subject"].to_csv(args.out_csv, index=False)
        print(f"  wrote {args.out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
