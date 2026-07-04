"""Aggregate the long metrics table into plot-ready inputs (aggregation layer).

Middle layer of ``simulation.evaluation`` (analogous to the ``aggregate_*``
helpers in ``nnunet.lib.viz``): consume the tidy per-structure table from
``metrics.build_metrics_table`` and reduce it to the arrays each figure needs.
Nothing here plots.

Depends on numpy + pandas.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from simulation.evaluation.metrics import METHODS, STRUCTURES

DEFAULT_MODE = "thick"          # sweep mode aggregated for CoV / surface panels
DEFAULT_BA_STRUCTURE = "Globe"  # structure shown in the Bland-Altman panels


def restrict_to_common(df, methods=None, verbose: bool = True):
    """Keep only (case, step) rows present for EVERY compared method.

    Makes every figure average over the IDENTICAL (source, step) population, so a
    difference reflects method quality rather than which cases a method happened
    to cover (e.g. Proposed producing fewer (case, step) than nnUNet). This is the
    figure-side analogue of ``eval_corrector.py --intersect-with``.

    ``methods`` defaults to every arm present in ``df`` EXCEPT ``GT`` (the GT
    reference is derived from the GT and is present for every case, so it never
    constrains the intersection; its rows are kept on the surviving keys).
    """
    if methods is None:
        present = set(df["arm"])
        methods = [m for m in METHODS if m != "GT" and m in present]
    if not methods:
        return df
    sub = df[df["arm"].isin(methods)]
    counts = sub.groupby(["case", "step"])["arm"].nunique()
    common = counts[counts == len(methods)].index          # (case, step) with all
    keep = df.set_index(["case", "step"]).index.isin(common)
    out = df.loc[keep].reset_index(drop=True)
    if verbose:
        import sys
        n_all = df.groupby(["case", "step"]).ngroups
        print(f"[common] {len(common)}/{n_all} (case, step) common to all "
              f"{len(methods)} methods {methods}; kept {len(out)}/{len(df)} rows.",
              file=sys.stderr)
    return out


def stability(df, mode: str = DEFAULT_MODE) -> Tuple[Dict, Dict, Dict]:
    """Volume CoV across step-sizes per (method, structure) + optic-nerve range.

    Returns ``(cov_mean, cov_sd, on_range)`` where cov_* are
    ``{method: {structure: value}}`` and on_range is ``{method: array}``.
    """
    sub = df[df["mode"] == mode]
    cov = (sub.groupby(["arm", "structure", "case"])["vol_pred"]
              .apply(lambda v: 100.0 * v.std(ddof=0) / v.mean() if v.mean() > 0 else np.nan)
              .reset_index(name="cov"))
    cov_mean = {m: {s: float(cov[(cov.arm == m) & (cov.structure == s)]["cov"].mean())
                    for s in STRUCTURES} for m in METHODS}
    cov_sd = {m: {s: float(cov[(cov.arm == m) & (cov.structure == s)]["cov"].std())
                  for s in STRUCTURES} for m in METHODS}
    on = sub[sub.structure == "Optic nerve"]
    rng = (on.groupby(["arm", "case"])["vol_pred"]
             .apply(lambda v: 100.0 * (v.max() - v.min()) / v.mean() if v.mean() > 0 else np.nan)
             .reset_index(name="rng"))
    on_range = {m: rng[rng.arm == m]["rng"].dropna().values for m in METHODS}
    return cov_mean, cov_sd, on_range


def volume_agreement(df, structure: str = DEFAULT_BA_STRUCTURE,
                     ba_arms=("nnUNet", "Proposed")) -> Tuple[Dict, Dict]:
    """Per-arm Bland-Altman inputs for one structure + pooled signed error/method."""
    per_arm = {}
    for m in ba_arms:
        r = df[(df.structure == structure) & (df.arm == m)]
        per_arm[m] = dict(v_gt=r["vol_gt"].values, v_pred=r["vol_pred"].values,
                          thickness=r["eff_res"].values)
    signed = {m: df[df.arm == m]["signed_pct"].dropna().values for m in METHODS}
    return per_arm, signed


def surface(df, mode: str = DEFAULT_MODE) -> Dict[str, Dict[str, np.ndarray]]:
    """Per-method surface-metric distributions (mean over structures per case/step)."""
    sub = df[df["mode"] == mode]
    g = sub.groupby(["arm", "case", "step"])[["assd", "hd95", "nsd"]].mean().reset_index()
    return {"ASSD (mm)":         {m: g[g.arm == m]["assd"].dropna().values for m in METHODS},
            "HD95 (mm)":         {m: g[g.arm == m]["hd95"].dropna().values for m in METHODS},
            "Surface Dice @1mm": {m: g[g.arm == m]["nsd"].dropna().values for m in METHODS}}


__all__ = ["stability", "volume_agreement", "surface", "restrict_to_common",
           "DEFAULT_MODE", "DEFAULT_BA_STRUCTURE"]
