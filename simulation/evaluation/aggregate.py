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
                     ba_arms=("nnU-Net", "Proposed")) -> Tuple[Dict, Dict]:
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


__all__ = ["stability", "volume_agreement", "surface",
           "DEFAULT_MODE", "DEFAULT_BA_STRUCTURE"]
