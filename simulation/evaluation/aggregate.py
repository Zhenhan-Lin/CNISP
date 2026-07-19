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


def _ccc(v_pred: np.ndarray, v_gt: np.ndarray) -> float:
    """Lin's concordance correlation coefficient (matches plots._bland_altman:108)."""
    v_pred = np.asarray(v_pred, dtype=float)
    v_gt = np.asarray(v_gt, dtype=float)
    if v_pred.size < 2:
        return float("nan")
    cov = float(np.cov(v_pred, v_gt, bias=True)[0, 1])
    denom = v_pred.var() + v_gt.var() + (v_pred.mean() - v_gt.mean()) ** 2
    return float(2.0 * cov / denom) if denom > 0 else float("nan")


def volume_agreement_per_arm(df, structure: str = DEFAULT_BA_STRUCTURE
                             ) -> Tuple[Dict, list]:
    """Bland-Altman inputs AND per-arm bias stats for EVERY arm in METHODS.

    The main figure (``volume_agreement``) only builds nnUNet + Proposed panels;
    this returns all five arms so the driver can print/dump a per-arm bias table
    and render one Bland-Altman panel per arm.

    Returns ``(per_arm, stats)``:
      * ``per_arm`` = ``{arm: {v_gt, v_pred, thickness}}`` for each METHODS arm.
      * ``stats``   = list of dicts (one per arm, METHODS order) with the SAME
        conventions the plotted panel uses (``diff = v_pred - v_gt``; ``bias =
        mean(diff)``; population ``std``; LoA = ``1.96*std``; Lin's CCC).
    """
    per_arm: Dict = {}
    stats: list = []
    for m in METHODS:
        r = df[(df.structure == structure) & (df.arm == m)]
        vp = np.asarray(r["vol_pred"].values, dtype=float)
        vg = np.asarray(r["vol_gt"].values, dtype=float)
        per_arm[m] = dict(v_gt=vg, v_pred=vp, thickness=r["eff_res"].values)
        diff = vp - vg
        n = int(diff.size)
        bias = float(diff.mean()) if n else float("nan")
        sd = float(diff.std()) if n else float("nan")          # ddof=0, matches plot
        sp = r["signed_pct"].dropna().values
        stats.append(dict(
            arm=m, structure=structure, n=n,
            bias_mm3=bias, sd_diff_mm3=sd,
            loa_lo_mm3=(bias - 1.96 * sd) if n else float("nan"),
            loa_hi_mm3=(bias + 1.96 * sd) if n else float("nan"),
            ccc=_ccc(vp, vg),
            mean_vol_pred_mm3=float(vp.mean()) if n else float("nan"),
            mean_vol_gt_mm3=float(vg.mean()) if n else float("nan"),
            mean_signed_pct=float(np.mean(sp)) if sp.size else float("nan"),
        ))
    return per_arm, stats


def stability_table(df, mode: str = DEFAULT_MODE) -> Tuple[list, list, list]:
    """Detailed + summary CoV / optic-nerve-range tables behind the stability fig.

    The figure (``stability`` -> ``plots.stability_figure``) keeps these numbers
    only in memory; this exposes them for a CSV dump so per-(arm, structure) CoV,
    the per-case detail, and the sample counts (e.g. how many cases -- and how many
    step_sizes per case -- feed each Oracle/ON value) are inspectable on disk.

    Same arithmetic as ``stability``: CoV = ``100 * std(ddof=0) / mean`` across
    step_sizes within a (arm, structure, case); ON range = ``100*(max-min)/mean``.
    ``n_steps`` = step_sizes present for that (arm, [structure,] case) -- a value
    of 1 yields CoV/range 0 (population std of a single sample), NOT dropped.

    Returns ``(cov_detail, cov_summary, rng_detail)`` as lists of dicts.
    """
    import pandas as pd
    sub = df[df["mode"] == mode]
    cov_detail: list = []
    for (arm, structure, case), v in sub.groupby(["arm", "structure", "case"])["vol_pred"]:
        mean = float(v.mean())
        cov = 100.0 * float(v.std(ddof=0)) / mean if mean > 0 else float("nan")
        cov_detail.append(dict(arm=arm, structure=structure, case=case,
                               n_steps=int(v.size), mean_vol_mm3=mean, cov_pct=cov))
    cd = pd.DataFrame(cov_detail)
    cov_summary: list = []
    for m in METHODS:
        for s in STRUCTURES:
            arr = (np.asarray(cd[(cd.arm == m) & (cd.structure == s)]["cov_pct"]
                              .dropna(), dtype=float) if not cd.empty else np.array([]))
            cov_summary.append(dict(
                arm=m, structure=s, n_cases=int(arr.size),
                cov_mean_pct=float(arr.mean()) if arr.size else float("nan"),
                cov_sd_pct=float(arr.std(ddof=1)) if arr.size > 1 else float("nan")))
    rng_detail: list = []
    on = sub[sub.structure == "Optic nerve"]
    for (arm, case), v in on.groupby(["arm", "case"])["vol_pred"]:
        mean = float(v.mean())
        rng = 100.0 * (float(v.max()) - float(v.min())) / mean if mean > 0 else float("nan")
        rng_detail.append(dict(arm=arm, case=case, n_steps=int(v.size),
                               mean_vol_mm3=mean, range_pct=rng))
    return cov_detail, cov_summary, rng_detail


def surface(df, mode: str = DEFAULT_MODE) -> Dict[str, Dict[str, np.ndarray]]:
    """Per-method surface-metric distributions (mean over structures per case/step)."""
    sub = df[df["mode"] == mode]
    g = sub.groupby(["arm", "case", "step"])[["assd", "hd95", "nsd"]].mean().reset_index()
    return {"ASSD (mm)":         {m: g[g.arm == m]["assd"].dropna().values for m in METHODS},
            "HD95 (mm)":         {m: g[g.arm == m]["hd95"].dropna().values for m in METHODS},
            "Surface Dice @1mm": {m: g[g.arm == m]["nsd"].dropna().values for m in METHODS}}


__all__ = ["stability", "stability_table", "volume_agreement",
           "volume_agreement_per_arm", "surface", "restrict_to_common",
           "METHODS", "STRUCTURES", "DEFAULT_MODE", "DEFAULT_BA_STRUCTURE"]
