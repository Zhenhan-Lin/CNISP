"""Aggregate plausibility metrics into plot-ready inputs (aggregation layer).

Middle layer: consume the tidy per-(case, arm, step, eye, structure) table from
``plausibility.build_plausibility_table`` and reduce to figure inputs +
statistical test results.

Two-layer comparison structure:
  Layer 1 (prior channel): nnUNet vs CNISP
  Layer 2 (cascade output): Cascade UNet vs Proposed

Depends on numpy, pandas, scipy.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from simulation.evaluation.metrics import METHODS, STRUCTURES

# Explicit layer comparison pairs: (source_a, source_b, layer_label)
LAYER_COMPARISONS: List[Tuple[str, str, str]] = [
    ("nnUNet", "CNISP", "Layer1_prior"),
    ("Cascade UNet", "Proposed", "Layer2_cascade"),
]

# Arms that appear in plausibility figures (subset of METHODS)
PLAUSIBILITY_ARMS: List[str] = ["nnUNet", "CNISP", "Cascade UNet", "Proposed"]

CONTINUOUS_METRICS: List[str] = [
    "max_centroid_jump_mm", "mean_centroid_jump_mm",
    "max_area_rel_change", "mean_area_rel_change",
    "num_gaps", "num_cc", "num_holes",
]

BINARY_METRICS: List[str] = ["has_multi_cc", "has_holes"]


def assign_buckets(df: pd.DataFrame, edges=None) -> pd.DataFrame:
    """Add bucket_idx and bucket_label columns from eff_res."""
    import math
    from nnunet.helpers.buckets import assign_bucket, DEFAULT_BUCKET_EDGES_MM

    if edges is None:
        edges = list(DEFAULT_BUCKET_EDGES_MM)

    idxs, labels = [], []
    for er in df["eff_res"]:
        i, lab = assign_bucket(er, edges)
        idxs.append(i)
        labels.append(lab)
    out = df.copy()
    out["bucket_idx"] = idxs
    out["bucket_label"] = labels
    return out


def restrict_to_common_cases(df: pd.DataFrame, comparisons=None) -> pd.DataFrame:
    """Keep only (case, step) present for BOTH arms in every comparison pair.

    Ensures fair paired evaluation: each pair is computed on the identical
    population, analogous to aggregate.restrict_to_common for the 5-arm system.
    """
    if comparisons is None:
        comparisons = LAYER_COMPARISONS

    arms_needed = set()
    for a, b, _ in comparisons:
        arms_needed.add(a)
        arms_needed.add(b)

    present_arms = set(df["arm"].unique())
    arms_needed = arms_needed & present_arms
    if not arms_needed:
        return df

    sub = df[df["arm"].isin(arms_needed)]
    counts = sub.groupby(["case", "step"])["arm"].nunique()
    common = counts[counts == len(arms_needed)].index
    keep = df.set_index(["case", "step"]).index.isin(common)
    return df.loc[keep].reset_index(drop=True)


# ============================================================
# Topology violation rate
# ============================================================

def _wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a proportion k/n."""
    if n == 0:
        return (0.0, 0.0)
    p_hat = k / n
    denom = 1 + z ** 2 / n
    centre = (p_hat + z ** 2 / (2 * n)) / denom
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z ** 2 / (4 * n ** 2)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def topology_violation_rate(
    df: pd.DataFrame,
    metric: str = "has_multi_cc",
    groupby: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Fraction of (case, step, eye) with a topology violation per group.

    Returns DataFrame with columns: [groupby...], rate, ci_lo, ci_hi, n.
    """
    if groupby is None:
        groupby = ["arm", "structure", "bucket_label"]

    grouped = df.groupby(groupby, dropna=False)
    rows = []
    for keys, grp in grouped:
        n = len(grp)
        k = int(grp[metric].sum())
        rate = k / n if n > 0 else 0.0
        ci_lo, ci_hi = _wilson_ci(k, n)
        row = dict(zip(groupby, keys if isinstance(keys, tuple) else (keys,)))
        row.update({"rate": rate, "ci_lo": ci_lo, "ci_hi": ci_hi, "n": n})
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================
# Continuity by bucket
# ============================================================

def continuity_by_bucket(
    df: pd.DataFrame,
    metric: str = "mean_centroid_jump_mm",
    groupby: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Mean/SD of a continuity metric per group.

    Returns DataFrame with columns: [groupby...], mean, sd, n.
    """
    if groupby is None:
        groupby = ["arm", "structure", "bucket_label"]

    grouped = df.groupby(groupby, dropna=False)
    rows = []
    for keys, grp in grouped:
        vals = grp[metric].dropna().values.astype(float)
        row = dict(zip(groupby, keys if isinstance(keys, tuple) else (keys,)))
        row.update({
            "mean": float(vals.mean()) if len(vals) > 0 else np.nan,
            "sd": float(vals.std()) if len(vals) > 1 else 0.0,
            "n": len(vals),
        })
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================
# Paired statistical tests
# ============================================================

def paired_tests(
    df: pd.DataFrame,
    comparisons: Optional[List[Tuple[str, str, str]]] = None,
    save_csv=None,
) -> pd.DataFrame:
    """Paired tests between Layer comparison sources.

    For each (source_a, source_b, layer_label):
      - Continuous metrics: Wilcoxon signed-rank test
      - Binary metrics: McNemar exact test (binomtest)

    Tests are per-structure overall, and per-bucket.
    Returns a DataFrame of test results.
    """
    from scipy.stats import wilcoxon, binomtest

    if comparisons is None:
        comparisons = LAYER_COMPARISONS

    results: List[Dict] = []

    for src_a, src_b, layer_name in comparisons:
        df_a = df[df["arm"] == src_a]
        df_b = df[df["arm"] == src_b]

        if df_a.empty or df_b.empty:
            continue

        for struct in STRUCTURES:
            da = df_a[df_a["structure"] == struct].set_index(["case", "step", "eye"])
            db = df_b[df_b["structure"] == struct].set_index(["case", "step", "eye"])
            common = da.index.intersection(db.index)

            if len(common) < 5:
                continue

            # Overall continuous metrics
            for metric in CONTINUOUS_METRICS:
                vals_a = da.loc[common, metric].values.astype(float)
                vals_b = db.loc[common, metric].values.astype(float)
                diff = vals_a - vals_b

                if np.all(diff == 0):
                    p_val = 1.0
                else:
                    try:
                        _, p_val = wilcoxon(diff, alternative="two-sided")
                    except Exception:
                        p_val = np.nan

                results.append({
                    "layer": layer_name,
                    "structure": struct,
                    "metric": metric,
                    "bucket": "overall",
                    "n": len(common),
                    "mean_a": float(np.mean(vals_a)),
                    "mean_b": float(np.mean(vals_b)),
                    "p_value": p_val,
                    "direction": "a>b" if np.mean(vals_a) > np.mean(vals_b) else "a<=b",
                })

            # Overall binary metrics
            for metric in BINARY_METRICS:
                vals_a = da.loc[common, metric].values.astype(bool)
                vals_b = db.loc[common, metric].values.astype(bool)

                rate_a = float(np.mean(vals_a))
                rate_b = float(np.mean(vals_b))

                # McNemar: discordant pairs
                a_only = int(np.sum(vals_a & ~vals_b))
                b_only = int(np.sum(~vals_a & vals_b))

                if a_only + b_only == 0:
                    p_val = 1.0
                else:
                    try:
                        res = binomtest(a_only, a_only + b_only, 0.5)
                        p_val = res.pvalue
                    except Exception:
                        p_val = np.nan

                results.append({
                    "layer": layer_name,
                    "structure": struct,
                    "metric": metric,
                    "bucket": "overall",
                    "n": len(common),
                    "mean_a": rate_a,
                    "mean_b": rate_b,
                    "p_value": p_val,
                    "direction": "a>b" if rate_a > rate_b else "a<=b",
                })

            # Per-bucket tests
            if "bucket_label" not in da.columns:
                continue
            # Need to re-index with bucket_label accessible
            da_full = df_a[df_a["structure"] == struct]
            db_full = df_b[df_b["structure"] == struct]

            for bucket_label in da_full["bucket_label"].unique():
                da_bkt = da_full[da_full["bucket_label"] == bucket_label].set_index(
                    ["case", "step", "eye"])
                db_bkt = db_full[db_full["bucket_label"] == bucket_label].set_index(
                    ["case", "step", "eye"])
                common_bkt = da_bkt.index.intersection(db_bkt.index)

                if len(common_bkt) < 3:
                    continue

                for metric in CONTINUOUS_METRICS:
                    vals_a = da_bkt.loc[common_bkt, metric].values.astype(float)
                    vals_b = db_bkt.loc[common_bkt, metric].values.astype(float)
                    diff = vals_a - vals_b

                    if np.all(diff == 0):
                        p_val = 1.0
                    else:
                        try:
                            _, p_val = wilcoxon(diff, alternative="two-sided")
                        except Exception:
                            p_val = np.nan

                    results.append({
                        "layer": layer_name,
                        "structure": struct,
                        "metric": metric,
                        "bucket": bucket_label,
                        "n": len(common_bkt),
                        "mean_a": float(np.mean(vals_a)),
                        "mean_b": float(np.mean(vals_b)),
                        "p_value": p_val,
                        "direction": "a>b" if np.mean(vals_a) > np.mean(vals_b) else "a<=b",
                    })

    df_tests = pd.DataFrame(results)
    if save_csv and len(df_tests) > 0:
        from pathlib import Path
        Path(save_csv).parent.mkdir(parents=True, exist_ok=True)
        df_tests.to_csv(str(save_csv), index=False)
    return df_tests


__all__ = [
    "LAYER_COMPARISONS", "PLAUSIBILITY_ARMS",
    "CONTINUOUS_METRICS", "BINARY_METRICS",
    "assign_buckets", "restrict_to_common_cases",
    "topology_violation_rate", "continuity_by_bucket", "paired_tests",
]
