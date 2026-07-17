"""Synthetic placeholder data for plausibility figures (layout verification).

Illustrative fallback: when no metrics table / mask_index is available, the
plausibility_summary driver can use these to render figure LAYOUTS with synthetic
data so the visual arrangement can be reviewed before real evaluation runs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from simulation.evaluation.metrics import STRUCTURES
from simulation.evaluation.plausibility_aggregate import PLAUSIBILITY_ARMS

_BUCKETS = ["(1.0, 2.0]", "(2.0, 3.0]", "(3.0, 4.0]",
            "(4.0, 5.0]", "(5.0, 6.5]", "(6.5, 8.5]", "(8.5, 11.0]"]


def plausibility_table(seed: int = 42, n_cases: int = 30) -> pd.DataFrame:
    """Generate a synthetic plausibility_long DataFrame for layout testing."""
    rng = np.random.default_rng(seed)

    # Violation rate tendencies per arm (lower = better topology)
    violation_prob = {
        "nnUNet": 0.35,
        "CNISP": 0.08,
        "Cascade UNet": 0.20,
        "Proposed": 0.05,
    }

    # Continuity jump tendencies (mean, sd)
    jump_params = {
        "nnUNet": (1.8, 0.6),
        "CNISP": (0.7, 0.3),
        "Cascade UNet": (1.2, 0.5),
        "Proposed": (0.5, 0.2),
    }

    rows = []
    for ci in range(n_cases):
        case_id = f"synth_{ci:03d}"
        step = rng.choice([3, 6, 9])
        eff_res = float(rng.uniform(1.5, 10.0))
        for arm in PLAUSIBILITY_ARMS:
            for eye in ("OD", "OS"):
                for struct in STRUCTURES:
                    vp = violation_prob[arm]
                    # Higher eff_res -> higher violation probability
                    adj_vp = min(vp * (1 + eff_res / 10.0), 0.8)
                    has_mc = bool(rng.random() < adj_vp)
                    has_h = bool(rng.random() < adj_vp * 0.3)

                    jm, js = jump_params[arm]
                    jump_scale = 1 + eff_res / 8.0
                    mean_jump = max(0, rng.normal(jm * jump_scale, js))
                    max_jump = mean_jump * rng.uniform(1.5, 3.0)
                    area_change = max(0, rng.normal(0.15 * jump_scale, 0.08))

                    rows.append({
                        "case": case_id,
                        "arm": arm,
                        "step": step,
                        "eff_res": eff_res,
                        "eye": eye,
                        "structure": struct,
                        "num_cc": (rng.integers(2, 5) if has_mc else 1),
                        "has_multi_cc": has_mc,
                        "num_holes": (rng.integers(1, 3) if has_h else 0),
                        "has_holes": has_h,
                        "volume_mm3": float(rng.uniform(200, 8000)),
                        "max_centroid_jump_mm": max_jump,
                        "mean_centroid_jump_mm": mean_jump,
                        "max_area_rel_change": area_change * 2,
                        "mean_area_rel_change": area_change,
                        "num_gaps": int(rng.integers(0, 3) if has_mc else 0),
                        "num_nonempty_slices": int(rng.integers(5, 40)),
                        "surface_area_mm2": None,
                        "compactness": float(rng.uniform(0.3, 0.95)),
                        "mean_curvature_var": None,
                    })

    return pd.DataFrame(rows)


__all__ = ["plausibility_table"]
