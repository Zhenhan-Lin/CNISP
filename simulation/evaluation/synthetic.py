"""Synthetic placeholder aggregates so the figures render without real masks.

Illustrative fallback layer: when no metrics table / MASK_INDEX is supplied, the
``*_summary`` drivers use these to render the figure LAYOUT (marked as synthetic
via ``plots._foot``). Shapes match the outputs of ``aggregate.*`` so the plotting
code path is identical to the real-data one.
"""

from __future__ import annotations

import numpy as np

from simulation.evaluation.metrics import METHODS, STRUCTURES


def stability(seed: int = 7):
    rng = np.random.default_rng(seed)
    cov_mean = {"nnU-Net": {"Globe": 12, "Optic nerve": 30, "Recti": 28, "Fat": 18},
                "CNISP": {"Globe": 5, "Optic nerve": 11, "Recti": 12, "Fat": 8},
                "nnU\u2192nnU": {"Globe": 9, "Optic nerve": 22, "Recti": 20, "Fat": 14},
                "Proposed": {"Globe": 3, "Optic nerve": 7, "Recti": 8, "Fat": 5},
                "Oracle": {"Globe": 2, "Optic nerve": 4, "Recti": 4, "Fat": 3}}
    cov_sd = {m: {s: cov_mean[m][s] * 0.18 for s in STRUCTURES} for m in METHODS}
    sh = {"nnU-Net": (4, 3, 4), "CNISP": (3, 1.2, 1), "nnU\u2192nnU": (4, 2.2, 3),
          "Proposed": (2.5, 0.7, 0.6), "Oracle": (2, 0.4, 0.3)}
    on_range = {m: rng.gamma(sh[m][0], sh[m][1], 60) + sh[m][2] for m in METHODS}
    return cov_mean, cov_sd, on_range


def volume_agreement(seed: int = 11):
    rng = np.random.default_rng(seed); n = 48
    v_gt = rng.uniform(5800, 7400, n); thickness = rng.uniform(0.5, 4.5, n)
    pred = lambda b0, sl, no: v_gt + b0 + sl * thickness + rng.standard_normal(n) * no
    per_arm = {"nnU-Net":  dict(v_gt=v_gt, thickness=thickness, v_pred=pred(300, 250, 200)),
               "Proposed": dict(v_gt=v_gt, thickness=thickness, v_pred=pred(30, 12, 80))}

    def sk(mean, sd, skew, k=90):
        g = rng.gamma(2.5, 1, k); g = (g - g.mean()) / g.std()
        return mean + sd * (g * skew + rng.standard_normal(k) * (1 - abs(skew)))

    signed = {"nnU-Net": sk(12, 7, .6), "CNISP": sk(4, 4, .3), "nnU\u2192nnU": sk(7, 6, .5),
              "Proposed": sk(1, 2.2, .2), "Oracle": sk(.3, 1.3, .1)}
    return per_arm, signed


def surface(seed: int = 5):
    rng = np.random.default_rng(seed)
    spec = {"ASSD (mm)": {"nnU-Net": (1.8, .5), "CNISP": (.9, .25), "nnU\u2192nnU": (1.4, .4), "Proposed": (.6, .18), "Oracle": (.4, .12)},
            "HD95 (mm)": {"nnU-Net": (6, 1.6), "CNISP": (3, .9), "nnU\u2192nnU": (5, 1.4), "Proposed": (2, .6), "Oracle": (1.4, .4)},
            "Surface Dice @1mm": {"nnU-Net": (.62, .08), "CNISP": (.85, .05), "nnU\u2192nnU": (.70, .07), "Proposed": (.92, .03), "Oracle": (.96, .02)}}
    return {name: {m: rng.normal(mu, sd, 40) for m, (mu, sd) in d.items()} for name, d in spec.items()}


__all__ = ["stability", "volume_agreement", "surface"]
