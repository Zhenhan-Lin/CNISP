#!/usr/bin/env python3
"""
Average-shape (ZERO-LATENT) Dice baseline on the test set.

The CNISP auto-decoder's zero latent is the "center" of the latent space (the L2
prior pulls every trained latent toward it), so f(x, z=0) is the shape prior's
learned MEAN anatomy. This script measures how much Dice that fixed mean shape
alone achieves on the test set — the FLOOR that per-case test-time latent fitting
improves on.

It reuses the exact test path (engine + diagnostics.resolution_sweep):

    per test case:
        eval_case_at_resolution(..., latent_override=ZERO, step_size=1)
            -> geometric canonical pose (NO pose optimization exists at test time)
            -> decode f(x, z=0) on the model's canonical grid
            -> _hard_dice(pred, GT) over classes {1:ON, 2:Globe, 3:Fat, 4:Recti}

so the numbers are computed with the SAME frame + SAME Dice function as the fitted-
latent test_results.csv, i.e. directly comparable. ``step_size=1`` centres the crop
on the true (dense) GT centroid, isolating shape mismatch from observation sparsity.

Usage (mirrors scripts/03_infer.py):
    cd orbital_shape_prior_st1
    python scripts/033_average_shape_dice.py \
        -p configs/paths.yaml \
        -t configs/train_sty2.yaml \
        -c configs/test_default.yaml \
        -m orbital_ad_v6 \
        --checkpoint best \
        --out-csv average_shape_dice.csv
    # subset:  --test-casefile test_cases_v7small.txt
    # (self-test the aggregation only, no model/GPU:  --self-test)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# canonical foreground labels (data_prep/canonical_align.CANONICAL_LABELS):
#   0=BG, 1=ON, 2=Globe, 3=Fat, 4=Recti  ->  _hard_dice per_class order is 1..4
STRUCT_NAMES = ["ON", "Globe", "Fat", "Recti"]


def _aggregate(per_case: List[Dict[str, float]]) -> Dict[str, float]:
    """Macro over cases of each per-structure Dice + the overall mean. Mirrors how
    test_results.csv is summarised (a jointly-absent structure already scored 0 in
    _hard_dice, so it is a real 0 in the mean, not dropped)."""
    agg: Dict[str, float] = {}
    for s in STRUCT_NAMES:
        vals = [r[s] for r in per_case if s in r and r[s] == r[s]]  # drop NaN if any
        agg[s] = float(np.mean(vals)) if vals else float("nan")
    finite = [agg[s] for s in STRUCT_NAMES if agg[s] == agg[s]]
    agg["macro"] = float(np.mean(finite)) if finite else float("nan")
    agg["n_cases"] = len(per_case)
    return agg


def evaluate_average_shape(params: dict, out_csv: Optional[str] = None,
                           step_axis: int = 2) -> Dict[str, float]:
    """Decode the zero-latent mean shape and score its Dice on the test set."""
    import functools

    import torch

    from engine.dataset import load_casenames
    from engine.infer import (_load_labels_dense_per_case, load_model_checkpoint,
                              optimize_latent)
    from engine.test_label_sources import build_run_layout
    from engine.train import create_model
    from diagnostics.resolution_sweep import eval_case_at_resolution

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── model (same load as infer_test_set) ──────────────────────────────────
    model_dir = Path(params["model_basedir"]) / params["model_name"]
    model_state, _meta = load_model_checkpoint(model_dir, params.get("checkpoint", "best"))
    net = create_model(params, torch.ones(3))          # image_size restored from ckpt buffers
    net.load_state_dict(model_state["net"], strict=True)
    net = net.to(device).eval()
    latent_dim = int(params["latent_dim"])
    num_classes = int(params.get("num_classes", 5))
    z_zero = np.zeros(latent_dim, dtype=np.float32)     # THE mean-shape latent
    # optimize_fn is required by the signature but never called (latent_override set).
    optimize_fn = functools.partial(optimize_latent, delta=None)

    # ── test set (same dense GT targets as infer_test_set) ───────────────────
    layout = build_run_layout(params)
    casefiles_dir = Path(params["casefiles_dir"])
    casenames_all = load_casenames(casefiles_dir / params["test_casefile"])
    labels_dense, spacings_dense, casenames = _load_labels_dense_per_case(layout, casenames_all)
    if not casenames:
        raise SystemExit("No test cases have a resolvable dense target "
                         f"(test_label_source={layout.test_label_source}).")

    print("=" * 64)
    print(f"[avg-shape] model={params['model_name']} ckpt={params.get('checkpoint','best')} "
          f"cases={len(casenames)}  (zero-latent mean shape, step_size=1)")
    print("=" * 64)

    per_case: List[Dict[str, float]] = []
    for ci, cn in enumerate(casenames):
        r = eval_case_at_resolution(
            net=net, optimize_fn=optimize_fn,
            label_dense=labels_dense[ci], spacing_dense=spacings_dense[ci],
            step_size=1, step_axis=step_axis,          # dense baseline -> centroid on true GT
            params=params, device=device,
            use_thick_slices=params.get("use_thick_slices", False),
            mode=params.get("sweep_mode", "thin"),
            modality=params.get("sweep_modality", "ct"),
            num_classes=num_classes,
            start=0,
            latent_override=z_zero,                    # <-- decode f(x, z=0), no latent opt
        )
        pc = r["dice"]["per_class"]                    # [ON, Globe, Fat, Recti]
        row = {"casename": cn, **{STRUCT_NAMES[k]: float(pc[k]) for k in range(len(pc))},
               "mean": float(r["dice"]["mean"])}
        per_case.append(row)
        print(f"  {cn:28s} " + " ".join(f"{s}={row[s]:.3f}" for s in STRUCT_NAMES)
              + f"  mean={row['mean']:.3f}")

    agg = _aggregate(per_case)
    print("\n" + "=" * 64)
    print("[avg-shape] ZERO-LATENT mean-shape Dice on the test set")
    for s in STRUCT_NAMES:
        print(f"    {s:6s}: {agg[s]:.4f}")
    print(f"    {'macro':6s}: {agg['macro']:.4f}   (mean over structures; n={agg['n_cases']} cases)")
    print("=" * 64)

    if out_csv:
        p = Path(out_csv)
        p.parent.mkdir(parents=True, exist_ok=True)
        cols = ["casename", *STRUCT_NAMES, "mean"]
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in per_case:
                w.writerow({k: r.get(k) for k in cols})
            w.writerow({"casename": "MACRO", **{s: round(agg[s], 5) for s in STRUCT_NAMES},
                        "mean": round(agg["macro"], 5)})
        print(f"[avg-shape] wrote per-case Dice + MACRO row -> {p}")
    return agg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-p", "--paths", help="paths.yaml (data locations)")
    ap.add_argument("-t", "--train_config", help="training config (latent_dim, num_classes, ...)")
    ap.add_argument("-c", "--config", help="test config (test_casefile, test_label_source, ...)")
    ap.add_argument("-m", "--model_name", help="model dir under model_basedir")
    ap.add_argument("--checkpoint", default="best", choices=["best", "latest"])
    ap.add_argument("--test-label-source", default=None,
                    choices=["atlas_gt", "nnunet_pred", "real_pair"],
                    help="Dice target source. atlas_gt (default in test yaml) = manual GT ceiling.")
    ap.add_argument("--test-casefile", default=None, help="override test_casefile (subset list)")
    ap.add_argument("--experiment", default=None, choices=["thin", "thick", "real", "fov"])
    ap.add_argument("--out-csv", default=None, help="write per-case + MACRO Dice CSV")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return _selftest()
    for req in ("paths", "train_config", "config", "model_name"):
        if getattr(args, req) in (None, ""):
            ap.error(f"--{req.replace('_', '-')} is required (or use --self-test).")

    import yaml
    _HERE = Path(__file__).resolve()
    sys.path.insert(0, str(_HERE.parents[1]))                  # orbital_shape_prior_st1
    sys.path.insert(0, str(_HERE.parents[2]))                  # repo root

    with open(args.paths) as f:
        params = yaml.safe_load(f)
    with open(args.train_config) as f:
        params.update(yaml.safe_load(f))
    with open(args.config) as f:
        params.update(yaml.safe_load(f))
    params["model_name"] = args.model_name
    params["checkpoint"] = args.checkpoint
    if args.test_label_source is not None:
        params["test_label_source"] = args.test_label_source
    if args.test_casefile is not None:
        params["test_casefile"] = args.test_casefile
    if args.experiment is not None:
        params["experiment"] = args.experiment

    evaluate_average_shape(params, out_csv=args.out_csv)
    return 0


def _selftest() -> int:
    # aggregation: macro over cases per structure, then mean over structures.
    per_case = [
        {"casename": "a_OD", "ON": 0.2, "Globe": 0.8, "Fat": 0.6, "Recti": 0.4, "mean": 0.5},
        {"casename": "a_OS", "ON": 0.4, "Globe": 0.9, "Fat": 0.7, "Recti": 0.5, "mean": 0.625},
        {"casename": "b_OD", "ON": 0.0, "Globe": 0.7, "Fat": 0.5, "Recti": 0.3, "mean": 0.375},
    ]
    agg = _aggregate(per_case)
    assert abs(agg["ON"] - 0.2) < 1e-9 and abs(agg["Globe"] - 0.8) < 1e-9
    assert abs(agg["Fat"] - 0.6) < 1e-9 and abs(agg["Recti"] - 0.4) < 1e-9
    assert abs(agg["macro"] - (0.2 + 0.8 + 0.6 + 0.4) / 4) < 1e-9 and agg["n_cases"] == 3
    print("aggregate:", {k: round(v, 4) if isinstance(v, float) else v for k, v in agg.items()})
    print("AVERAGE-SHAPE-DICE SELF-TEST PASSED (aggregation)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
