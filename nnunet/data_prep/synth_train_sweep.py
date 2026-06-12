#!/usr/bin/env python3
"""Synthesize a ``sweep_results.pkl``-compatible step grid for the train split.

Why this exists
---------------
``sparsify_inputs.py`` is normally driven by a CNISP test-time
``sweep_results.pkl`` that lists, per ``(casename, step)``, the
effective resolution and canonical step axis CNISP actually evaluated.
For the v6 nnUNet-obs data-gen we want to degrade the MODELING scans
(``train_cases.txt`` ∪ ``val_cases.txt``) along the SAME step grid the
CNISP degradation bank uses at training time -- but there is no CNISP
inference run for those scans, so no pickle exists.

This script builds that pickle directly from each case's canonical GT
patch geometry:

* through-plane spacing  = ``argmax`` of the GT patch column norms,
* step list              = ``adaptive_steps_for_bank`` (the SAME function
                           the training degradation bank uses), so the
                           nnUNet-obs items line up 1:1 with the GT bank
                           items per ``(scan, mode, step)``,
* ``step==1`` is DROPPED  (nnUNet predicts on DEGRADED images only; the
                           dense baseline is never an nnUNet output here),
* ``effective_resolution_mm = spacing[axis] * step``.

The emitted rows carry exactly the keys ``sparsify_inputs._build_sweep_set``
reads: ``casename``, ``step_size``, ``effective_resolution_mm``,
``step_axis``.

Usage
-----
    python nnunet/synth_train_sweep.py --config nnunet/configs.yaml \
        [--train-config orbital_shape_prior_st1/configs/train_sty2.yaml] \
        [--out <path>]  [--increment-mm 2.0] [--max-eff-res-mm 12.0]
"""

from __future__ import annotations

import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import nibabel as nib
import numpy as np

# Make ``nnunet.*`` importable when run as ``python nnunet/...``.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nnunet.helpers.config import load_yaml  # noqa: E402
from orbital_shape_prior_st1.engine.dataset import (  # noqa: E402
    adaptive_steps_for_bank,
)


def _load_casenames(path: Path) -> List[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def run(args) -> int:
    cfg = load_yaml(Path(args.config))
    cnisp_paths = load_yaml(Path(cfg["cnisp_paths_yaml"]))
    work_dir = Path(cfg["work_dir"]) / "train_split"
    aligned_dir = Path(cnisp_paths["aligned_dir"])
    labels_dir = aligned_dir / cnisp_paths.get("labels_dirname", "labels")
    casefiles_dir = Path(cnisp_paths["casefiles_dir"])

    # Step-grid knobs: CLI > train-config bank > defaults.
    inc = args.increment_mm
    max_eff = args.max_eff_res_mm
    if (inc is None or max_eff is None) and args.train_config:
        tcfg = load_yaml(Path(args.train_config))
        bank = tcfg.get("degradation_bank", {}) or {}
        if inc is None:
            inc = bank.get("target_eff_res_increment_mm")
        if max_eff is None:
            max_eff = bank.get("max_eff_resolution_mm")
    inc = 2.0 if inc is None else float(inc)
    max_eff = 12.0 if max_eff is None else float(max_eff)

    casenames: List[str] = []
    for fn in ("train_cases.txt", "val_cases.txt"):
        p = casefiles_dir / fn
        if p.exists():
            casenames.extend(_load_casenames(p))
        else:
            print(f"[synth_train_sweep] WARN: casefile missing: {p}",
                  file=sys.stderr)
    casenames = sorted(set(casenames))

    print(f"[synth_train_sweep] modeling cases:     {len(casenames)}")
    print(f"[synth_train_sweep] labels_dir:         {labels_dir}")
    print(f"[synth_train_sweep] step grid: increment={inc} mm, "
          f"max_eff_res={max_eff} mm (drop step==1)")

    rows: List[dict] = []
    per_step_count: Dict[int, int] = defaultdict(int)
    n_missing = 0
    issues: List[str] = []
    for cn in casenames:
        patch = labels_dir / f"{cn}.nii.gz"
        if not patch.exists():
            n_missing += 1
            issues.append(f"{cn}: GT patch missing at {patch}")
            continue
        aff = nib.load(str(patch)).affine
        spacing = np.sqrt((aff[:3, :3] ** 2).sum(axis=0))
        step_axis = int(np.argmax(spacing))
        spacing_axis = float(spacing[step_axis])
        steps = [s for s in adaptive_steps_for_bank(spacing_axis, inc, max_eff)
                 if s >= 2]
        for step in steps:
            rows.append({
                "casename": cn,
                "step_size": int(step),
                "effective_resolution_mm": float(spacing_axis * step),
                "step_axis": step_axis,
            })
            per_step_count[step] += 1

    if any(r["step_size"] <= 1 for r in rows):  # invariant guard
        raise SystemExit("[synth_train_sweep] BUG: step==1 row leaked in.")

    out_path = Path(args.out) if args.out else (
        work_dir / "synth_sweep_results.pkl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(rows, f)

    if issues:
        print(f"\n[synth_train_sweep] {len(issues)} issue(s):", file=sys.stderr)
        for line in issues[:25]:
            print(f"  - {line}", file=sys.stderr)
        if len(issues) > 25:
            print(f"  ... and {len(issues) - 25} more", file=sys.stderr)

    print(f"\n[synth_train_sweep] wrote {len(rows)} row(s) over "
          f"{len(casenames) - n_missing} case(s); {n_missing} missing GT.")
    print(f"[synth_train_sweep] steps present: "
          f"{dict(sorted(per_step_count.items()))}")
    print(f"[synth_train_sweep] pickle: {out_path}")
    return 0 if rows else 3
