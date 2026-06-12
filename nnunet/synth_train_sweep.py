#!/usr/bin/env python3
"""CLI entry: synthesize a sweep_results.pkl-compatible step grid (train split).

Thin orchestration wrapper. Functionality lives in
``nnunet.data_prep.synth_train_sweep.run``.

Usage
-----
    python nnunet/synth_train_sweep.py --config nnunet/configs.yaml \\
        [--train-config orbital_shape_prior_st1/configs/train_sty2.yaml] \\
        [--out <path>]  [--increment-mm 2.0] [--max-eff-res-mm 12.0]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nnunet.data_prep.synth_train_sweep import run  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="nnunet/configs.yaml")
    ap.add_argument("--train-config", default=None,
                    help="CNISP train yaml whose degradation_bank knobs "
                         "(target_eff_res_increment_mm, max_eff_resolution_mm) "
                         "define the step grid. Falls back to the bank "
                         "defaults / the CLI overrides when omitted.")
    ap.add_argument("--out", default=None,
                    help="Output pickle path. Default: "
                         "<work_dir>/train_split/synth_sweep_results.pkl")
    ap.add_argument("--increment-mm", type=float, default=None,
                    help="Override target_eff_res_increment_mm (else from "
                         "train-config bank, else 2.0).")
    ap.add_argument("--max-eff-res-mm", type=float, default=None,
                    help="Override max_eff_resolution_mm (else from "
                         "train-config bank, else 12.0).")
    return ap


if __name__ == "__main__":
    sys.exit(run(build_parser().parse_args()))
