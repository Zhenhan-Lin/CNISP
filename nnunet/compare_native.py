#!/usr/bin/env python3
"""CLI entry: per-source paired Dice (nnUNet vs one CNISP run), native space.

Thin orchestration wrapper. All functionality lives in
``nnunet.engine.compare_native`` (``run`` plus the Dice/aggregation helpers,
some of which are reused by ``nnunet/build_nnunet_native_summary.py``).

Usage
-----
    python nnunet/compare_native.py --config nnunet/configs.yaml \\
        --cnisp-run-tag atlas_gt
    python nnunet/compare_native.py --config nnunet/configs.yaml \\
        --cnisp-run-tag nnunet_pred
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make ``nnunet.*`` importable when run as ``python nnunet/compare_native.py``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nnunet.engine.compare_native import run  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="nnunet/configs.yaml")
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--cnisp-run-tag", default="atlas_gt",
                    help="Which CNISP run to compare against (subdir under "
                         "output_basedir/<model>/runs/<experiment>/). Default "
                         "atlas_gt preserves the ceiling-curve comparison.")
    ap.add_argument("--experiment", choices=["thin", "thick", "real"],
                    default="thin",
                    help="Experiment directory layer (thin|thick|real). "
                         "Reads CNISP masks from runs/<experiment>/<run-tag>/ "
                         "and nnUNet sparse preds from prediction/<experiment>/"
                         ", and exp-suffixes the output CSVs so thin/thick "
                         "comparisons coexist.")
    ap.add_argument("--cnisp-method-label", default=None,
                    help="Override the CNISP method label. If unset, look up "
                         "cnisp_runs_to_compare in the config.")
    ap.add_argument("--out-suffix", default=None,
                    help="Suffix for output filenames. Default is "
                         "'__<cnisp_run_tag>__<experiment>' so multiple runs "
                         "do not collide.")
    ap.add_argument("--strict-shape", action="store_true",
                    help="Fail if a prediction's shape differs from GT "
                         "(default: skip the source with a warning).")
    return ap


if __name__ == "__main__":
    sys.exit(run(build_parser().parse_args()))
