#!/usr/bin/env python3
"""CLI entry: nnUNet Taubin post-processing control (mask gen + native Dice).

Thin orchestration wrapper. All functionality lives in
``nnunet.engine.interpolate_native`` (the embedded Taubin smoother, the
degraded->native generation pass, and the standalone Dice summary).

This control is nnUNet-only: it Taubin-smooths the degraded-grid nnUNet
prediction, resamples it (order=0) onto the native CT grid, and Dices it
against the native GT -- a baseline for comparison against CNISP. It is
surfaced both as a standalone summary (this script, --mode summarize) and as
an extra ``nnUNet-interp`` column in ``nnunet/compare_native.py``.

Usage
-----
    # generate masks + standalone summary (default --mode all)
    python nnunet/interpolate_native.py --config nnunet/configs.yaml \\
        --experiment thick

    # only (re)generate the smoothed native masks
    python nnunet/interpolate_native.py --config nnunet/configs.yaml \\
        --experiment thin --mode generate --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make ``nnunet.*`` importable when run as ``python nnunet/interpolate_native.py``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nnunet.engine.interpolate_native import run  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="nnunet/configs.yaml")
    ap.add_argument("--work-dir", default=None,
                    help="Override work_dir from the config.")
    ap.add_argument("--experiment", choices=["thin", "thick", "real"],
                    default="thin",
                    help="Experiment layer: reads prediction/<experiment>/ "
                         "sparse_step_XX(/_native) and writes "
                         "prediction/<experiment>/interpolation/.")
    ap.add_argument("--split", choices=["test", "train"], default="test",
                    help="'test' (default) uses work_dir/; 'train' uses "
                         "work_dir/train_split/ (no step_01 dense baseline).")
    ap.add_argument("--mode", choices=["generate", "summarize", "all"],
                    default="all",
                    help="generate: write smoothed native masks; summarize: "
                         "Dice them vs native GT into the standalone bundle; "
                         "all (default): both.")
    ap.add_argument("--smoothing-factor", type=float, default=0.7,
                    help="Taubin smoothing factor f (Slicer mapping; default "
                         "0.7 -> passband ~1.6e-3, 48 iterations).")
    ap.add_argument("--force", action="store_true",
                    help="Recompute smoothed masks even if they already exist.")
    ap.add_argument("--out-dir", default=None,
                    help="Summary output dir. Default: "
                         "${work_dir}/prediction/<exp>/interpolation/summary/.")
    ap.add_argument("--include-source-prefixes", default=None,
                    help="Comma-separated source_id prefixes to keep. Default: "
                         "'viz_include_source_prefixes' from --config.")
    ap.add_argument("--exclude-source-prefixes", default=None,
                    help="Comma-separated source_id prefixes to drop. Default: "
                         "'viz_exclude_source_prefixes' from --config "
                         "(usually 'chk_'). Pass '' to keep ALL sources.")
    return ap


if __name__ == "__main__":
    sys.exit(run(build_parser().parse_args()))
