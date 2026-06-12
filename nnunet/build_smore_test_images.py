#!/usr/bin/env python3
"""CLI entry: produce SMORE super-resolved CTs for the nnUNet SMORE baseline.

Thin orchestration wrapper. Functionality lives in
``nnunet.engine.build_smore_test_images.run``.

Usage
-----
    python nnunet/build_smore_test_images.py --config nnunet/configs.yaml [...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nnunet.engine.build_smore_test_images import run  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="nnunet/configs.yaml")
    ap.add_argument("--smore-out-root", default=None,
                    help="Override smore_out_root from config")
    ap.add_argument("--cases", default=None,
                    help="Optional path to a casename file (one casename "
                         "or source_id per line). Default: CNISP's test_cases.txt.")

    # SMORE backend / runtime knobs (mirror nnunetv2_build_datasets2.py)
    ap.add_argument("--smore-backend", choices=["local", "container"],
                    default="local")
    ap.add_argument("--smore-sif", default="",
                    help="Path to SMORE SIF (required when backend=container)")
    ap.add_argument("--smore-bind-roots", default="",
                    help="Extra comma-separated bind roots for the container "
                         "backend (in addition to auto-detected ones).")
    ap.add_argument("--smore-gpu-ids", default="",
                    help="Comma-separated GPU ids (e.g. '0,1').")
    ap.add_argument("--smore-gpu-id", type=int, default=0,
                    help="Single-GPU fallback when --smore-gpu-ids is empty.")
    ap.add_argument("--smore-per-gpu-concurrency", type=int, default=1)
    ap.add_argument("--smore-patch-sampling", default="gradient")
    ap.add_argument("--smore-slice-thickness", type=float, default=None)
    ap.add_argument("--smore-blur-kernel-fpath", type=str, default=None)
    ap.add_argument("--smore-suffix", default="_smore")
    ap.add_argument("--smore-on-incompatible",
                    choices=["original", "skip"], default="original",
                    help="What to do if a case fails the SMORE compatibility "
                         "check. 'original' copies the source CT through with "
                         "the SMORE suffix so downstream code stays uniform.")
    # Compat check thresholds (defaults match the existing script).
    ap.add_argument("--smore-min-slice-separation", type=float, default=1.2)
    ap.add_argument("--smore-inplane-atol", type=float, default=1e-2)
    ap.add_argument("--smore-isotropic-eps", type=float, default=1e-3)
    ap.add_argument("--smore-require-unique-worst-axis", action="store_true",
                    default=True)
    return ap


if __name__ == "__main__":
    sys.exit(run(build_parser().parse_args()))
