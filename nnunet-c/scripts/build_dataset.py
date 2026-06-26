#!/usr/bin/env python3
"""CLI: build a corrector control's nnUNet raw dataset.

Usage:
    python nnunet-c/scripts/build_dataset.py --control B
    python nnunet-c/scripts/build_dataset.py --control C --splits train

Writes ${nnUNet_raw}/Dataset{ID}_{NAME}/{imagesTr,labelsTr,dataset.json}.
Control A is external (Dataset835) and is rejected by the builder.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `lib.*` / `engine.*` importable (nnunet-c/ on path) ...
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.config import add_repo_to_syspath  # noqa: E402

# ... and `nnunet.*` importable (repo root on path).
add_repo_to_syspath(__file__)

from engine.build_dataset import build_control  # noqa: E402

_DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "corrector.yaml"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(_DEFAULT_CONFIG),
                    help="corrector.yaml (default: %(default)s)")
    ap.add_argument("--control", required=True, choices=["A", "B", "C", "a", "b", "c"],
                    help="which control to build (B/C; A is external)")
    ap.add_argument("--splits", default="train",
                    help="comma-separated splits to stage; only 'train' is "
                         "assembled into imagesTr/labelsTr (default: train)")
    ap.add_argument("--raw-root", default=None,
                    help="override $nnUNet_raw")
    args = ap.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    build_control(
        config_path=args.config,
        control_name=args.control,
        caller_file=__file__,
        splits=splits,
        raw_root=args.raw_root,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
