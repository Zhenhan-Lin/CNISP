#!/usr/bin/env python3
"""CLI: adapt a Dataset835 checkpoint's first conv from 1 -> N input channels.

Usage:
    python nnunet-c/scripts/adapt_checkpoint.py \
        --in  $nnUNet_results/Dataset835_PHOTON_CT_QAfiltered/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_2/checkpoint_final.pth \
        --out /tmp/checkpoint_835_to5ch.pth \
        --channels 5 --mask-init zero

The adapted checkpoint is then passed to nnUNetv2_train via -pretrained_weights.
Reports adapted layers and loaded-vs-newly-initialized parameter counts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.finetune import adapt_checkpoint, print_report  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_path", required=True,
                    help="source (Dataset835) checkpoint .pth")
    ap.add_argument("--out", dest="out_path", required=True,
                    help="destination adapted checkpoint .pth")
    ap.add_argument("--channels", type=int, default=5,
                    help="target number of input channels (default: 5)")
    ap.add_argument("--mask-init", choices=["zero", "small_random"],
                    default="zero",
                    help="mask-channel init; small_random=x0.01 fallback for "
                         "dead gradients (default: zero)")
    ap.add_argument("--report-json", default=None,
                    help="optional path to dump the adaptation report JSON")
    args = ap.parse_args()

    report = adapt_checkpoint(
        in_path=args.in_path,
        out_path=args.out_path,
        n_new_channels=args.channels,
        mask_init=args.mask_init,
    )
    print_report(report)
    if args.report_json:
        with open(args.report_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  report -> {args.report_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
