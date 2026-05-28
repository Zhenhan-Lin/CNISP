#!/usr/bin/env python3
"""
Step 1: Prepare canonical-aligned orbital patches.

Usage:
    python scripts/01_prepare_data.py -p configs/paths.yaml -c configs/train_strategyB.yaml
    python scripts/01_prepare_data.py -p configs/paths.yaml   # val_fraction defaults to 0
"""

import argparse
import yaml

from data_prep.canonical_align import align_dataset
from data_prep.build_caselist import generate_train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--paths", required=True, help="paths.yaml")
    parser.add_argument("-c", "--config", default=None, help="train config yaml (reads val_fraction, test_fraction)")
    parser.add_argument("--patch_size", type=float, default=80.0)
    args = parser.parse_args()

    with open(args.paths) as f:
        paths = yaml.safe_load(f)

    # Read split fractions from train config if provided
    train_cfg = {}
    if args.config:
        with open(args.config) as f:
            train_cfg = yaml.safe_load(f) or {}

    test_fraction = train_cfg.get("test_fraction", 0.2)
    val_fraction = train_cfg.get("val_fraction", 0.0)
    atlas_to_test = train_cfg.get("atlas_to_test", True)

    # Step 1: Canonical alignment (checklist + atlas)
    print("=" * 60)
    print("STEP 1: Canonical alignment")
    print("=" * 60)
    align_dataset(
        checklist_csv=paths.get("checklist_csv"),
        atlas_label_dir=paths.get("atlas_label_dir"),
        output_dir=paths["aligned_dir"],
        patch_size_mm=args.patch_size,
    )

    # Step 2: Train/val/test split (by patient)
    # (alignment QC report can be regenerated separately with
    #  data_prep.alignment_qc.compute_alignment_stats; not run here.)
    print("\n" + "=" * 60)
    print("STEP 2: Train/val/test split")
    print("=" * 60)
    generate_train_test_split(
        aligned_dir=paths["aligned_dir"],
        output_dir=paths["casefiles_dir"],
        test_fraction=test_fraction,
        val_fraction=val_fraction,
        atlas_to_test=atlas_to_test,
    )

    print(f"\nDone. Patches: {paths['aligned_dir']}")


if __name__ == "__main__":
    main()