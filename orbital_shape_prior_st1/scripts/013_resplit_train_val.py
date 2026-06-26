#!/usr/bin/env python3
"""
Step 01.3: Re-split train/val from the metadata, by patient.

(Sub-step of data prep / step 01: redistribute train/val without re-aligning.)

This is the split-only counterpart of 012_add_training_data.py's Phase 2 (and
the analog of generate_train_test_split in 01_prepare_data.py). It does NOT
align anything; it just rebuilds train_cases.txt + val_cases.txt by scanning
the WHOLE aligned_dir/metadata/ tree so that:

  * every previously-aligned case (plus anything added by 012) is re-listed --
    nothing is dropped just because it was missing from a .txt file;
  * both eyes (and any variants) of a patient (same source_id) always land in
    the SAME split; and
  * val holds ~--val-fraction of all CASES (eyes), not patients.

test_cases.txt is read verbatim and left unchanged; its cases are held out of
the train/val pool (leakage guard).

Usage:
    python scripts/013_resplit_train_val.py -p configs/paths.yaml            # 10% val
    python scripts/013_resplit_train_val.py -p configs/paths.yaml --val-fraction 0.1 --seed 42
    python scripts/013_resplit_train_val.py -p configs/paths.yaml --dry-run  # preview only
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from data_prep.build_caselist import rebuild_train_val_from_metadata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--paths", required=True, help="paths.yaml")
    ap.add_argument("--val-fraction", type=float, default=0.10,
                    help="target fraction of CASES (eyes) in val (default 0.10)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-structures", type=int, default=3,
                    help="metadata cases with fewer structures are excluded "
                         "from the train/val pool")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the proposed split without writing files")
    args = ap.parse_args()

    with open(args.paths) as f:
        paths = yaml.safe_load(f)

    rebuild_train_val_from_metadata(
        aligned_dir=paths["aligned_dir"],
        casefiles_dir=paths["casefiles_dir"],
        val_fraction=args.val_fraction,
        seed=args.seed,
        min_structures=args.min_structures,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
