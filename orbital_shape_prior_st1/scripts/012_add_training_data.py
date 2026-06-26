#!/usr/bin/env python3
"""
Step 01.2: Align the addon list, then rebuild train/val from the metadata.

(Sub-step of data prep / step 01: process the NEW masks into the pool.)

This is the incremental analog of 01_prepare_data.py, which does two things:
  (1) align_dataset(...)               -> write canonical patches + metadata
  (2) generate_train_test_split(...)   -> derive the casefiles from metadata

Here we do the same two phases, but:
  (1) align ONLY the scans in the manifest from 011_build_addon_list.py, writing
      them into the SAME aligned_dir/labels + metadata tree the model trains
      from (existing patches are untouched -- additive); and
  (2) rebuild train_cases.txt + val_cases.txt by scanning the WHOLE
      aligned_dir/metadata/ (so previously-aligned data is always re-listed,
      never dropped), holding test_cases.txt fixed.

Patch size is inferred from the existing metadata so the new patches land in the
exact physical frame the model was trained on (canonical_align.infer_patch_size_mm).

Usage:
    python scripts/012_add_training_data.py -p configs/paths.yaml \
        --manifest <casefiles_dir>/train_addon_manifest.csv
    # align only, leave casefiles alone (rebuild later with 013):
    python scripts/012_add_training_data.py -p configs/paths.yaml \
        --manifest <...> --no-rebuild
    # preview the rebuilt split without writing casefiles:
    python scripts/012_add_training_data.py -p configs/paths.yaml \
        --manifest <...> --dry-run
"""

import argparse
import sys
from pathlib import Path

# Make the project root importable so `data_prep` resolves even when this
# script is invoked directly (python scripts/012_add_training_data.py) without
# PYTHONPATH set (the run_*.sh wrappers export it; this is a fallback).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from data_prep.canonical_align import align_dataset, infer_patch_size_mm
from data_prep.build_caselist import rebuild_train_val_from_metadata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--paths", required=True, help="paths.yaml")
    ap.add_argument("--manifest", required=True,
                    help="manifest CSV from 011_build_addon_list.py")
    ap.add_argument("--patch-size", type=float, default=None,
                    help="override patch_size_mm (default: infer from existing "
                         "aligned_dir/metadata)")
    ap.add_argument("--val-fraction", type=float, default=0.10,
                    help="target fraction of CASES (eyes) in val when "
                         "rebuilding the split (default 0.10)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-structures", type=int, default=3,
                    help="metadata cases with fewer structures are excluded "
                         "from the train/val pool")
    ap.add_argument("--no-rebuild", action="store_true",
                    help="align only; do not rebuild train/val casefiles")
    ap.add_argument("--dry-run", action="store_true",
                    help="align, then PREVIEW the rebuilt split without writing "
                         "the casefiles")
    args = ap.parse_args()

    with open(args.paths) as f:
        paths = yaml.safe_load(f)

    aligned_dir = Path(paths["aligned_dir"])
    casefiles_dir = Path(paths["casefiles_dir"])

    if args.patch_size is not None:
        patch_size_mm = args.patch_size
    else:
        patch_size_mm = infer_patch_size_mm(aligned_dir / "metadata")
    print(f"patch_size_mm = {patch_size_mm}")

    # ── Phase 1: align the new masks into aligned_dir ──────────────
    print("=" * 60)
    print("Phase 1: align addon scans from manifest")
    print("=" * 60)
    metas = align_dataset(
        manifest_csv=args.manifest,
        output_dir=str(aligned_dir),
        patch_size_mm=patch_size_mm,
    )
    print(f"\nAligned {len(metas)} new patch(es): "
          f"{sorted(m.casename for m in metas)}")

    if args.no_rebuild:
        print("\n[--no-rebuild] casefiles untouched. Run "
              "013_resplit_train_val.py to rebuild train/val.")
        return

    # ── Phase 2: rebuild train/val from the WHOLE metadata tree ────
    print("\n" + "=" * 60)
    print("Phase 2: rebuild train/val from metadata "
          f"(test fixed{'; DRY-RUN' if args.dry_run else ''})")
    print("=" * 60)
    rebuild_train_val_from_metadata(
        aligned_dir=str(aligned_dir),
        casefiles_dir=str(casefiles_dir),
        val_fraction=args.val_fraction,
        seed=args.seed,
        min_structures=args.min_structures,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
