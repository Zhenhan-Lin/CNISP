#!/usr/bin/env python3
"""
Step 6: Align the addon list and append the new cases to train_cases.txt.

Reuses the existing canonical-align pipeline (data_prep.canonical_align) to crop
per-eye canonical patches for every scan in the manifest produced by
scripts/05_build_addon_list.py, writing them into the SAME aligned_dir/labels +
metadata tree the model already trains from. The resulting casenames are then
appended (deduped + sorted) to the training casefile.

Patch size is inferred from the existing metadata so the new patches land in the
exact physical frame the model was trained on (see canonical_align.infer_patch_size_mm).

Cases that fail alignment, fall below --min-structures, or already appear in
val/test are skipped so the addition can't introduce patient leakage.

Usage:
    python scripts/06_add_training_data.py -p configs/paths.yaml \
        --manifest <casefiles_dir>/train_addon_manifest.csv
    # dry run (align + report, but don't touch train_cases.txt):
    python scripts/06_add_training_data.py -p configs/paths.yaml \
        --manifest <...> --dry-run
"""

import argparse
import sys
from pathlib import Path

# Make the project root importable so `data_prep` resolves even when this
# script is invoked directly (python scripts/06_add_training_data.py) without
# PYTHONPATH set (the run_*.sh wrappers export it; this is a fallback).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from data_prep.canonical_align import align_dataset, infer_patch_size_mm


def _read_cases(fp: Path):
    if not fp.exists():
        return []
    return [l.strip() for l in fp.read_text().splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--paths", required=True, help="paths.yaml")
    ap.add_argument("--manifest", required=True,
                    help="manifest CSV from 05_build_addon_list.py")
    ap.add_argument("--casefile", default="train_cases.txt",
                    help="training casefile under casefiles_dir to append to")
    ap.add_argument("--patch-size", type=float, default=None,
                    help="override patch_size_mm (default: infer from existing "
                         "aligned_dir/metadata)")
    ap.add_argument("--min-structures", type=int, default=3,
                    help="skip aligned cases with fewer labelled structures")
    ap.add_argument("--dry-run", action="store_true",
                    help="align + report, but do not modify the casefile")
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

    print("=" * 60)
    print("Aligning addon scans from manifest")
    print("=" * 60)
    metas = align_dataset(
        manifest_csv=args.manifest,
        output_dir=str(aligned_dir),
        patch_size_mm=patch_size_mm,
    )

    # Leakage guard: never add a case already held out for val/test.
    val_test = set(_read_cases(casefiles_dir / "val_cases.txt")) | \
        set(_read_cases(casefiles_dir / "test_cases.txt"))

    new_cases = []
    dropped_structs, dropped_leak = [], []
    for m in metas:
        if m.num_structures_found < args.min_structures:
            dropped_structs.append(m.casename)
            continue
        if m.casename in val_test:
            dropped_leak.append(m.casename)
            continue
        new_cases.append(m.casename)

    casefile = casefiles_dir / args.casefile
    existing = _read_cases(casefile)
    merged = sorted(set(existing) | set(new_cases))
    added = sorted(set(new_cases) - set(existing))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  aligned patches:        {len(metas)}")
    if dropped_structs:
        print(f"  dropped (<{args.min_structures} structs): "
              f"{len(dropped_structs)} {dropped_structs}")
    if dropped_leak:
        print(f"  dropped (in val/test):  {len(dropped_leak)} {dropped_leak}")
    print(f"  {args.casefile}: {len(existing)} -> {len(merged)} "
          f"(+{len(added)} new)")
    for c in added:
        print(f"      + {c}")

    if args.dry_run:
        print("\n[dry-run] casefile NOT modified.")
        return

    if added:
        casefile.write_text("\n".join(merged) + "\n")
        print(f"\nWrote {casefile}")
    else:
        print("\nNothing new to add; casefile unchanged.")


if __name__ == "__main__":
    main()
