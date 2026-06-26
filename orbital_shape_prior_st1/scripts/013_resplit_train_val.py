#!/usr/bin/env python3
"""
Step 01.3: Re-split the train/val pool by patient.

(Sub-step of data prep / step 01: redistribute train/val after adding data.)

Pools the current train_cases.txt + val_cases.txt and re-partitions them so that:
  * both eyes of a patient (same source_id, i.e. casename minus the _OD/_OS
    suffix) always land in the SAME split -- no patient is ever split across
    train and val; and
  * val holds ~--val-fraction of all CASES (eyes), not patients.

test_cases.txt is NOT touched. Any pooled case that also appears in test is
dropped from the pool (and reported) so the re-split can't create leakage.

Usage:
    python scripts/013_resplit_train_val.py -p configs/paths.yaml            # 10% val
    python scripts/013_resplit_train_val.py -p configs/paths.yaml --val-fraction 0.1 --seed 42
    python scripts/013_resplit_train_val.py -p configs/paths.yaml --dry-run  # preview only
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import yaml


def _read_cases(fp: Path):
    if not fp.exists():
        return []
    return [l.strip() for l in fp.read_text().splitlines() if l.strip()]


def _patient_of(case: str) -> str:
    # casename = {source_id}_{eye}; eye is the trailing OD/OS token.
    return case.rsplit("_", 1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--paths", required=True, help="paths.yaml")
    ap.add_argument("--val-fraction", type=float, default=0.10,
                    help="target fraction of CASES (eyes) in val (default 0.10)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true",
                    help="report the proposed split without writing files")
    args = ap.parse_args()

    with open(args.paths) as f:
        paths = yaml.safe_load(f)
    casefiles_dir = Path(paths["casefiles_dir"])

    train_fp = casefiles_dir / "train_cases.txt"
    val_fp = casefiles_dir / "val_cases.txt"
    test_fp = casefiles_dir / "test_cases.txt"

    pool = _read_cases(train_fp) + _read_cases(val_fp)
    test = set(_read_cases(test_fp))

    # Leakage guard: drop any pooled case that is also in test.
    leaked = sorted(c for c in pool if c in test)
    if leaked:
        print(f"WARN dropping {len(leaked)} case(s) present in test_cases.txt: "
              f"{leaked}")
    pool = sorted(set(c for c in pool if c not in test))

    # Group eyes by patient.
    by_patient = defaultdict(list)
    for c in pool:
        by_patient[_patient_of(c)].append(c)
    patients = sorted(by_patient)
    total_cases = len(pool)
    target_val = int(round(total_cases * args.val_fraction))
    print(f"Pool: {total_cases} cases / {len(patients)} patients")
    print(f"Target val: ~{args.val_fraction:.0%} -> {target_val} cases")

    # Shuffle patients deterministically, then greedily fill val until the
    # case-count target is reached (patient atomically assigned -> eyes stay
    # together).
    rng = np.random.RandomState(args.seed)
    shuffled = list(rng.permutation(patients))

    val_patients, val_count = [], 0
    for p in shuffled:
        if val_count >= target_val:
            break
        val_patients.append(p)
        val_count += len(by_patient[p])
    val_patients = set(val_patients)

    val_cases = sorted(c for p in val_patients for c in by_patient[p])
    train_cases = sorted(c for p in patients if p not in val_patients
                         for c in by_patient[p])

    print("\n" + "=" * 60)
    print("Proposed split")
    print("=" * 60)
    print(f"  train: {len(train_cases)} cases / "
          f"{len(patients) - len(val_patients)} patients")
    print(f"  val:   {len(val_cases)} cases / {len(val_patients)} patients "
          f"({len(val_cases)/total_cases:.1%} of pool)")
    print(f"  val patients: {sorted(val_patients)}")

    # Sanity: no patient on both sides, every pooled case placed exactly once.
    assert set(train_cases).isdisjoint(val_cases)
    assert len(train_cases) + len(val_cases) == total_cases
    train_pat = {_patient_of(c) for c in train_cases}
    assert train_pat.isdisjoint(val_patients)

    if args.dry_run:
        print("\n[dry-run] files NOT modified.")
        return

    train_fp.write_text("\n".join(train_cases) + "\n")
    val_fp.write_text("\n".join(val_cases) + "\n")
    print(f"\nWrote {train_fp}")
    print(f"Wrote {val_fp}")
    print("test_cases.txt left unchanged.")


if __name__ == "__main__":
    main()
