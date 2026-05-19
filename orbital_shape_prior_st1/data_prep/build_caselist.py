"""
Generate train/val/test split casename files.

Splitting strategy:
    - Atlas cases (manual GT) → always TEST
    - Checklist cases (nnUNet predictions) → TRAIN / VAL
    - Checklist split by PATIENT (source_id), not by eye
    - Both OD and OS from same patient go to same split
    - This prevents data leakage (same patient's eyes are correlated)

When val_fraction > 0, produces train_cases.txt + val_cases.txt + test_cases.txt.
When val_fraction == 0, produces train_cases.txt + test_cases.txt.
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


def generate_train_test_split(
    aligned_dir: str,
    output_dir: str,
    test_fraction: float = 0.2,
    val_fraction: float = 0.0,
    seed: int = 42,
    min_structures: int = 3,
    atlas_to_test: bool = True,
) -> Tuple[List[str], Optional[List[str]], List[str]]:
    """
    Generate casename files.

    When atlas_to_test=True (default):
        - All atlas_* cases → test_cases.txt
        - Checklist (chk_*) cases → train (+ val if val_fraction > 0)
        - test_fraction is ignored (test = all atlas cases)

    When atlas_to_test=False:
        - All cases mixed, split by patient as before
    """
    meta_dir = Path(aligned_dir) / "metadata"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_meta = []
    for json_path in sorted(meta_dir.glob("*.json")):
        with open(json_path) as f:
            all_meta.append(json.load(f))

    valid_meta = [m for m in all_meta if m["num_structures_found"] >= min_structures]
    print(f"Cases with >= {min_structures} structures: "
          f"{len(valid_meta)}/{len(all_meta)}")

    if atlas_to_test:
        atlas_meta = [m for m in valid_meta if m["source"] == "atlas"]
        chk_meta = [m for m in valid_meta if m["source"] != "atlas"]
        print(f"  Atlas (→ test): {len(atlas_meta)} cases")
        print(f"  Checklist (→ train/val/test): {len(chk_meta)} cases")

        # Split checklist by patient into train / val / test
        chk_patients = sorted(set(m["source_id"] for m in chk_meta))
        n_chk = len(chk_patients)
        n_test_chk = max(1, round(n_chk * test_fraction))
        n_val = max(1, round(n_chk * val_fraction)) if val_fraction > 0 else 0

        rng = np.random.RandomState(seed)
        shuffled = rng.permutation(chk_patients)
        test_patients = set(shuffled[:n_test_chk])
        val_patients = set(shuffled[n_test_chk:n_test_chk + n_val])
        train_patients = set(shuffled[n_test_chk + n_val:])

        train_cases = sorted([m["casename"] for m in chk_meta
                              if m["source_id"] in train_patients])
        val_cases = (sorted([m["casename"] for m in chk_meta
                             if m["source_id"] in val_patients])
                     if n_val > 0 else None)
        # Test = checklist test patients + ALL atlas cases
        test_cases = sorted(
            [m["casename"] for m in chk_meta if m["source_id"] in test_patients]
            + [m["casename"] for m in atlas_meta]
        )
    else:
        # Original behavior: mix all sources, split by patient
        patient_ids = sorted(set(m["source_id"] for m in valid_meta))
        n_patients = len(patient_ids)
        n_test = max(1, round(n_patients * test_fraction))
        n_val = max(1, round(n_patients * val_fraction)) if val_fraction > 0 else 0

        rng = np.random.RandomState(seed)
        shuffled = rng.permutation(patient_ids)
        test_patients = set(shuffled[:n_test])
        val_patients = set(shuffled[n_test:n_test + n_val])
        train_patients = set(shuffled[n_test + n_val:])

        def _collect(patient_set):
            return sorted([m["casename"] for m in valid_meta
                           if m["source_id"] in patient_set])

        train_cases = _collect(train_patients)
        val_cases = _collect(val_patients) if n_val > 0 else None
        test_cases = _collect(test_patients)

    # Write files
    outputs = [("train_cases.txt", train_cases), ("test_cases.txt", test_cases)]
    if val_cases is not None:
        outputs.append(("val_cases.txt", val_cases))

    for fname, cases in outputs:
        with open(output_dir / fname, "w") as f:
            f.write("\n".join(cases) + "\n")
        n_pat = len(set(c.rsplit("_", 1)[0] for c in cases))
        print(f"  {fname}: {len(cases)} cases ({n_pat} patients)")

    return train_cases, val_cases, test_cases