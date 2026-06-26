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
from typing import Dict, List, Optional, Tuple

import numpy as np


def _read_casefile(fp: Path) -> List[str]:
    if not fp.exists():
        return []
    return [l.strip() for l in fp.read_text().splitlines() if l.strip()]


def rebuild_train_val_from_metadata(
    aligned_dir: str,
    casefiles_dir: str,
    val_fraction: float = 0.10,
    seed: int = 42,
    min_structures: int = 3,
    dry_run: bool = False,
    verbose: bool = True,
) -> Tuple[List[str], List[str]]:
    """Rebuild train_cases.txt + val_cases.txt from the on-disk metadata.

    This mirrors ``generate_train_test_split``'s metadata-driven design (the
    authoritative list of cases is ``aligned_dir/metadata/*.json``, NOT the
    existing .txt files) but with two differences tailored to *incrementally
    adding* training data:

      * ``test_cases.txt`` is treated as FIXED and read verbatim -- every case
        listed there is held out of the train/val pool (and never reassigned),
        so a re-split can't leak a test eye into training nor silently drop the
        test set.
      * The remaining (non-test) cases -- i.e. ALL previously-aligned training
        data plus anything newly added by ``012_add_training_data.py`` -- are
        re-partitioned into train/val. Both eyes (and any variants) of a
        patient share one ``source_id`` and are assigned atomically, and val is
        filled to ~``val_fraction`` of CASES (eyes), not patients.

    Because the pool is derived from metadata, previous data is never lost just
    because it was missing from train_cases.txt/val_cases.txt.

    Returns ``(train_cases, val_cases)`` (each sorted).
    """
    meta_dir = Path(aligned_dir) / "metadata"
    casefiles_dir = Path(casefiles_dir)
    casefiles_dir.mkdir(parents=True, exist_ok=True)

    if not meta_dir.is_dir():
        raise FileNotFoundError(
            f"rebuild_train_val_from_metadata: {meta_dir} not found. Run the "
            f"canonical-align step (01_prepare_data.py / 012_add_training_data.py) "
            f"first so the metadata is on disk."
        )

    test_cases = set(_read_casefile(casefiles_dir / "test_cases.txt"))

    all_meta = []
    for jp in sorted(meta_dir.glob("*.json")):
        with open(jp) as f:
            all_meta.append(json.load(f))

    n_total = len(all_meta)
    valid = [m for m in all_meta
             if m.get("num_structures_found", 0) >= min_structures]
    n_low = n_total - len(valid)

    # Pool = every valid case that is not held out for test.
    pool = [m for m in valid if m["casename"] not in test_cases]

    # Group eyes/variants by patient (source_id).
    by_pat: Dict[str, List[str]] = {}
    for m in pool:
        by_pat.setdefault(m["source_id"], []).append(m["casename"])
    patients = sorted(by_pat)
    total_cases = sum(len(v) for v in by_pat.values())
    target_val = int(round(total_cases * val_fraction))

    # Permute indices (not the array itself) so the patient ids stay plain
    # Python str -- np.random.permutation(list_of_str) would yield np.str_,
    # which prints as np.str_('...') and is confusing in logs.
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(patients)) if patients else []
    shuffled = [patients[i] for i in order]

    val_patients, val_count = set(), 0
    for p in shuffled:
        if val_count >= target_val:
            break
        val_patients.add(p)
        val_count += len(by_pat[p])

    val_cases = sorted(c for p in val_patients for c in by_pat[p])
    train_cases = sorted(c for p in patients if p not in val_patients
                         for c in by_pat[p])

    if verbose:
        print(f"Metadata scan: {n_total} cases "
              f"({n_low} dropped <{min_structures} structs, "
              f"{len(test_cases)} held in test)")
        print(f"  pool (non-test): {total_cases} cases / {len(patients)} patients")
        print(f"  target val ~{val_fraction:.0%} -> {target_val} cases")
        print(f"  train: {len(train_cases)} cases / "
              f"{len(patients) - len(val_patients)} patients")
        print(f"  val:   {len(val_cases)} cases / {len(val_patients)} patients "
              f"({(len(val_cases)/total_cases if total_cases else 0):.1%})")
        print(f"  val patients: {sorted(val_patients)}")

    # Invariants: disjoint, total preserved, no patient straddles the split.
    assert set(train_cases).isdisjoint(val_cases)
    assert len(train_cases) + len(val_cases) == total_cases
    assert {c for c in train_cases} | {c for c in val_cases} == \
        {c for v in by_pat.values() for c in v}

    if not dry_run:
        (casefiles_dir / "train_cases.txt").write_text(
            "\n".join(train_cases) + "\n")
        (casefiles_dir / "val_cases.txt").write_text(
            "\n".join(val_cases) + "\n")
        if verbose:
            print(f"  wrote {casefiles_dir/'train_cases.txt'}")
            print(f"  wrote {casefiles_dir/'val_cases.txt'}")
            print("  test_cases.txt left unchanged.")

    return train_cases, val_cases


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