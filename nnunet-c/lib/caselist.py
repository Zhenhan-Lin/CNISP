"""Split/caselist handling + leakage asserts for the corrector experiment.

  * corrector training set  = source_ids from nnunet-c/splits/corrector_train.txt
  * CNISP train/val/test    = casefiles_dir/{train,val,test}_cases.txt (casenames)
  * test set for ALL controls = CNISP test_cases.txt

Hard guarantees (assert at startup, abort on violation):
  * corrector_train ∩ CNISP_train = ∅
  * corrector_train ∩ CNISP_test  = ∅
  * checked at the patient (source_id) level.

Also derives a casenames file (corrector_train_cases.txt) under casefiles_dir so
the casename-based CNISP/nnUNet machinery can consume the corrector split.

Depends only on stdlib.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set


def read_source_ids(path: Path) -> List[str]:
    """Read a source_id-per-line file; ignore '#' comments and blank lines."""
    out: List[str] = []
    with open(path) as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if line:
                out.append(line)
    # de-dup, preserve order
    seen: Set[str] = set()
    uniq = []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


def read_casenames(path: Path) -> List[str]:
    """Read a casename-per-line file (CNISP convention); ignore blanks."""
    if not path.exists():
        raise FileNotFoundError(f"casefile not found: {path}")
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def casename_to_source_id(casename: str) -> str:
    """`chk_14455_OD` -> `chk_14455` (strip the _OD/_OS eye suffix)."""
    if casename.endswith("_OD") or casename.endswith("_OS"):
        return casename[:-3]
    return casename


def cnisp_sources(cfg: Dict, which: str) -> Set[str]:
    """Source_id set from CNISP {train,val,test}_cases.txt."""
    casefiles_dir: Path = cfg["_resolved"]["casefiles_dir"]
    fname = {"train": "train_cases.txt", "val": "val_cases.txt",
             "test": "test_cases.txt"}[which]
    return {casename_to_source_id(c)
            for c in read_casenames(casefiles_dir / fname)}


def corrector_train_sources(cfg: Dict) -> List[str]:
    """Source_ids for the corrector training split."""
    split_path: Path = cfg["_resolved"]["corrector_train_split"]
    sids = read_source_ids(split_path)
    if not sids:
        raise RuntimeError(
            f"no source_ids in {split_path}; fill it with the corrector "
            f"training source_ids (one per line) before building."
        )
    return sids


def test_sources(cfg: Dict) -> List[str]:
    """Source_ids for the unified test set (= CNISP test_cases.txt)."""
    return sorted(cnisp_sources(cfg, "test"))


def assert_no_leakage(cfg: Dict) -> None:
    """Enforce corrector_train disjoint from CNISP train AND test (patient-level)."""
    corr = set(corrector_train_sources(cfg))
    cn_train = cnisp_sources(cfg, "train")
    cn_test = cnisp_sources(cfg, "test")

    leak_train = sorted(corr & cn_train)
    leak_test = sorted(corr & cn_test)
    msgs = []
    if leak_train:
        msgs.append(
            f"corrector_train ∩ CNISP_train is non-empty ({len(leak_train)}): "
            f"{leak_train[:10]}{' ...' if len(leak_train) > 10 else ''}"
        )
    if leak_test:
        msgs.append(
            f"corrector_train ∩ CNISP_test is non-empty ({len(leak_test)}): "
            f"{leak_test[:10]}{' ...' if len(leak_test) > 10 else ''}"
        )
    if msgs:
        raise RuntimeError(
            "LEAKAGE detected in corrector split (aborting):\n  - "
            + "\n  - ".join(msgs)
        )


def derive_train_casefile(cfg: Dict) -> Path:
    """Expand corrector_train source_ids -> casenames; write to casefiles_dir.

    Returns the path to the written casenames file (corrector_train_cases.txt),
    which the CNISP/nnUNet (casename-based) machinery consumes.
    """
    res = cfg["_resolved"]
    meta_dir: Path = res["metadata_dir"]
    casefiles_dir: Path = res["casefiles_dir"]
    out_path = casefiles_dir / cfg["corrector_train_casefile"]

    casenames: List[str] = []
    for sid in corrector_train_sources(cfg):
        eyes = sorted(p.stem for p in meta_dir.glob(f"{sid}_O*.json"))
        if not eyes:
            raise FileNotFoundError(
                f"no alignment metadata for {sid!r} under {meta_dir}"
            )
        casenames.extend(eyes)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for cn in casenames:
            f.write(cn + "\n")
    return out_path
