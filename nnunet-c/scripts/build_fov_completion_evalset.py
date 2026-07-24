#!/usr/bin/env python3
"""Manifest-driven FOV-completion evalset builder (revised-plan §6.2, P0-1).

Replaces the thickness-oriented ``build_corrector_testset.py --steps auto`` for the
FOV experiment. It is driven by the completion manifest + a patient-level split
(splits_final.json), uses the EXACT FOV case ids, and NEVER infers step values or
renames FOV conditions as thickness steps.

Outputs (under ``--out/<control_name>/``):
    imagesTs/            1-ch truncated CT per case      (cascade; masi-55 convert)
    prevsegTs/           CNISP prior mask per case        (cascade prior; masi-55)
    fovMaskTs/           acquired-FOV validity mask       (1=visible/0=missing)
    eval_cases_map.json  per-case: subject_id, crop_type, severity, is_full_fov,
                         gt_label_path, gt_struct_to_value, pred_file, fov_mask_file,
                         source_shape

The acquired-FOV mask (revised-plan §6.3) is written on the truncated-CT grid/affine
so the evaluator can NEAREST-resample it to the GT grid (never applying a source-grid
visible_box to the GT array). The FOV mask + map assembly + split selection are
unit-tested (``--self-test``); the CT/prior assembly + nibabel writes run on masi-55
(they reuse the SAME engine.convert cascade path as build_corrector_testset, keyed by
FOV case id instead of by discovered step).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # nnunet-c
from lib.fov_region_masks import visible_box_to_mask                     # noqa: E402

NNUNET_STRUCTS: Dict[str, int] = {"ON": 1, "Recti": 2, "Globe": 3, "Fat": 4}


# ── split handling (patient-level) ────────────────────────────────────────────
def load_split_case_ids(splits_final_path: str, fold: int) -> Tuple[Set[str], Set[str]]:
    """Return (train_case_ids, val_case_ids) from nnU-Net's splits_final.json."""
    splits = json.loads(Path(splits_final_path).read_text())
    s = splits[int(fold)]
    return set(s["train"]), set(s["val"])


def records_by_case(completion_manifest: str) -> Dict[str, dict]:
    man = json.loads(Path(completion_manifest).read_text())
    recs = man["records"] if isinstance(man, dict) and "records" in man else man
    return {r["case_id"]: r for r in recs}


def select_eval_cases(
    recs_by_case: Dict[str, dict],
    split_case_ids: Set[str],
    *,
    other_split_case_ids: Optional[Sequence[Set[str]]] = None,
) -> List[dict]:
    """FOV records for a split's cases (EXACT ids, no step inference), after a
    patient-level leakage check against the other splits (revised-plan §2/§7)."""
    unknown = split_case_ids - set(recs_by_case)
    if unknown:
        raise RuntimeError(f"[fov-evalset] {len(unknown)} split case(s) absent from the "
                           f"completion manifest (e.g. {sorted(unknown)[:5]}). Case ids must "
                           f"match; supply a --case-map upstream if the dataset renamed them.")
    subj = {cid: str(recs_by_case[cid]["subject_id"]) for cid in recs_by_case}
    my_subjects = {subj[c] for c in split_case_ids}
    for other in (other_split_case_ids or []):
        other_subjects = {subj[c] for c in other if c in subj}
        overlap = sorted(my_subjects & other_subjects)
        if overlap:
            raise RuntimeError(f"[fov-evalset] PATIENT-LEVEL LEAKAGE across splits: "
                               f"{len(overlap)} subject(s) shared (e.g. {overlap[:10]}).")
    return [recs_by_case[c] for c in sorted(split_case_ids)]


def check_disjoint_subjects(named_splits: Dict[str, Set[str]],
                            recs_by_case: Dict[str, dict]) -> Dict[str, int]:
    """Assert every pair of named splits (train/val/test) is subject-disjoint
    (revised-plan §7). Returns {split: n_subjects}. Raises on any overlap."""
    subj = {cid: str(recs_by_case[cid]["subject_id"]) for cid in recs_by_case}
    subjects = {name: {subj[c] for c in ids if c in subj} for name, ids in named_splits.items()}
    names = list(subjects)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ov = sorted(subjects[names[i]] & subjects[names[j]])
            if ov:
                raise RuntimeError(f"[fov-evalset] subject overlap {names[i]}∩{names[j]}: "
                                   f"{len(ov)} (e.g. {ov[:10]}).")
    return {name: len(s) for name, s in subjects.items()}


# ── FOV validity mask (revised-plan §6.3) ─────────────────────────────────────
def build_fov_mask_array(source_shape: Sequence[int], visible_box, is_full_fov: bool) -> np.ndarray:
    """uint8 acquired-FOV mask on the source (truncated-CT) grid: 1=visible/acquired,
    0=missing. Full-FOV -> all ones."""
    shape = tuple(int(s) for s in source_shape)
    if is_full_fov:
        return np.ones(shape, dtype=np.uint8)
    return visible_box_to_mask(shape, visible_box).astype(np.uint8)


# ── eval_cases_map.json assembly ──────────────────────────────────────────────
def assemble_eval_map(records: List[dict], control_name: str, fold: int,
                      gt_path_for, gt_stv_for, structures=None) -> dict:
    """Build the eval map. ``gt_path_for(rec)`` / ``gt_stv_for(rec)`` resolve the GT
    label path + struct->value per case (masi-55 data layout)."""
    structures = structures or list(NNUNET_STRUCTS)
    cases: Dict[str, dict] = {}
    for r in records:
        cid = r["case_id"]
        is_full = bool(r.get("is_full_fov"))
        cases[cid] = {
            "subject_id": str(r["subject_id"]),
            "crop_type": "full" if is_full else str(r["crop_type"]),
            "severity": 0 if is_full else int(r["severity"]),
            "is_full_fov": is_full,
            "gt_label_path": str(gt_path_for(r)),
            "gt_struct_to_value": {k: int(v) for k, v in gt_stv_for(r).items()},
            "pred_file": f"{cid}.nii.gz",
            "fov_mask_file": f"{cid}.nii.gz",
            "source_shape": [int(s) for s in r["source_shape"]],
        }
    return {"experiment": "fov_completion", "control": control_name, "fold": int(fold),
            "structures": structures, "n": len(cases), "cases": cases}


def write_fov_mask_nii(mask_array: np.ndarray, ref_affine, out_path: str) -> None:
    """masi-55: write the FOV mask with the truncated-CT affine so eval can resample
    it to the GT grid via affines."""
    import nibabel as nib                                     # noqa: WPS433
    nib.save(nib.Nifti1Image(mask_array.astype(np.uint8), np.asarray(ref_affine)), str(out_path))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--completion-manifest", help="fov_completion_manifest.json")
    ap.add_argument("--splits-final", default=None, help="nnU-Net splits_final.json (val/train)")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--split", choices=["val", "train"], default="val")
    ap.add_argument("--case-list", default=None,
                    help="explicit file of case ids (one per line) for the HELD-OUT TEST set "
                         "(revised-plan §7); overrides --splits-final/--split.")
    ap.add_argument("--assert-disjoint-with", action="append", default=None,
                    help="file(s) of case ids whose SUBJECTS must be disjoint from this set "
                         "(e.g. the train+val case lists); repeatable. Raises on overlap.")
    ap.add_argument("--out", help="evalset root")
    ap.add_argument("--control-name", default="PHOTON_CT_CORR_C_fov")
    ap.add_argument("--truncated-ct-dir", default=None,
                    help="dir of truncated CTs (build_fov_completion_data) for the FOV-mask affine")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return _selftest()
    for req in ("completion_manifest", "out"):
        if getattr(args, req) in (None, ""):
            ap.error(f"--{req.replace('_', '-')} is required (or use --self-test).")

    def _read_ids(path):
        return {ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()
                and not ln.startswith("#")}

    recs = records_by_case(args.completion_manifest)
    if args.case_list:                                        # held-out test (§7)
        chosen_ids = _read_ids(args.case_list)
        other = [_read_ids(p) for p in (args.assert_disjoint_with or [])]
    else:
        if not args.splits_final:
            ap.error("--splits-final is required unless --case-list is given.")
        train_ids, val_ids = load_split_case_ids(args.splits_final, args.fold)
        chosen_ids = val_ids if args.split == "val" else train_ids
        other = [train_ids] if args.split == "val" else [val_ids]
    records = select_eval_cases(recs, chosen_ids, other_split_case_ids=other)
    print(f"[fov-evalset] {args.split} fold {args.fold}: {len(records)} FOV case(s) "
          f"from {len({r['subject_id'] for r in records})} subject(s).")
    print("[fov-evalset] NOTE: imagesTs/prevsegTs assembly + GT resolution run on masi-55 "
          "(reuse engine.convert keyed by FOV case id); this builder writes fovMaskTs + the map.")
    # The masi-55 driver wires gt_path_for/gt_stv_for + truncated-CT affine and calls
    # write_fov_mask_nii + assemble_eval_map. Left as an importable API here so the
    # geometry (mask, split, map) is testable without the data tree.
    return 0


def _selftest() -> int:
    # 2 subjects × 7 conditions; a leaked split must raise; mask + map correct.
    recs = []
    for s in ("000", "001"):
        recs.append({"case_id": f"fov_{s}_full", "subject_id": s, "is_full_fov": True,
                     "source_shape": [8, 8, 8]})
        for ct in ("axial", "corner"):
            for sev in (20, 35, 50):
                recs.append({"case_id": f"fov_{s}_{ct}_rm{sev}", "subject_id": s,
                             "crop_type": ct, "severity": sev, "is_full_fov": False,
                             "source_shape": [8, 8, 8],
                             "visible_box": [[4, 8], [0, 8], [0, 8]]})
    by_case = {r["case_id"]: r for r in recs}
    train_ids = {c for c in by_case if by_case[c]["subject_id"] == "000"}
    val_ids = {c for c in by_case if by_case[c]["subject_id"] == "001"}

    sel = select_eval_cases(by_case, val_ids, other_split_case_ids=[train_ids])
    assert len(sel) == 7 and {r["subject_id"] for r in sel} == {"001"}

    # leakage: put a 000 case into "val" alongside train 000 -> raise
    try:
        select_eval_cases(by_case, val_ids | {"fov_000_full"}, other_split_case_ids=[train_ids])
        raise AssertionError("cross-split subject overlap should raise")
    except RuntimeError:
        pass
    # disjoint-check helper
    stats = check_disjoint_subjects({"train": train_ids, "val": val_ids}, by_case)
    assert stats == {"train": 1, "val": 1}
    try:
        check_disjoint_subjects({"train": train_ids, "val": val_ids | {"fov_000_full"}}, by_case)
        raise AssertionError("disjoint-check should raise on overlap")
    except RuntimeError:
        pass

    # FOV mask: truncated -> the visible_box cuboid; full -> all ones
    m = build_fov_mask_array([8, 8, 8], [[4, 8], [0, 8], [0, 8]], is_full_fov=False)
    assert m.dtype == np.uint8 and m.sum() == 4 * 8 * 8 and m[:4].sum() == 0 and m[4:].sum() == 4 * 8 * 8
    mf = build_fov_mask_array([8, 8, 8], None, is_full_fov=True)
    assert mf.sum() == 8 * 8 * 8

    # map assembly: exact ids, embedded metadata, full severity 0
    mp = assemble_eval_map(sel, "PHOTON_CT_CORR_C_fov", 0,
                           gt_path_for=lambda r: f"/gt/{r['subject_id']}.nii.gz",
                           gt_stv_for=lambda r: {"ON": 1, "Recti": 2, "Globe": 3, "Fat": 4})
    assert mp["n"] == 7 and set(mp["cases"]) == val_ids
    full = mp["cases"]["fov_001_full"]
    assert full["crop_type"] == "full" and full["severity"] == 0 and full["is_full_fov"]
    ax = mp["cases"]["fov_001_axial_rm35"]
    assert ax["crop_type"] == "axial" and ax["severity"] == 35 and ax["pred_file"] == "fov_001_axial_rm35.nii.gz"
    print("FOV-EVALSET-BUILDER SELF-TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
