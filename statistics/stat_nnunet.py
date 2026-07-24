#!/usr/bin/env python3
"""nnUNet dataset statistics — reads splits_final.json (or imagesTr).

Usage (on the GPU server where $nnUNet_preprocessed is mounted):
    python statistics/stat_nnunet.py

Writes to statistics/:
    nnunet835_foldN_{train,val}.txt   (per-fold case lists)
    nnunet835_all.txt                 (union across folds)
    nnunet845_all.txt / nnunet855_all.txt  (corrector datasets)
    nnunet_summary.txt                (counts)

Naming conventions handled:
    835:  {subj}_{subj}_{date}_CT_{img}_{na}__{index}
          subject = parts[0], session = parts[2] (date)
    845/855: corr_{case}_step{XX}
          strip corr_ and _stepNN, then subject = parts[0], session = parts[1]
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "statistics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PP = os.environ.get("nnUNet_preprocessed",
                     "/fs5/p_masi/linz18/EyeSegmentation/nnUNet_preprocessed")
RAW = os.environ.get("nnUNet_raw",
                      "/fs5/p_masi/linz18/EyeSegmentation/nnUNet_raw")

DATASETS = {
    835: "PHOTON_CT_QAfiltered",
    845: "PHOTON_CT_CORR_C_cnisp",
    855: "PHOTON_CT_CORR_B_stacked",
}


def ds_dir(ds_id: int, root: str = PP) -> Path:
    return Path(root) / f"Dataset{ds_id:03d}_{DATASETS[ds_id]}"


# ── Parsing ──────────────────────────────────────────────────────────

def parse_835(case_id: str):
    """835 case: {subj}_{subj}_{date}_CT_{img}_{na}__{index}"""
    core = re.sub(r"__\d+$", "", case_id)
    parts = core.split("_")
    if len(parts) >= 3 and parts[0] == parts[1]:
        subj, ses = parts[0], parts[2]
        scan_core = "_".join([parts[0]] + parts[2:])
    elif len(parts) >= 2:
        subj, ses = parts[0], parts[1]
        scan_core = core
    else:
        subj, ses, scan_core = parts[0], "NA", core
    return {"case_id": case_id, "subject": subj, "session": ses,
            "step": "NA", "scan_core": scan_core}


def parse_corrector(case_id: str):
    """845/855 case: corr_{case}_step{XX}"""
    s = case_id
    step = "NA"
    m = re.search(r"_step(\d+)$", s)
    if m:
        step = m.group(1)
        s = s[:m.start()]
    if s.startswith("corr_"):
        s = s[len("corr_"):]
    parts = s.split("_")
    subj = parts[0]
    ses = parts[1] if len(parts) > 1 else "NA"
    scan_core = re.sub(r"_step\d+$", "", re.sub(r"^corr_", "", case_id))
    return {"case_id": case_id, "subject": subj, "session": ses,
            "step": step, "scan_core": scan_core}


def get_parser(ds_id: int):
    return parse_835 if ds_id == 835 else parse_corrector


# ── I/O helpers ──────────────────────────────────────────────────────

HEADER = "case_id\tsubject\tsession\tstep\tscan_core"


def write_cases(tag: str, rows: list):
    out = OUT_DIR / f"{tag}.txt"
    with open(out, "w") as f:
        f.write(HEADER + "\n")
        for r in rows:
            f.write("\t".join(str(r[k]) for k in HEADER.split("\t")) + "\n")
    print(f"  wrote {out.name}  ({len(rows)} rows)")
    return out


def summarize(rows: list, ds_id: int):
    subjects = set()
    scans = set()
    steps_hist = defaultdict(int)
    for r in rows:
        subjects.add(r["subject"])
        scans.add((r["subject"], r["session"]))
        steps_hist[r["step"]] += 1
    info = {
        "cases": len(rows),
        "scans": len(scans),
        "subjects": len(subjects),
    }
    if ds_id != 835:
        info["steps_hist"] = dict(sorted(steps_hist.items()))
    return info


# ── Per-dataset logic ────────────────────────────────────────────────

def load_splits(ds_id: int):
    """Try splits_final.json; fall back to imagesTr listing."""
    pp_dir = ds_dir(ds_id, PP)
    splits_path = pp_dir / "splits_final.json"
    if splits_path.exists():
        return json.load(open(splits_path)), "splits_final"

    raw_images = ds_dir(ds_id, RAW) / "imagesTr"
    if raw_images.is_dir():
        ids = sorted({
            re.sub(r"_\d{4}\.(nii\.gz|nii)$", "", f.name)
            for f in raw_images.iterdir() if f.name.endswith((".nii.gz", ".nii"))
        })
        return [{"train": ids, "val": []}], "imagesTr"

    return None, None


def process_835():
    print(f"\n{'='*60}")
    print(f"Dataset835 ({DATASETS[835]})")
    print(f"{'='*60}")
    folds, src = load_splits(835)
    if folds is None:
        print("  NOT FOUND — check $nnUNet_preprocessed or $nnUNet_raw")
        return []

    print(f"  source: {src}, {len(folds)} fold(s)")
    parse = get_parser(835)
    all_ids = set()
    summaries = []

    for i, fold in enumerate(folds):
        for which in ("train", "val"):
            ids = fold.get(which, [])
            rows = [parse(c) for c in ids]
            write_cases(f"nnunet835_fold{i}_{which}", rows)
            s = summarize(rows, 835)
            summaries.append({"tag": f"fold{i}_{which}", **s})
            print(f"    fold{i} {which:5s}: cases={s['cases']:4d}  "
                  f"scans={s['scans']:4d}  subjects={s['subjects']:4d}")
            all_ids.update(ids)

    all_rows = [parse(c) for c in sorted(all_ids)]
    write_cases("nnunet835_all", all_rows)
    s_all = summarize(all_rows, 835)
    summaries.append({"tag": "ALL", **s_all})
    print(f"    ALL        : cases={s_all['cases']:4d}  "
          f"scans={s_all['scans']:4d}  subjects={s_all['subjects']:4d}")
    return summaries


def process_corrector(ds_id: int):
    name = DATASETS[ds_id]
    print(f"\n{'='*60}")
    print(f"Dataset{ds_id} ({name})")
    print(f"{'='*60}")
    folds, src = load_splits(ds_id)
    if folds is None:
        print("  NOT FOUND — check $nnUNet_preprocessed or $nnUNet_raw")
        return []

    print(f"  source: {src}, {len(folds)} fold(s)")
    parse = get_parser(ds_id)
    all_ids = set()
    summaries = []

    for i, fold in enumerate(folds):
        for which in ("train", "val"):
            ids = fold.get(which, [])
            rows = [parse(c) for c in ids]
            tag = f"nnunet{ds_id}_fold{i}_{which}"
            write_cases(tag, rows)
            s = summarize(rows, ds_id)
            summaries.append({"tag": f"fold{i}_{which}", **s})
            steps = s.get("steps_hist", {})
            print(f"    fold{i} {which:5s}: cases={s['cases']:4d}  "
                  f"scans={s['scans']:4d}  subjects={s['subjects']:4d}  "
                  f"steps={steps}")
            all_ids.update(ids)

    all_rows = [parse(c) for c in sorted(all_ids)]
    write_cases(f"nnunet{ds_id}_all", all_rows)
    s_all = summarize(all_rows, ds_id)
    summaries.append({"tag": "ALL", **s_all})
    steps = s_all.get("steps_hist", {})
    print(f"    ALL        : cases={s_all['cases']:4d}  "
          f"scans={s_all['scans']:4d}  subjects={s_all['subjects']:4d}  "
          f"steps={steps}")
    return summaries


def main():
    print("=== nnUNet dataset statistics ===")
    print(f"nnUNet_preprocessed = {PP}")
    print(f"nnUNet_raw          = {RAW}")

    all_summaries = {}
    all_summaries[835] = process_835()
    for ds_id in (845, 855):
        all_summaries[ds_id] = process_corrector(ds_id)

    summary_path = OUT_DIR / "nnunet_summary.txt"
    with open(summary_path, "w") as f:
        f.write("dataset\ttag\tcases\tscans\tsubjects\tsteps_hist\n")
        for ds_id, entries in all_summaries.items():
            for e in entries:
                steps = e.get("steps_hist", "")
                f.write(f"{ds_id}\t{e['tag']}\t{e['cases']}\t{e['scans']}\t"
                        f"{e['subjects']}\t{steps}\n")
    print(f"\nwrote {summary_path.name}")


if __name__ == "__main__":
    main()
