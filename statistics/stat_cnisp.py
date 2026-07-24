#!/usr/bin/env python3
"""CNISP + corrector dataset statistics from casename_files.

Usage:
    python statistics/stat_cnisp.py

Splits processed:
    train              CNISP AutoDecoder training (chk_* naming)
    val                CNISP validation (atlas_flair/t2 + chk)
    test               shared test set (atlas only; chk_ rows excluded)
    corrector_train    stage-2 corrector training (PHOTON direct naming)

Writes to statistics/:
    cnisp_{split}.txt      per-case detail (TSV)
    cnisp_summary.txt      counts
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CASEFILE_DIR = REPO / "orbital_shape_prior_st1" / "casename_files"
META_DIR = REPO / "orbital_shape_prior_st1" / "aligned_patches" / "metadata"
OUT_DIR = REPO / "statistics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HEADER = "casename\tsource_id\tsubject\tsession\timg_label\teye\tsource_type\tscan_core"


def _parse_nifti_stem(stem: str):
    """Extract (subject, session) from a PHOTON nifti filename stem.

    Raw:  {subj}_{subj}_{date}_CT_{img}_{na}__{index}
    """
    core = re.sub(r"__\d+$", "", stem)
    parts = core.split("_")
    if len(parts) >= 3 and parts[0] == parts[1]:
        return parts[0], parts[2]
    elif len(parts) >= 2:
        return parts[0], parts[1]
    return parts[0], "NA"


def parse_casename(casename: str, meta_dir: Path | None = None):
    """Parse any casename into a uniform dict."""
    eye = "NA"
    if casename.endswith("_OD"):
        eye = "OD"
        core = casename[:-3]
    elif casename.endswith("_OS"):
        eye = "OS"
        core = casename[:-3]
    else:
        core = casename

    # ── Detect naming convention ──
    # A) chk_{subject_id}
    if core.startswith("chk_"):
        subject = core[4:]
        session, img_label = "NA", "NA"
        scan_core = core
        source_type = "checklist"
        if meta_dir:
            mp = meta_dir / f"{casename}.json"
            if mp.exists():
                meta = json.load(open(mp))
                nifti = Path(meta["original_nifti_path"]).stem.replace(".nii", "")
                subject, session = _parse_nifti_stem(nifti)
        return {
            "casename": casename, "source_id": core,
            "subject": subject, "session": session,
            "img_label": img_label, "eye": eye,
            "source_type": source_type, "scan_core": scan_core,
        }

    # B) atlas_*  (atlas_orbit..., atlas_flair_..., atlas_t2_...)
    if core.startswith("atlas_"):
        subject = core[6:]  # everything after "atlas_"
        return {
            "casename": casename, "source_id": core,
            "subject": subject, "session": "NA",
            "img_label": "NA", "eye": eye,
            "source_type": "atlas", "scan_core": core,
        }

    # C) PHOTON direct:  {subject}_{date}_CT_{img_label}
    #    e.g. 10058_20330227_CT_0
    parts = core.split("_")
    if len(parts) >= 4 and parts[2] == "CT":
        subject = parts[0]
        session = parts[1]
        img_label = parts[3]
        scan_core = core
        return {
            "casename": casename, "source_id": core,
            "subject": subject, "session": session,
            "img_label": img_label, "eye": eye,
            "source_type": "photon", "scan_core": scan_core,
        }

    # D) fallback
    subject = parts[0] if parts else core
    session = parts[1] if len(parts) > 1 else "NA"
    return {
        "casename": casename, "source_id": core,
        "subject": subject, "session": session,
        "img_label": "NA", "eye": eye,
        "source_type": "unknown", "scan_core": core,
    }


def read_cases(path: Path):
    if not path.exists():
        return []
    return [l.strip() for l in path.read_text().splitlines() if l.strip()]


def process_split(split: str, casenames: list, use_meta: bool = True):
    meta = META_DIR if use_meta else None
    return [parse_casename(c, meta) for c in casenames]


def write_split(tag: str, rows: list):
    out = OUT_DIR / f"cnisp_{tag}.txt"
    cols = HEADER.split("\t")
    with open(out, "w") as f:
        f.write(HEADER + "\n")
        for r in rows:
            f.write("\t".join(str(r[k]) for k in cols) + "\n")
    print(f"  wrote {out.name}  ({len(rows)} rows)")


def summarize(rows: list):
    subjects = set()
    sessions = set()      # (subject, session)
    volumes = set()       # (subject, session, img_label) — unique CT series
    sources = set()
    by_type = defaultdict(int)
    for r in rows:
        subjects.add(r["subject"])
        if r["session"] != "NA":
            sessions.add((r["subject"], r["session"]))
        volumes.add((r["subject"], r["session"], r["img_label"]))
        sources.add(r["source_id"])
        by_type[r["source_type"]] += 1
    return {
        "eyes": len(rows),
        "sources": len(sources),
        "subjects": len(subjects),
        "sessions": len(sessions) if sessions else len(subjects),
        "volumes": len(volumes),
        "by_type": dict(by_type),
    }


def main():
    print("=" * 60)
    print("CNISP + Corrector dataset statistics")
    print("=" * 60)

    splits = {}

    # ── CNISP train (chk_* format) ──
    cn = read_cases(CASEFILE_DIR / "train_cases.txt")
    splits["train"] = process_split("train", cn)

    # ── CNISP val (atlas_flair/t2 + chk, mixed) ──
    cn = read_cases(CASEFILE_DIR / "val_cases.txt")
    splits["val"] = process_split("val", cn)

    # ── Test: atlas only (exclude chk_*) ──
    cn_all = read_cases(CASEFILE_DIR / "test_cases.txt")
    cn_atlas = [c for c in cn_all if not c.startswith("chk_")]
    cn_chk = [c for c in cn_all if c.startswith("chk_")]
    splits["test"] = process_split("test", cn_atlas)
    if cn_chk:
        splits["test_chk_excluded"] = process_split("test_chk_excluded", cn_chk)

    # ── Corrector train (PHOTON direct naming) ──
    cn = read_cases(CASEFILE_DIR / "corrector_train_cases.txt")
    splits["corrector_train"] = process_split("corrector_train", cn, use_meta=False)

    # ── Print + write ──
    all_summaries = []
    for tag, rows in splits.items():
        s = summarize(rows)
        write_split(tag, rows)
        label = f"({tag})"
        if tag == "test_chk_excluded":
            label = "(test_chk — excluded from test)"
        print(f"\n  [{tag}]")
        print(f"    eyes       = {s['eyes']}")
        print(f"    sources    = {s['sources']}  (unique source_id)")
        print(f"    subjects   = {s['subjects']}")
        print(f"    sessions   = {s['sessions']}  (unique subject+date)")
        print(f"    volumes    = {s['volumes']}  (unique subject+date+img_label)")
        print(f"    by_type    = {s['by_type']}")
        all_summaries.append({"split": tag, **s})

    summary_path = OUT_DIR / "cnisp_summary.txt"
    with open(summary_path, "w") as f:
        f.write("split\teyes\tsources\tsubjects\tsessions\tvolumes\tby_type\n")
        for s in all_summaries:
            f.write(f"{s['split']}\t{s['eyes']}\t{s['sources']}\t"
                    f"{s['subjects']}\t{s['sessions']}\t{s['volumes']}\t"
                    f"{s['by_type']}\n")
    print(f"\nwrote {summary_path.name}")


if __name__ == "__main__":
    main()
