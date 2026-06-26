#!/usr/bin/env python3
"""
Step 01.1: Build the "addon" training list (manifest) for CNISP.

(Sub-step of data prep / step 01: enumerate NEW masks to add to training.)

Generates ONE reviewable CSV manifest that enumerates the *new* GT masks to add
to the CNISP training set. Three sources are merged:

  1. review_checklist_part2.csv, rows with keep==True  -> source "checklist"
     (nnUNet CT predictions kept after QA; deduped to one session per subject
      to mirror data_prep.canonical_align._collect_scan_list).
  2. FLAIR atlas manual labels (atlas_labels/*.nii.gz) -> source "atlas_flair"
  3. T2   atlas manual labels (atlas_labels/*.nii.gz) -> source "atlas_t2"
     (these carry a mislabeled raw value 2; recorded in ignore_labels so the
      aligner drops it before label-scheme detection).

CNISP only consumes masks, so the image modality (CT vs MRI) is irrelevant —
the manifest lists masks only. The CT degradation bank (modality: ct in
train_v6_5_gt.yaml) is applied to these GT masks at train time regardless.

Subjects/cases already present in an existing split (train/val/test_cases.txt)
are excluded so a part2 patient never lands in two splits (e.g. subject 10398
is already in val_cases.txt).

The manifest columns are: seg_path, source_id, source, ignore_labels[, note].
Feed the manifest to scripts/012_add_training_data.py.

Usage:
    python scripts/011_build_addon_list.py -p configs/paths.yaml \
        --part2-csv /path/review_checklist_part2.csv \
        --flair-label-dir /fs5/.../FLAIR-atlas/atlas_labels \
        --t2-label-dir    /fs5/.../T2-atlas/atlas_labels \
        -o <casefiles_dir>/train_addon_manifest.csv
"""

import argparse
import csv
import glob
from pathlib import Path

KEEP_TRUE = {"true", "1", "yes", "y"}


def _strip_nii(name: str) -> str:
    for suf in (".nii.gz", ".nii"):
        if name.endswith(suf):
            return name[: -len(suf)]
    return name


def _existing_chk_subjects(casefiles_dir: Path):
    """Return the set of subject IDs already used by chk_ cases in any split."""
    subjects = set()
    for fname in ("train_cases.txt", "val_cases.txt", "test_cases.txt"):
        fp = casefiles_dir / fname
        if not fp.exists():
            continue
        for line in fp.read_text().splitlines():
            case = line.strip()
            if not case.startswith("chk_"):
                continue
            # chk_{subject}_OD / chk_{subject}_OS -> subject
            body = case[len("chk_"):]
            body = body.rsplit("_", 1)[0]  # drop OD/OS
            subjects.add(body)
    return subjects


def _existing_source_ids(casefiles_dir: Path):
    """Return the set of source_ids (case minus _OD/_OS) already in any split."""
    ids = set()
    for fname in ("train_cases.txt", "val_cases.txt", "test_cases.txt"):
        fp = casefiles_dir / fname
        if not fp.exists():
            continue
        for line in fp.read_text().splitlines():
            case = line.strip()
            if not case:
                continue
            ids.add(case.rsplit("_", 1)[0])
    return ids


def _collect_part2(part2_csv: Path, exclude_subjects):
    """Keep==True rows, one session per subject, excluding known subjects."""
    by_subject = {}  # subject -> (session, pred_path, note)
    with open(part2_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get("keep") or "").strip().lower() not in KEEP_TRUE:
                continue
            subject = (row.get("subject") or "").strip()
            session = (row.get("session") or "").strip()
            pred = (row.get("pred_path") or "").strip()
            if not subject or not pred:
                continue
            if subject in exclude_subjects:
                continue
            # keep the first session per subject (sorted), mirroring
            # canonical_align._collect_scan_list dedup behaviour
            prev = by_subject.get(subject)
            if prev is None or session < prev[0]:
                by_subject[subject] = (session, pred, (row.get("notes") or "").strip())

    rows = []
    for subject in sorted(by_subject):
        session, pred, note = by_subject[subject]
        rows.append({
            "seg_path": pred,
            "source_id": f"chk_{subject}",
            "source": "checklist",
            "ignore_labels": "",
            "note": f"part2 keep=True session={session} {note}".strip(),
        })
    return rows


def _collect_atlas(label_dir: Path, source: str, prefix: str,
                   ignore_labels: str, exclude_ids):
    rows = []
    if not label_dir.exists():
        print(f"  WARN atlas dir not found: {label_dir}")
        return rows
    for fp in sorted(glob.glob(str(label_dir / "*.nii.gz"))):
        stem = _strip_nii(Path(fp).name)
        source_id = f"{prefix}{stem}"
        if source_id in exclude_ids:
            continue
        rows.append({
            "seg_path": fp,
            "source_id": source_id,
            "source": source,
            "ignore_labels": ignore_labels,
            "note": "",
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--paths", default=None,
                    help="paths.yaml (for casefiles_dir; used to exclude "
                         "subjects already in a split)")
    ap.add_argument("--casefiles-dir", default=None,
                    help="override casefiles_dir from paths.yaml")
    ap.add_argument("--part2-csv", required=True,
                    help="review_checklist_part2.csv")
    ap.add_argument("--flair-label-dir",
                    default="/fs5/p_masi/linz18/data/atlas_megadocker_export/"
                            "FLAIR-atlas/atlas_labels")
    ap.add_argument("--t2-label-dir",
                    default="/fs5/p_masi/linz18/data/atlas_megadocker_export/"
                            "T2-atlas/atlas_labels")
    ap.add_argument("--t2-ignore-labels", default="2",
                    help="raw label value(s) to drop from T2 atlas masks "
                         "(space/comma separated). Default '2'.")
    ap.add_argument("-o", "--out", default=None,
                    help="output manifest CSV "
                         "(default <casefiles_dir>/train_addon_manifest.csv)")
    args = ap.parse_args()

    casefiles_dir = None
    if args.casefiles_dir:
        casefiles_dir = Path(args.casefiles_dir)
    elif args.paths:
        import yaml
        with open(args.paths) as f:
            casefiles_dir = Path(yaml.safe_load(f)["casefiles_dir"])

    exclude_subjects = set()
    exclude_ids = set()
    if casefiles_dir and casefiles_dir.exists():
        exclude_subjects = _existing_chk_subjects(casefiles_dir)
        exclude_ids = _existing_source_ids(casefiles_dir)
        print(f"Existing splits: {len(exclude_subjects)} chk subjects, "
              f"{len(exclude_ids)} source_ids (will be excluded)")
    else:
        print("No casefiles_dir given/found -> NOT excluding existing cases. "
              "Pass -p configs/paths.yaml to avoid train/val/test leakage.")

    rows = []
    p2 = _collect_part2(Path(args.part2_csv), exclude_subjects)
    print(f"checklist (part2 keep=True, deduped, excluded): {len(p2)} subjects")
    rows += p2

    fl = _collect_atlas(Path(args.flair_label_dir), "atlas_flair",
                        "atlas_flair_", "", exclude_ids)
    print(f"atlas_flair: {len(fl)} masks")
    rows += fl

    t2 = _collect_atlas(Path(args.t2_label_dir), "atlas_t2",
                        "atlas_t2_", args.t2_ignore_labels, exclude_ids)
    print(f"atlas_t2: {len(t2)} masks (ignore_labels='{args.t2_ignore_labels}')")
    rows += t2

    if args.out:
        out = Path(args.out)
    elif casefiles_dir:
        out = casefiles_dir / "train_addon_manifest.csv"
    else:
        out = Path("train_addon_manifest.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    fields = ["seg_path", "source_id", "source", "ignore_labels", "note"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\nWrote {len(rows)} scans -> {out}")
    print("Review it, then run:")
    print(f"  python scripts/012_add_training_data.py -p configs/paths.yaml "
          f"--manifest {out}")


if __name__ == "__main__":
    main()
