#!/usr/bin/env python3
"""Stage CT inputs for nnUNetv2_predict.

Reads CNISP's test_cases.txt, resolves the source CT image path for each
of the 31 unique source scans, and symlinks them into
``{work_dir}/input/native/{source_id}_0000.nii.gz`` (nnUNetv2's
channel-0 naming convention).

Also writes ``{work_dir}/source_to_path.json`` so downstream scripts
(SMORE prep, compare) can find each source's CT, GT, scheme, and the
metadata JSON paths without re-doing the lookup.

The source resolution lives in ``nnunet.data_prep.resolve_gt`` and the
symlink helper in ``nnunet.helpers.fs``; this script wires them into the
staging loop.

Usage
-----
    python nnunet/prepare_inputs.py --config nnunet/configs.yaml \
        [--atlas-image-dir /path/...] [--pivot-csv /path/...] \
        [--work-dir /path/...] [--split {test,train}] [--casefile FILE ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

# Make ``nnunet.*`` importable when run as ``python nnunet/...``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nnunet.helpers.config import load_yaml  # noqa: E402
from nnunet.helpers.fs import safe_symlink as _safe_symlink  # noqa: E402
from nnunet.data_prep.resolve_gt import fail_on_missing, resolve_sources  # noqa: E402


def run(args) -> int:
    cfg = load_yaml(Path(args.config))
    cnisp_paths = load_yaml(Path(cfg["cnisp_paths_yaml"]))

    atlas_image_dir = Path(args.atlas_image_dir or cfg["atlas_image_dir"])
    pivot_csv = Path(args.pivot_csv or cfg["pivot_csv"])
    work_dir = Path(args.work_dir or cfg["work_dir"])
    # 'train' namespaces every output under work_dir/train_split/ so the
    # modeling-split data-gen stays isolated from the test deployment tree.
    if args.split == "train":
        work_dir = work_dir / "train_split"

    casefiles_dir = Path(cnisp_paths["casefiles_dir"])
    if args.casefile:
        case_paths: List[Path] = [Path(c) for c in args.casefile]
    elif args.split == "train":
        case_paths = [
            casefiles_dir / "train_cases.txt",
            casefiles_dir / "val_cases.txt",
        ]
    else:
        case_paths = [casefiles_dir / "test_cases.txt"]
    meta_dir = Path(cnisp_paths["aligned_dir"]) / "metadata"

    input_dir = work_dir / "input" / "native"
    input_dir.mkdir(parents=True, exist_ok=True)

    print(f"[prepare_inputs] split:      {args.split}")
    print(f"[prepare_inputs] cases:      {[str(p) for p in case_paths]}")
    print(f"[prepare_inputs] meta_dir:   {meta_dir}")
    print(f"[prepare_inputs] atlas img:  {atlas_image_dir}")
    print(f"[prepare_inputs] pivot csv:  {pivot_csv}")
    print(f"[prepare_inputs] out dir:    {input_dir}")

    sources, missing = resolve_sources(
        test_cases_path=case_paths,
        meta_dir=meta_dir,
        atlas_image_dir=atlas_image_dir,
        pivot_csv=pivot_csv,
        pivot_subject_column=cfg.get("pivot_subject_column", "subject"),
        pivot_image_path_columns=cfg.get("pivot_image_path_columns"),
        detect_atlas_offset=False,         # not needed at staging time
        require_ct=True,
    )
    fail_on_missing(missing, "prepare_inputs")

    print(f"[prepare_inputs] resolved {len(sources)} source(s); "
          f"writing channel-0 symlinks…")

    manifest: Dict[str, Dict] = {}
    for src in sources:
        if src.ct_image_path is None:  # safety; require_ct already enforced
            continue
        dst = input_dir / f"{src.source_id}_0000.nii.gz"
        _safe_symlink(src.ct_image_path, dst)
        manifest[src.source_id] = {
            "ct_image_path": str(src.ct_image_path),
            "gt_label_path": str(src.gt_label_path),
            "gt_scheme": src.gt_scheme,
            "gt_source": src.gt_source,
            "casenames": list(src.casenames),
            "metadata_jsons": [str(p) for p in src.metadata_json_paths],
            "input_symlink": str(dst),
        }

    manifest_path = work_dir / "source_to_path.json"
    work_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    n_atlas = sum(1 for s in sources if s.gt_source == "atlas")
    n_chk = sum(1 for s in sources if s.gt_source == "chk_pseudo")
    print(f"[prepare_inputs] wrote {len(manifest)} symlinks "
          f"({n_atlas} atlas + {n_chk} chk_)")
    print(f"[prepare_inputs] manifest:  {manifest_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="nnunet/configs.yaml",
                    help="YAML config (default: %(default)s)")
    ap.add_argument("--atlas-image-dir", default=None,
                    help="Override atlas_image_dir from config")
    ap.add_argument("--pivot-csv", default=None,
                    help="Override pivot_csv from config")
    ap.add_argument("--work-dir", default=None,
                    help="Override work_dir from config")
    ap.add_argument("--split", choices=["test", "train"], default="test",
                    help="Which case split to stage. 'test' (default) reads "
                         "test_cases.txt and writes under work_dir/. 'train' "
                         "reads train_cases.txt + val_cases.txt (the modeling "
                         "scans, all chk_*) and writes under a namespaced "
                         "work_dir/train_split/ so the v6 nnUNet-obs data-gen "
                         "never clobbers the test deployment artifacts. No "
                         "nnUNet inference is run here either way -- this only "
                         "resolves/symlinks the source CTs.")
    ap.add_argument("--casefile", action="append", default=None,
                    help="Override the casefile(s) for the chosen split. May "
                         "be passed multiple times; each is read and merged.")
    return ap


if __name__ == "__main__":
    sys.exit(run(build_parser().parse_args()))
