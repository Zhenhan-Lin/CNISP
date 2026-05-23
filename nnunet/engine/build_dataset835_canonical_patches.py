#!/usr/bin/env python3
"""Canonical-align Dataset835's DENSE pred for every test source.

Inputs
------
* ``${work_dir}/nnunet_pred_native/<sid>.nii.gz`` -- dense Dataset835
  prediction per source on the native CT grid (Phase 1 output of the
  ``nnunet-predict`` pipeline phase). Same label scheme as the nnUNet
  training plan: {0:BG, 1:ON, 2:Recti, 3:Globe, 4:Fat}.
* ``${work_dir}/source_to_path.json`` -- 31-source manifest produced by
  ``nnunet/data_prep/prepare_inputs.py``.

Outputs
-------
* ``${aligned_dir}/${labels_dataset835_dirname}/{casename}.nii.gz``
  -- canonical-aligned orbital patches (one per eye per source).
* ``${aligned_dir}/${metadata_dataset835_dirname}/{casename}.json``
  -- alignment metadata so ``engine/native_mapping.invert_alignment_single_eye``
  can place CNISP predictions back into the source's native head volume.

Used as the dense Dice target for chk_* cases in the Option C
deployment curve (test_label_source=nnunet_pred). For atlas cases the
patches are written too -- they are not used as a Dice target (atlas
manual GT wins there), but they serve as the step-1 latent-opt input
for atlas cases and keep the on-disk layout symmetric.

Skip-if-done
------------
For each (source, eye) the script writes only when both the label
NIfTI and the metadata JSON are absent (or ``--force`` is passed). A
final summary prints the counts so it's easy to confirm the
``cnisp-prep-dataset835-gt`` phase has covered every expected case.

Usage
-----
    python nnunet/engine/build_dataset835_canonical_patches.py --config nnunet/configs.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import nibabel as nib
import numpy as np
import yaml


# Make orbital_shape_prior_st1 importable when run as a script.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CNISP_SRC = _REPO_ROOT / "orbital_shape_prior_st1"
if str(_CNISP_SRC) not in sys.path:
    sys.path.insert(0, str(_CNISP_SRC))

from data_prep.canonical_align import align_single_case  # noqa: E402


def _load_yaml(p: Path) -> Dict:
    with open(p) as f:
        return yaml.safe_load(f) or {}


def _aligned_dir_layout(cnisp_paths: dict, aligned_dir: Path) -> Dict[str, Path]:
    labels = cnisp_paths.get("labels_dataset835_dirname", "labels_dataset835")
    meta = cnisp_paths.get("metadata_dataset835_dirname", "metadata_dataset835")
    return {
        "labels_dir": aligned_dir / labels,
        "meta_dir":   aligned_dir / meta,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="nnunet/configs.yaml")
    ap.add_argument("--force", action="store_true",
                    help="Re-write patches even if both label + metadata "
                         "already exist on disk.")
    ap.add_argument("--patch-size", type=float, default=64.0)
    args = ap.parse_args()

    cfg = _load_yaml(Path(args.config))
    cnisp_paths = _load_yaml(Path(cfg["cnisp_paths_yaml"]))
    work_dir = Path(cfg["work_dir"])
    aligned_dir = Path(cnisp_paths["aligned_dir"])

    layout = _aligned_dir_layout(cnisp_paths, aligned_dir)
    layout["labels_dir"].mkdir(parents=True, exist_ok=True)
    layout["meta_dir"].mkdir(parents=True, exist_ok=True)

    source_manifest = work_dir / "source_to_path.json"
    if not source_manifest.exists():
        print(f"[dataset835_canonical] {source_manifest} missing -- "
              f"run nnunet/data_prep/prepare_inputs.py first.",
              file=sys.stderr)
        return 2
    with open(source_manifest) as f:
        src_to_path = json.load(f)

    dense_pred_dir = work_dir / "nnunet_pred_native"
    if not dense_pred_dir.is_dir():
        print(f"[dataset835_canonical] {dense_pred_dir} missing -- "
              f"did you run the `nnunet-predict` phase?",
              file=sys.stderr)
        return 2

    print(f"[dataset835_canonical] sources to process: {len(src_to_path)}")
    print(f"[dataset835_canonical] output labels  -> {layout['labels_dir']}")
    print(f"[dataset835_canonical] output metadata-> {layout['meta_dir']}")

    n_written = 0
    n_skipped_existing = 0
    n_failed = 0
    n_dropped_eye = 0
    issues: List[str] = []

    for sid in sorted(src_to_path):
        seg_path = dense_pred_dir / f"{sid}.nii.gz"
        if not seg_path.exists():
            issues.append(f"{sid}: dense pred missing at {seg_path}")
            n_failed += 1
            continue

        # Default expected eyes: 2 (OD, OS). If both are already on disk
        # we skip the whole source. If only one is on disk and the other
        # genuinely doesn't exist in this source (e.g. nnUNet dropped a
        # globe), we still re-align so the missing-eye signal stays
        # consistent with re-runs.
        all_existing = (
            (layout["labels_dir"] / f"{sid}_OD.nii.gz").exists()
            and (layout["labels_dir"] / f"{sid}_OS.nii.gz").exists()
            and (layout["meta_dir"]   / f"{sid}_OD.json").exists()
            and (layout["meta_dir"]   / f"{sid}_OS.json").exists()
        )
        if all_existing and not args.force:
            n_skipped_existing += 1
            continue

        try:
            results = align_single_case(
                seg_path=str(seg_path),
                source_id=sid,
                source="dataset835",
                patch_size_mm=args.patch_size,
            )
        except Exception as e:  # noqa: BLE001
            n_failed += 1
            issues.append(f"{sid}: align_single_case raised "
                          f"{type(e).__name__}: {e}")
            continue

        if not results:
            n_failed += 1
            issues.append(f"{sid}: no eyes found (Dataset835 pred may have "
                          f"dropped both globes)")
            continue
        if len(results) == 1:
            n_dropped_eye += 1
            issues.append(f"{sid}: only one eye found -- the other globe "
                          f"is absent from Dataset835's dense pred")

        for patch, pa, meta in results:
            label_out = layout["labels_dir"] / f"{meta.casename}.nii.gz"
            meta_out  = layout["meta_dir"]   / f"{meta.casename}.json"
            nib.save(
                nib.Nifti1Image(patch.astype(np.uint8), pa),
                str(label_out),
            )
            with open(meta_out, "w") as f:
                json.dump(asdict(meta), f, indent=2)
            n_written += 1

    if issues:
        print(f"\n[dataset835_canonical] {len(issues)} issue(s):",
              file=sys.stderr)
        for line in issues[:20]:
            print(f"  - {line}", file=sys.stderr)
        if len(issues) > 20:
            print(f"  ... and {len(issues) - 20} more", file=sys.stderr)

    print(f"\n[dataset835_canonical] wrote {n_written} patch(es); "
          f"{n_skipped_existing} source(s) already complete on disk; "
          f"{n_dropped_eye} source(s) with one eye dropped; "
          f"{n_failed} source(s) failed.")
    return 0 if n_failed == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
