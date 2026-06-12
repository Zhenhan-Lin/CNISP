#!/usr/bin/env python3
"""Canonical-align REAL paired acquisitions for the Turella sim3 eval line.

Real paired data = two SEPARATE acquisitions of the same subject:
  * a low-resolution (thick-slice / anisotropic) scan, and
  * a high-resolution scan used as ground truth.

Unlike the simulated curves, there is no voxel correspondence between the
two scans. Each is canonical-aligned INDEPENDENTLY (registration-free),
exactly like the training patches. At eval time CNISP reconstructs from the
aligned low-res input and the reconstructed mask is RIGIDLY registered to
the aligned GT mask before Dice (post-hoc), following Turella et al.

Inputs
------
* A manifest JSON mapping each source id to its two scans::

    {
      "<source_id>": {
        "lowres_pred": "/abs/path/lowres_nnunet_pred.nii.gz",
        "hires_gt":    "/abs/path/hires_gt.nii.gz"
      },
      ...
    }

  ``lowres_pred`` is the nnUNet prediction on the REAL low-res scan (native
  grid, plan label scheme). ``hires_gt`` is the high-resolution ground truth
  (manual annotation or nnUNet-on-hires), same label scheme.

Outputs (under ``aligned_dir``)
-------------------------------
* ``labels_realpair_input/{casename}.nii.gz`` -- aligned low-res input patch
* ``labels_realpair_gt/{casename}.nii.gz``    -- aligned hi-res GT patch
* ``metadata_realpair_gt/{casename}.json``    -- GT alignment metadata (for
  native unmap of the registered prediction)

Each scan is aligned with the SAME ``source_id`` so the per-eye casenames
(``{source_id}_OD`` / ``{source_id}_OS``) match between input and GT.

The output-directory layout lives in ``nnunet.lib.patches``; this script
wires it into the per-side align/match/save loop.

Usage
-----
    python nnunet/build_realpair_patches.py \
        --config nnunet/configs.yaml \
        --manifest /abs/path/realpair_manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List

import nibabel as nib
import numpy as np

# Make ``nnunet.*`` importable when run as ``python nnunet/...``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nnunet.helpers.config import (  # noqa: E402
    add_cnisp_src_to_syspath,
    load_yaml,
)
from nnunet.helpers.patch_size import resolve_patch_size_mm  # noqa: E402
from nnunet.lib.patches import realpair_layout  # noqa: E402

add_cnisp_src_to_syspath(__file__)

from data_prep.canonical_align import (  # noqa: E402
    align_single_case,
    infer_patch_size_mm,
)


def _align_side(seg_path: Path, sid: str, patch_size_mm: float,
                issues: List[str]):
    """Align one scan; return list of (patch, affine, meta) or [] on failure."""
    if not seg_path.exists():
        issues.append(f"{sid}: missing scan at {seg_path}")
        return []
    try:
        return align_single_case(
            seg_path=str(seg_path),
            source_id=sid,
            source="realpair",
            patch_size_mm=patch_size_mm,
        )
    except Exception as e:  # noqa: BLE001
        issues.append(f"{sid}: align_single_case raised "
                      f"{type(e).__name__}: {e}")
        return []


def run(args) -> int:
    cfg = load_yaml(Path(args.config))
    cnisp_paths = load_yaml(Path(cfg["cnisp_paths_yaml"]))
    aligned_dir = Path(cnisp_paths["aligned_dir"])

    patch_size_mm = resolve_patch_size_mm(
        args.patch_size, aligned_dir / "metadata",
        log_prefix="realpair", infer_fn=infer_patch_size_mm,
    )

    layout = realpair_layout(cnisp_paths, aligned_dir)
    for d in layout.values():
        d.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"[realpair] manifest missing: {manifest_path}", file=sys.stderr)
        return 2
    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"[realpair] sources to process: {len(manifest)}")
    print(f"[realpair] input  patches -> {layout['input_dir']}")
    print(f"[realpair] GT     patches -> {layout['gt_dir']}")
    print(f"[realpair] GT     metadata-> {layout['gt_meta_dir']}")

    n_written = n_skipped = n_failed = 0
    issues: List[str] = []

    for sid in sorted(manifest):
        entry = manifest[sid]
        lowres_path = Path(entry["lowres_pred"])
        hires_path = Path(entry["hires_gt"])

        # Skip-if-done: both eyes' input + GT + GT-metadata present.
        existing = all(
            (layout["input_dir"] / f"{sid}_{eye}.nii.gz").exists()
            and (layout["gt_dir"] / f"{sid}_{eye}.nii.gz").exists()
            and (layout["gt_meta_dir"] / f"{sid}_{eye}.json").exists()
            for eye in ("OD", "OS")
        )
        if existing and not args.force:
            n_skipped += 1
            continue

        in_results = _align_side(lowres_path, sid, patch_size_mm, issues)
        gt_results = _align_side(hires_path, sid, patch_size_mm, issues)
        if not in_results or not gt_results:
            n_failed += 1
            continue

        # Match eyes by casename so input/GT correspond per eye.
        gt_by_case = {meta.casename: (patch, pa, meta)
                      for patch, pa, meta in gt_results}
        for patch, pa, meta in in_results:
            if meta.casename not in gt_by_case:
                issues.append(f"{meta.casename}: input eye has no GT counterpart")
                continue
            # input patch
            nib.save(
                nib.Nifti1Image(patch.astype(np.uint8), pa),
                str(layout["input_dir"] / f"{meta.casename}.nii.gz"),
            )
            # GT patch + metadata
            gt_patch, gt_pa, gt_meta = gt_by_case[meta.casename]
            nib.save(
                nib.Nifti1Image(gt_patch.astype(np.uint8), gt_pa),
                str(layout["gt_dir"] / f"{gt_meta.casename}.nii.gz"),
            )
            with open(layout["gt_meta_dir"] / f"{gt_meta.casename}.json", "w") as f:
                json.dump(asdict(gt_meta), f, indent=2)
            n_written += 1

    if issues:
        print(f"\n[realpair] {len(issues)} issue(s):", file=sys.stderr)
        for line in issues[:20]:
            print(f"  - {line}", file=sys.stderr)
        if len(issues) > 20:
            print(f"  ... and {len(issues) - 20} more", file=sys.stderr)

    print(f"\n[realpair] wrote {n_written} eye-patch pair(s); "
          f"{n_skipped} source(s) already complete; {n_failed} source(s) failed.")
    return 0 if n_failed == 0 else 3


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="nnunet/configs.yaml")
    ap.add_argument("--manifest", required=True,
                    help="JSON mapping source_id -> {lowres_pred, hires_gt}.")
    ap.add_argument("--force", action="store_true",
                    help="Re-write patches even if they already exist.")
    ap.add_argument("--patch-size", type=float, default=None,
                    help="Patch extent (mm). Defaults to the trained model's "
                         "value recorded in aligned_dir/metadata/.")
    return ap


if __name__ == "__main__":
    sys.exit(run(build_parser().parse_args()))
