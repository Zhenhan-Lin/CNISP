#!/usr/bin/env python3
"""Canonical-align the corrector's nnUNet predictions into CNISP patches.

Bridges the corrector data tree (nnunet-c/data/) to what CNISP inference (032)
reads. For each selected image it canonical-aligns:

  * DENSE target frame  : the full-res Dataset835 prediction (CSV pred_path,
    recorded as gt_candidate_pred) -> aligned_dir/labels_dataset835/<case>_O{D,S}.nii.gz
    + aligned_dir/metadata_dataset835/<case>_O{D,S}.json  (native-inversion frame)
  * PER-STEP observation: the 835 pred on each degraded image
    (data/nnunet_pred/{case}_step{XX}.nii.gz) ->
    aligned_dir/labels_dataset835_{exp}_step_{XX}/<case>_O{D,S}.nii.gz
    (the CNISP latent-opt input under test_label_source=nnunet_pred)
    + aligned_dir/metadata_dataset835_{exp}_step_{XX}/<case>_O{D,S}.json
    (this observed crop's frame; CNISP native/iso inversion re-frames the
    reconstruction onto it so the OS mask isn't mirrored/misplaced)

It also writes the corrector casenames file (casefiles_dir/<corrector_train_casefile>)
so 032 can run with --test-casefile that list.

Casename scheme: source_id = CSV case_id (e.g. 10058_20330227_CT_0), eyes _OD/_OS.

Usage:
    python nnunet-c/scripts/align_corrector_data.py            # all cases/steps
    python nnunet-c/scripts/align_corrector_data.py --force
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List

# lib.* (nnunet-c) + nnunet.* (repo root)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.config import add_repo_to_syspath, load_corrector_config  # noqa: E402

add_repo_to_syspath(__file__)

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402

# data_prep.* (orbital_shape_prior_st1 on path) + reusable nnUNet helpers
from nnunet.helpers.config import add_cnisp_src_to_syspath  # noqa: E402
from nnunet.helpers.patch_size import resolve_patch_size_mm  # noqa: E402

add_cnisp_src_to_syspath(__file__)

from data_prep.canonical_align import align_single_case, infer_patch_size_mm  # noqa: E402
from engine.test_label_sources import exp_step_prefix  # noqa: E402

_DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "corrector.yaml"


def _align_and_write(seg_path: Path, source_id: str, source: str,
                     patch_size_mm: float, labels_dir: Path,
                     meta_dir: Path = None, force: bool = False) -> List[str]:
    """Align one seg into per-eye patches; write patches (+ metadata if meta_dir).

    Returns the list of casenames written/observed.
    """
    labels_dir.mkdir(parents=True, exist_ok=True)
    if meta_dir is not None:
        meta_dir.mkdir(parents=True, exist_ok=True)
    results = align_single_case(
        seg_path=str(seg_path), source_id=source_id, source=source,
        patch_size_mm=patch_size_mm,
    )
    written = []
    for patch, pa, meta in results:
        cn = meta.casename
        lp = labels_dir / f"{cn}.nii.gz"
        if not (lp.exists() and not force):
            nib.save(nib.Nifti1Image(patch.astype(np.uint8), pa), str(lp))
        if meta_dir is not None:
            mp = meta_dir / f"{cn}.json"
            if not (mp.exists() and not force):
                with open(mp, "w") as f:
                    json.dump(asdict(meta), f, indent=2)
        written.append(cn)
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(_DEFAULT_CONFIG))
    ap.add_argument("--patch-size", type=float, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_corrector_config(args.config, caller_file=__file__)
    res = cfg["_resolved"]
    cd = cfg["corrector_data"]
    cnisp_paths = cfg["_cnisp_paths"]
    experiment = cfg["experiment"]
    cnisp_aligned_dir = res["aligned_dir"]   # CNISP training patches (for patch-size only)

    data_root = Path(cd["data_root"])
    data_root = data_root if data_root.is_absolute() else (res["repo_root"] / data_root)
    nnunet_pred_dir = data_root / cd.get("nnunet_pred_dirname", "nnunet_pred")
    # Corrector aligned patches live under data/aligned_patch (not CNISP aligned_dir).
    aligned_patch = data_root / cd.get("aligned_patch_dirname", "aligned_patch")
    manifest_path = data_root / "corrector_data_manifest.json"
    if not manifest_path.is_file():
        print(f"[align] {manifest_path} missing -- run build_corrector_data.py first.",
              file=sys.stderr)
        return 2
    manifest = json.load(open(manifest_path))

    # Patch size pinned to the trained CNISP metadata (must match the AutoDecoder).
    patch_size_mm = resolve_patch_size_mm(
        args.patch_size, cnisp_aligned_dir / "metadata",
        log_prefix="corrector_align", infer_fn=infer_patch_size_mm,
    )

    # Output dirs: same subdir NAMES as the CNISP convention, but rooted under the
    # corrector's own aligned_patch dir so 032 reads them with --aligned-dir.
    dense_labels_dir = aligned_patch / cnisp_paths.get(
        "labels_dataset835_dirname", "labels_dataset835")
    dense_meta_dir = aligned_patch / cnisp_paths.get(
        "metadata_dataset835_dirname", "metadata_dataset835")
    base_prefix = cnisp_paths.get("labels_dataset835_step_prefix",
                                  "labels_dataset835_step_")
    step_prefix = exp_step_prefix(base_prefix, experiment)      # per-step obs dirs
    # Parallel per-step OBSERVED-metadata prefix ("labels"->"metadata"). CNISP's
    # native/iso inversion (engine.native_mapping._deployment_index_shift, via
    # 032 --observed-meta) reads these to re-frame the reconstruction from the
    # dense target crop to THIS observed input crop; without them the OS mask is
    # mirrored/misplaced at high step.
    meta_step_prefix = step_prefix.replace(
        "labels_dataset835", "metadata_dataset835", 1
    )

    print(f"[align] experiment={experiment} patch={patch_size_mm}mm")
    print(f"[align] aligned_patch root -> {aligned_patch}")
    print(f"[align] dense target -> {dense_labels_dir} (+ {dense_meta_dir})")
    print(f"[align] step obs     -> {aligned_patch}/{step_prefix}XX/")

    # Casefile (for 032 --test-casefile) is written INCREMENTALLY after each
    # case so a still-running / interrupted align already exposes ready cases.
    casefiles_dir = res["casefiles_dir"]
    casefiles_dir.mkdir(parents=True, exist_ok=True)
    out_cf = casefiles_dir / cfg["corrector_train_casefile"]

    def _flush_casefile():
        uniq = sorted(set(all_casenames))
        tmp = out_cf.with_suffix(out_cf.suffix + ".tmp")
        tmp.write_text("\n".join(uniq) + ("\n" if uniq else ""))
        tmp.replace(out_cf)   # atomic: readers never see a half-written file
        return uniq

    all_casenames: List[str] = []
    n_dense = n_step = n_fail = 0
    issues: List[str] = []

    for case_id, entry in sorted(manifest["cases"].items()):
        # (1) DENSE frame from the full-res 835 pred (gt_candidate_pred).
        gt_pred = entry.get("gt_candidate_pred", "")
        if gt_pred and Path(gt_pred).exists():
            try:
                cns = _align_and_write(
                    Path(gt_pred), case_id, "corrector_dense", patch_size_mm,
                    dense_labels_dir, dense_meta_dir, force=args.force,
                )
                all_casenames.extend(cns)
                n_dense += 1
            except Exception as e:  # noqa: BLE001
                n_fail += 1
                issues.append(f"{case_id} dense: {type(e).__name__}: {e}")
                continue
        else:
            issues.append(f"{case_id}: gt_candidate_pred missing ({gt_pred}); "
                          f"skipping (no dense frame -> CNISP can't run this case)")
            continue

        # (2) PER-STEP observation from the degraded 835 preds.
        for step_s, sinfo in entry.get("steps", {}).items():
            if not sinfo.get("kept"):
                continue
            step = int(step_s)
            seg = nnunet_pred_dir / f"{case_id}_step{step:02d}.nii.gz"
            if not seg.exists():
                issues.append(f"{case_id} step={step:02d}: nnUNet pred missing "
                              f"({seg}); run run_corrector_data.sh predict")
                continue
            step_dir = aligned_patch / f"{step_prefix}{step:02d}"
            meta_step_dir = aligned_patch / f"{meta_step_prefix}{step:02d}"
            try:
                _align_and_write(seg, case_id, f"corrector_step_{step:02d}",
                                 patch_size_mm, step_dir, meta_dir=meta_step_dir,
                                 force=args.force)
                n_step += 1
            except Exception as e:  # noqa: BLE001
                n_fail += 1
                issues.append(f"{case_id} step={step:02d}: {type(e).__name__}: {e}")

        # Case fully processed -> (re)write the casefile so far.
        _flush_casefile()

    uniq = _flush_casefile()

    if issues:
        print(f"\n[align] {len(issues)} issue(s):", file=sys.stderr)
        for s in issues[:25]:
            print(f"  - {s}", file=sys.stderr)
        if len(issues) > 25:
            print(f"  ... and {len(issues) - 25} more", file=sys.stderr)

    print(f"\n[align] dense cases aligned : {n_dense}")
    print(f"[align] step obs aligned    : {n_step}")
    print(f"[align] casenames           : {len(uniq)} -> {out_cf}")
    print(f"[align] hard failures       : {n_fail}")
    return 0 if n_fail == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
