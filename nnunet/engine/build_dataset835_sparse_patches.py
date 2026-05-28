#!/usr/bin/env python3
"""Per-step canonical-align of Dataset835 SPARSE-CT preds for the deployment curve.

Inputs
------
* ``${work_dir}/prediction/sparse_step_{XX}/<sid>.nii.gz`` --
  Dataset835 prediction on the sparsified CT for one source, on the
  sparse CT's voxel grid (through-plane spacing already multiplied by
  step). Produced by ``nnunet/run_predict_sparse_sweep.sh``.
* ``${work_dir}/input/sparse_manifest.json`` -- per-(source, step)
  sparsification bookkeeping from ``nnunet/data_prep/sparsify_inputs.py``.
* ``${work_dir}/prediction/sweep_manifest.json`` (optional) --
  consulted only to fill in step_01, since step_01 is symlinked, not
  sparsified, and lives under ``prediction/native/``.

Outputs
-------
* ``${aligned_dir}/${labels_dataset835_step_prefix}{XX}/{casename}.nii.gz``
  -- canonical-aligned orbital patches (one per eye per source per
  step) carved out of the sparse-grid Dataset835 prediction.

These patches are the latent-opt INPUT for the Option C deployment
curve (test_label_source=nnunet_pred). Their through-plane voxel count
shrinks with step (e.g. ~6 slices at step=11 with 1 mm orig spacing) --
that's intentional: CNISP sees exactly the slices nnUNet saw.

The canonical-align crop is computed fresh per (source, step) from the
sparse pred's own globe CC. When nnUNet at high sparsity drops a globe
entirely, ``align_single_case`` returns an empty / single-eye list and
the corresponding (case, step) row is skipped by ``engine/infer.py``.
This is the deployment-quality signal we want surfaced, not papered
over with a fallback to the dense crop.

Skip-if-done
------------
Per (source, eye, step) we skip when the label NIfTI already exists
unless ``--force`` is passed.

Usage
-----
    python nnunet/engine/build_dataset835_sparse_patches.py --config nnunet/configs.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import nibabel as nib
import numpy as np

# Make ``nnunet.*`` importable when run as ``python nnunet/engine/...``.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nnunet.helpers.config import (  # noqa: E402
    add_cnisp_src_to_syspath,
    load_yaml,
)
from nnunet.helpers.patch_size import resolve_patch_size_mm  # noqa: E402

add_cnisp_src_to_syspath(__file__)

from data_prep.canonical_align import (  # noqa: E402
    align_single_case,
    infer_patch_size_mm,
)


def _iter_sparse_inputs(
    work_dir: Path,
    sparse_manifest: dict,
) -> Iterable[Tuple[int, str, Path]]:
    """Yield ``(step_size, source_id, sparse_pred_path)`` for steps >= 2."""
    by_step = sparse_manifest.get("by_step", {})
    pred_root = work_dir / "prediction"
    for step_tag in sorted(by_step.keys()):
        step = int(step_tag)
        step_pred_dir = pred_root / f"sparse_step_{step_tag}"
        for sid in sorted(by_step[step_tag]):
            sparse_pred = step_pred_dir / f"{sid}.nii.gz"
            if not sparse_pred.exists():
                # Missing rows are bookkept later by the main loop.
                yield step, sid, sparse_pred
                continue
            yield step, sid, sparse_pred


def _iter_step_01(
    work_dir: Path,
    source_ids: Iterable[str],
) -> Iterable[Tuple[int, str, Path]]:
    """Yield ``(1, source_id, dense_pred_path)`` for the dense baseline.

    step_01 inputs share their content with the dense canonical-aligned
    patches written by build_dataset835_canonical_patches.py, but we
    still emit a dedicated step_01 patch directory so the deployment
    loader in ``engine/infer.py`` does a single uniform lookup per
    (case, step) without branching on step==1.
    """
    dense_dir = work_dir / "prediction" / "native"
    for sid in sorted(source_ids):
        yield 1, sid, dense_dir / f"{sid}.nii.gz"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="nnunet/configs.yaml")
    ap.add_argument("--force", action="store_true",
                    help="Re-canonical-align even when patch already exists.")
    ap.add_argument(
        "--patch-size",
        type=float,
        default=None,
        help="Physical extent (mm) of the canonical-aligned cubic patch. "
             "Defaults to the value recorded in the existing CNISP training "
             "metadata under aligned_dir/metadata/ so the sparse latent-opt "
             "input grid matches the size the MLP was trained on. Override "
             "only when you intentionally want a different physical extent.",
    )
    ap.add_argument("--skip-step-01", action="store_true",
                    help="Don't emit step_01/ patches (e.g. when the "
                         "dense baseline isn't part of this run).")
    args = ap.parse_args()

    cfg = load_yaml(Path(args.config))
    cnisp_paths = load_yaml(Path(cfg["cnisp_paths_yaml"]))
    work_dir = Path(cfg["work_dir"])
    aligned_dir = Path(cnisp_paths["aligned_dir"])
    prefix = cnisp_paths.get(
        "labels_dataset835_step_prefix", "labels_dataset835_step_"
    )

    # Pin the patch size to whatever the model was trained on, unless
    # the caller explicitly overrides it. Mismatching patch sizes
    # between the latent-opt input and the trained MLP's coordinate
    # frame would translate the predicted globe by
    # (training_patch - this_patch) / 2 millimetres per axis.
    train_meta_dir = aligned_dir / "metadata"
    patch_size_mm = resolve_patch_size_mm(
        args.patch_size, train_meta_dir,
        log_prefix="dataset835_sparse",
        infer_fn=infer_patch_size_mm,
    )

    sparse_manifest_path = work_dir / "input" / "sparse_manifest.json"
    if not sparse_manifest_path.exists():
        print(f"[dataset835_sparse] {sparse_manifest_path} missing -- "
              f"run nnunet/data_prep/sparsify_inputs.py first.",
              file=sys.stderr)
        return 2
    with open(sparse_manifest_path) as f:
        sparse_manifest = json.load(f)

    # step_01 = dense pred (no sparsification, always available). Source
    # list comes from source_to_path.json so we include the full 31
    # sources at step_01 -- even sources sparsify_inputs.py rejected for
    # axis-detection reasons still have a dense pred to align.
    # Higher steps come from sparse_manifest.by_step (which omits rejected
    # sources for step >= 2, by design).
    src_to_path_p = work_dir / "source_to_path.json"
    if not src_to_path_p.exists():
        print(f"[dataset835_sparse] {src_to_path_p} missing -- "
              f"run nnunet/data_prep/prepare_inputs.py first.",
              file=sys.stderr)
        return 2
    with open(src_to_path_p) as f:
        all_source_ids = sorted(json.load(f))

    # ── Iterate over (step, source) ──────────────────────────────
    n_written = 0
    n_skipped_existing = 0
    n_missing_pred = 0
    n_dropped_eye = 0
    n_failed = 0
    issues: List[str] = []

    work_items: List[Tuple[int, str, Path]] = []
    if not args.skip_step_01:
        work_items.extend(_iter_step_01(work_dir, all_source_ids))
    work_items.extend(_iter_sparse_inputs(work_dir, sparse_manifest))

    seen_step_dirs: set = set()
    for step, sid, seg_path in work_items:
        step_dir = aligned_dir / f"{prefix}{step:02d}"
        if step not in seen_step_dirs:
            step_dir.mkdir(parents=True, exist_ok=True)
            seen_step_dirs.add(step)

        if not seg_path.exists():
            n_missing_pred += 1
            issues.append(f"step={step:02d} {sid}: pred missing at {seg_path}")
            continue

        # Skip when both eyes for this (sid, step) already on disk.
        both_eyes_done = (
            (step_dir / f"{sid}_OD.nii.gz").exists()
            and (step_dir / f"{sid}_OS.nii.gz").exists()
        )
        if both_eyes_done and not args.force:
            n_skipped_existing += 1
            continue

        try:
            results = align_single_case(
                seg_path=str(seg_path),
                source_id=sid,
                source=f"dataset835_step_{step:02d}",
                patch_size_mm=patch_size_mm,
            )
        except Exception as e:  # noqa: BLE001
            n_failed += 1
            issues.append(f"step={step:02d} {sid}: align_single_case raised "
                          f"{type(e).__name__}: {e}")
            continue

        if not results:
            n_failed += 1
            issues.append(f"step={step:02d} {sid}: no eyes detected "
                          f"(nnUNet may have dropped both globes at this "
                          f"sparsity)")
            continue
        if len(results) == 1:
            n_dropped_eye += 1
            issues.append(f"step={step:02d} {sid}: only one eye detected")

        for patch, pa, meta in results:
            out_path = step_dir / f"{meta.casename}.nii.gz"
            if out_path.exists() and not args.force:
                continue
            nib.save(
                nib.Nifti1Image(patch.astype(np.uint8), pa),
                str(out_path),
            )
            n_written += 1

    if issues:
        print(f"\n[dataset835_sparse] {len(issues)} issue(s):", file=sys.stderr)
        for line in issues[:25]:
            print(f"  - {line}", file=sys.stderr)
        if len(issues) > 25:
            print(f"  ... and {len(issues) - 25} more", file=sys.stderr)

    print(f"\n[dataset835_sparse] wrote {n_written} patch(es); "
          f"{n_skipped_existing} (source,step) pairs already complete; "
          f"{n_missing_pred} pred files missing; "
          f"{n_dropped_eye} (source,step) with one eye dropped; "
          f"{n_failed} hard failures.")

    # Hard failures are fatal; missing predictions / dropped eyes are
    # informational (the deployment curve is allowed to skip them).
    return 0 if n_failed == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
