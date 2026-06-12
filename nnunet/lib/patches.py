#!/usr/bin/env python3
"""Output-directory layouts + input iteration for the patch builders.

Pure path/bookkeeping helpers shared by the ``engine/build_*_patches.py``
drivers, kept out of those orchestrators so each ``run(args)`` is just the
per-(source, eye, step) align/save loop. None of these touch
``canonical_align`` (the actual alignment call stays in the driver, which
sets up the CNISP ``sys.path`` first).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple


def dataset835_layout(cnisp_paths: dict, aligned_dir: Path) -> Dict[str, Path]:
    """`{labels_dir, meta_dir}` for the dense Dataset835 canonical patches."""
    labels = cnisp_paths.get("labels_dataset835_dirname", "labels_dataset835")
    meta = cnisp_paths.get("metadata_dataset835_dirname", "metadata_dataset835")
    return {
        "labels_dir": aligned_dir / labels,
        "meta_dir":   aligned_dir / meta,
    }


def realpair_layout(cnisp_paths: dict, aligned_dir: Path) -> Dict[str, Path]:
    """`{input_dir, gt_dir, gt_meta_dir}` for the real-pair (Turella) patches."""
    return {
        "input_dir": aligned_dir / cnisp_paths.get(
            "labels_realpair_input_dirname", "labels_realpair_input"
        ),
        "gt_dir": aligned_dir / cnisp_paths.get(
            "labels_realpair_gt_dirname", "labels_realpair_gt"
        ),
        "gt_meta_dir": aligned_dir / cnisp_paths.get(
            "metadata_realpair_gt_dirname", "metadata_realpair_gt"
        ),
    }


def iter_sparse_inputs(
    work_dir: Path,
    sparse_manifest: dict,
    experiment: str,
) -> Iterable[Tuple[int, str, Path]]:
    """Yield ``(step_size, source_id, sparse_pred_path)`` for steps >= 2.

    Missing files are still yielded so the caller's loop can bookkeep them.
    """
    by_step = sparse_manifest.get("by_step", {})
    pred_root = work_dir / "prediction" / experiment
    for step_tag in sorted(by_step.keys()):
        step = int(step_tag)
        step_pred_dir = pred_root / f"sparse_step_{step_tag}"
        for sid in sorted(by_step[step_tag]):
            yield step, sid, step_pred_dir / f"{sid}.nii.gz"


def iter_step_01(
    work_dir: Path,
    source_ids: Iterable[str],
) -> Iterable[Tuple[int, str, Path]]:
    """Yield ``(1, source_id, dense_pred_path)`` for the dense baseline.

    step_01 inputs share their content with the dense canonical-aligned
    patches, but we still emit a dedicated step_01 patch directory so the
    deployment loader in ``engine/infer.py`` does a single uniform lookup per
    (case, step) without branching on step==1.
    """
    dense_dir = work_dir / "prediction" / "native"
    for sid in sorted(source_ids):
        yield 1, sid, dense_dir / f"{sid}.nii.gz"
