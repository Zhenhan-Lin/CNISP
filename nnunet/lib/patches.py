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


def parse_step_tag(tag: str) -> Tuple[int, int]:
    """``"03"`` -> (3, 0); ``"03_o1"`` -> (3, 1) for the start-offset fan-out."""
    if "_o" in tag:
        s, o = tag.split("_o", 1)
        return int(s), int(o)
    return int(tag), 0


def iter_sparse_inputs(
    work_dir: Path,
    sparse_manifest: dict,
    experiment: str,
) -> Iterable[Tuple[str, str, Path]]:
    """Yield ``(step_tag, source_id, sparse_pred_path)`` for steps >= 2.

    ``step_tag`` is the manifest key, ``"XX"`` for the canonical start=0 and
    ``"XX_oN"`` for the high-eff_res start-offset fan-out, so the caller can
    name the output patch dir directly. Missing files are still yielded so the
    caller's loop can bookkeep them.
    """
    by_step = sparse_manifest.get("by_step", {})
    pred_root = work_dir / "prediction" / experiment
    for step_tag in sorted(by_step.keys()):
        step_pred_dir = pred_root / f"sparse_step_{step_tag}"
        for sid in sorted(by_step[step_tag]):
            yield step_tag, sid, step_pred_dir / f"{sid}.nii.gz"


def iter_step_01(
    work_dir: Path,
    source_ids: Iterable[str],
) -> Iterable[Tuple[str, str, Path]]:
    """Yield ``("01", source_id, dense_pred_path)`` for the dense baseline.

    step_01 inputs share their content with the dense canonical-aligned
    patches, but we still emit a dedicated step_01 patch directory so the
    deployment loader in ``engine/infer.py`` does a single uniform lookup per
    (case, step) without branching on step==1.
    """
    dense_dir = work_dir / "prediction" / "native"
    for sid in sorted(source_ids):
        yield "01", sid, dense_dir / f"{sid}.nii.gz"
