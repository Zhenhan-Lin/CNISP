"""
Test-time label-source resolver for the Option C deployment curve.

Centralises the path conventions shared by:

* ``engine/infer.py`` -- when assembling latent-opt input + dense Dice
  target for each (case, step) under ``test_label_source=nnunet_pred``.
* ``engine/visualize.py`` and ``scripts/04_visualization.py`` -- when
  resolving ``recon_dir`` for a given ``run_tag``.
* ``nnunet/build_cnisp_native_sweep.py`` and
  ``nnunet/compare_native.py`` -- when finding per-step manifests for
  each CNISP run.

Two test_label_source modes
---------------------------

* ``atlas_gt`` (default ceiling curve)
    - Dense Dice target  : ``aligned_dir/labels/<casename>.nii.gz``
    - Latent-opt input   : derived from the same patch via
                           ``sparsen_volume`` (the existing path).
    - Native inversion   : ``aligned_dir/metadata/<casename>.json``

* ``nnunet_pred`` (deployment curve)
    - Dense Dice target  :
        atlas cases : ``aligned_dir/labels/<casename>.nii.gz`` (manual GT)
        chk_*  cases : ``aligned_dir/labels_dataset835/<casename>.nii.gz``
                       (Dataset835 dense pred canonical-aligned)
    - Latent-opt input   :
        ``aligned_dir/labels_dataset835_step_{XX}/<casename>.nii.gz``
        (Dataset835 sparse-CT pred per step, canonical-aligned by
        ``nnunet/build_dataset835_sparse_patches.py``).
    - Native inversion   :
        atlas cases : ``aligned_dir/metadata/<casename>.json`` (existing)
        chk_*  cases : ``aligned_dir/metadata_dataset835/<casename>.json``
                       (sidecar JSON produced alongside the dense
                       canonical-aligned target above)

* ``real_pair`` (Turella sim3 — REAL paired acquisitions)
    The low-res input and the hi-res GT are SEPARATE real acquisitions in
    different physical frames. No ``simulation/`` degradation is applied
    (the scanner already produced the anisotropy). Each is canonical-aligned
    independently (registration-free); at eval time the CNISP-reconstructed
    mask is rigidly registered to the GT mask (post-hoc) to absorb the
    subject's inter-acquisition repositioning, following Turella et al.
    - Dense Dice target  : ``aligned_dir/labels_realpair_gt/<casename>.nii.gz``
                           (hi-res GT: manual annotation or nnUNet-on-hires)
    - Latent-opt input   : ``aligned_dir/labels_realpair_input/<casename>.nii.gz``
                           (nnUNet pred on the REAL low-res scan, aligned)
    - Native inversion   : ``aligned_dir/metadata_realpair_gt/<casename>.json``
    - Single observation per case (no resolution sweep).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import nibabel as nib
import numpy as np
import torch


VALID_LABEL_SOURCES = ("atlas_gt", "nnunet_pred", "real_pair")

# Experiment dimension (simulation strategy). Inserted as a directory layer
# under runs/ -- runs/<experiment>/<run_tag>/ -- so thin / thick / real
# results coexist on disk instead of overwriting one another.
VALID_EXPERIMENTS = ("thin", "thick", "real", "fov")


def resolve_experiment(params: dict) -> str:
    """Single source of truth for the experiment dir name (thin|thick|real).

    Precedence:
      1. explicit ``params['experiment']`` (set by the pipeline / CLI),
      2. ``real`` whenever ``test_label_source == real_pair`` (the real
         paired-data line is its own degradation regime),
      3. ``params['sweep_mode']`` (thin|thick) for the latent-opt sweep,
      4. ``thin`` (legacy default).
    """
    exp = params.get("experiment")
    if exp:
        return str(exp)
    if str(params.get("test_label_source", "atlas_gt")) == "real_pair":
        return "real"
    return str(params.get("sweep_mode", "thin"))


def exp_step_prefix(base_prefix: str, experiment: str) -> str:
    """Inject the experiment token into the per-step sparse-label prefix.

    ``labels_dataset835_step_`` + ``thin`` -> ``labels_dataset835_thin_step_``
    so the deployment-curve (nnunet_pred) input patches are exp-keyed and
    thin / thick never collide one directory level deeper.
    """
    if base_prefix.endswith("_step_"):
        return f"{base_prefix[:-len('_step_')]}_{experiment}_step_"
    return f"{base_prefix}{experiment}_"


@dataclass
class RunLayout:
    """All paths a single inference run reads/writes.

    Concentrating this here means ``engine/infer.py``, the visualiser,
    the legacy backfill (``build_cnisp_native_sweep.py``), and the
    comparison driver (``compare_native.py``) all agree about where
    artifacts live for a given ``(model_name, run_tag)``.
    """

    model_name: str
    run_tag: str
    test_label_source: str
    experiment: str                 # thin | thick | real (runs/<experiment>/)

    # Inputs (read from aligned_dir)
    aligned_dir: Path
    labels_dir: Path                # ceiling-curve label directory (always labels/)
    metadata_dir: Path              # ceiling-curve metadata (always metadata/)
    labels_dataset835_dir: Path     # nnunet_pred: chk_* dense target
    metadata_dataset835_dir: Path   # nnunet_pred: chk_* native-mapping metadata
    labels_dataset835_step_prefix: Path  # template; per-step dir = prefix + "XX"

    # real_pair (Turella sim3): real low-res input + separate hi-res GT.
    # Both are canonical-aligned (registration-free); the predicted mask is
    # rigidly registered to the GT mask at eval time (post-hoc).
    labels_realpair_input_dir: Path   # nnUNet pred on REAL low-res, aligned
    labels_realpair_gt_dir: Path      # hi-res GT (manual or nnUNet-hires), aligned
    metadata_realpair_gt_dir: Path    # native-mapping metadata for the GT frame

    # Outputs (written under output_basedir/<model_name>/runs/<run_tag>/)
    output_dir: Path


def _path(d: Path | str) -> Path:
    return d if isinstance(d, Path) else Path(d)


def _resolve_aligned_subdirs(params: dict, aligned_dir: Path) -> dict:
    """Pull subdirectory names from params with sensible defaults."""
    return {
        "labels_dirname": params.get("labels_dirname", "labels"),
        "metadata_dirname": params.get("metadata_dirname", "metadata"),
        "labels_dataset835_dirname": params.get(
            "labels_dataset835_dirname", "labels_dataset835"
        ),
        "metadata_dataset835_dirname": params.get(
            "metadata_dataset835_dirname", "metadata_dataset835"
        ),
        "labels_dataset835_step_prefix": params.get(
            "labels_dataset835_step_prefix", "labels_dataset835_step_"
        ),
        "labels_realpair_input_dirname": params.get(
            "labels_realpair_input_dirname", "labels_realpair_input"
        ),
        "labels_realpair_gt_dirname": params.get(
            "labels_realpair_gt_dirname", "labels_realpair_gt"
        ),
        "metadata_realpair_gt_dirname": params.get(
            "metadata_realpair_gt_dirname", "metadata_realpair_gt"
        ),
    }


def build_run_layout(params: dict) -> RunLayout:
    """Resolve the on-disk paths for one inference run.

    Reads from ``params`` (merged paths.yaml + train yaml + test yaml):

    * ``aligned_dir``, ``output_basedir``, ``model_name`` (required)
    * ``run_tag``                 (default ``atlas_gt``)
    * ``test_label_source``       (default ``atlas_gt``)
    * Aligned-subdir name knobs (all have safe defaults; see
      ``configs/paths.yaml``).

    For backward compatibility with on-disk runs that pre-date Option C,
    ``output_dir`` always lands at ``output_basedir/<model_name>/runs/<run_tag>/``.
    Users migrating an existing run should move its artifacts into
    ``runs/atlas_gt/`` so the default code path resumes correctly.
    """
    aligned_dir = _path(params["aligned_dir"])
    sub = _resolve_aligned_subdirs(params, aligned_dir)

    model_name = params["model_name"]
    run_tag = str(params.get("run_tag", "atlas_gt"))
    test_label_source = str(params.get("test_label_source", "atlas_gt"))
    if test_label_source not in VALID_LABEL_SOURCES:
        raise ValueError(
            f"test_label_source={test_label_source!r} not in "
            f"{VALID_LABEL_SOURCES}. Update configs/test_default.yaml."
        )

    experiment = resolve_experiment(params)
    if experiment not in VALID_EXPERIMENTS:
        raise ValueError(
            f"experiment={experiment!r} not in {VALID_EXPERIMENTS}. "
            f"Set params['experiment'] / sweep_mode to one of these."
        )

    output_basedir = _path(params["output_basedir"])
    # runs/<experiment>/<run_tag>/ -- the experiment layer keeps thin / thick
    # / real result trees side by side instead of overwriting each other.
    output_dir = output_basedir / model_name / "runs" / experiment / run_tag

    # Sparse deployment-curve patches are exp-keyed too (thin vs thick produce
    # different nnUNet sparse preds, so their canonical-aligned patches differ).
    step_prefix = exp_step_prefix(
        sub["labels_dataset835_step_prefix"], experiment
    )

    return RunLayout(
        model_name=model_name,
        run_tag=run_tag,
        test_label_source=test_label_source,
        experiment=experiment,
        aligned_dir=aligned_dir,
        labels_dir=aligned_dir / sub["labels_dirname"],
        metadata_dir=aligned_dir / sub["metadata_dirname"],
        labels_dataset835_dir=aligned_dir / sub["labels_dataset835_dirname"],
        metadata_dataset835_dir=aligned_dir / sub["metadata_dataset835_dirname"],
        labels_dataset835_step_prefix=(aligned_dir / step_prefix),
        labels_realpair_input_dir=aligned_dir / sub["labels_realpair_input_dirname"],
        labels_realpair_gt_dir=aligned_dir / sub["labels_realpair_gt_dirname"],
        metadata_realpair_gt_dir=aligned_dir / sub["metadata_realpair_gt_dirname"],
        output_dir=output_dir,
    )


def step_input_patch_path(
    layout: RunLayout, casename: str, step: int, start: int = 0,
) -> Path:
    """Where to read the latent-opt input patch for a given (case, step, start).

    ``nnunet_pred`` : per-step Dataset835 sparse-CT patch. ``start``>0 selects
                      the start-offset fan-out dir (``..._step_03_o1/``); the
                      canonical ``start==0`` keeps the legacy ``..._step_03/``.
    ``real_pair``   : single real low-res nnUNet-pred patch (step is
                      informational; the real anisotropy is fixed per case).
    """
    if layout.test_label_source == "real_pair":
        return layout.labels_realpair_input_dir / f"{casename}.nii.gz"
    ostr = "" if int(start) == 0 else f"_o{int(start)}"
    step_dir = Path(
        f"{layout.labels_dataset835_step_prefix.as_posix()}{step:02d}{ostr}"
    )
    return step_dir / f"{casename}.nii.gz"


def dense_target_paths(
    layout: RunLayout,
    casename: str,
) -> Tuple[Path, Path]:
    """Where to read the dense Dice target and its native-mapping metadata.

    Returns (label_path, metadata_path).
    * atlas_gt / atlas cases  : ceiling-curve files (manual GT).
    * nnunet_pred chk_*       : Dataset835 dense canonical-aligned pred.
    * real_pair               : separate hi-res GT, canonical-aligned.
    """
    if layout.test_label_source == "real_pair":
        return (
            layout.labels_realpair_gt_dir / f"{casename}.nii.gz",
            layout.metadata_realpair_gt_dir / f"{casename}.json",
        )
    is_atlas = casename.startswith("atlas_")
    if layout.test_label_source == "atlas_gt" or is_atlas:
        return (
            layout.labels_dir / f"{casename}.nii.gz",
            layout.metadata_dir / f"{casename}.json",
        )
    return (
        layout.labels_dataset835_dir / f"{casename}.nii.gz",
        layout.metadata_dataset835_dir / f"{casename}.json",
    )


def load_patch_as_label_tensor(path: Path) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load a canonical-aligned NIfTI as (volume, spacing, offset).

    spacing: [3] column norms of the affine's linear part.
    offset:  [3] = spacing / 2 to match the
             ``coord = voxel_idx * spacing + spacing/2`` convention used
             by ``OrbitalImplicitDataset`` and the latent-opt path.

    The labels are returned as float32 (consistent with
    ``load_orbital_volumes``).
    """
    img = nib.load(str(path))
    vol = np.asarray(img.dataobj, dtype=np.float32)
    aff = img.affine
    spacing = np.sqrt((aff[:3, :3] ** 2).sum(axis=0)).astype(np.float32)
    spacing_t = torch.from_numpy(spacing)
    offset_t = spacing_t / 2.0
    return torch.from_numpy(vol), spacing_t, offset_t


def case_input_kind(layout: RunLayout, casename: str) -> str:
    """Two-character tag for stdout/logging: 'AT' atlas | 'CH' chk_*."""
    return "AT" if casename.startswith("atlas_") else "CH"
