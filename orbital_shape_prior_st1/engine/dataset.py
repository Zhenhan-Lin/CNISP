"""
Dataset for orbital implicit shape prior training.

Supports two training strategies via config:

  Strategy A (original Amiranashvili convention):
      num_sparsify_offsets: 1
      val_casefile: null          # val split from train scans
    Each scan → 1 training sample. Val uses the same scans with
    complementary sparsification (secondary step=2 split).

  Strategy B (multi-offset + separate val):
      num_sparsify_offsets: 4     # = slice_step_size
      val_casefile: val_cases.txt # disjoint scan pool
    Each scan → N training samples (one per offset), each with its own
    latent code. Val uses completely separate scans.

Coordinate convention (following Amiranashvili et al.):
    - offset = spacing/2  (align_corners=False)
    - coordinates are in mm, physical space
    - image_size = INNER_PATCH_SIZE_MM (fixed; see below)

Two-patch system: 80 mm disk patch + 64 mm visible-centroid inner crop
----------------------------------------------------------------------
Disk patches (produced by ``data_prep/canonical_align.py``) are 80 mm cubic
crops centred on each eye's dense-LCC centroid, with the contralateral eye's
voxels already zeroed out by single-eye LCC cleanup.

At training and inference time each sample is sparsified along its
through-plane axis. The OBSERVED foreground centroid drifts away from the
dense centroid by up to ``step_size * spacing / 2`` — typically several mm,
sometimes 8+ mm at large step sizes. The implicit MLP must learn to
reconstruct the dense shape from this drifted view, so we feed it an inner
crop of **fixed 64 mm extent centred on the visible-LCC centroid**, not the
raw 80 mm disk patch. The 16 mm buffer (80 − 64) absorbs the drift so the
inner crop never falls off the disk patch.

The MLP's ``image_size`` buffer is therefore fixed at 64 mm and
``latent_coords = 32 mm``. Inference unmap composes two steps:
    64 mm sub-patch  --place_at(sub_crop_in_disk)-->  80 mm disk patch
                     --place_at(crop_slices_in_full)-->  full-head volume
"""

import hashlib
import json
import sys
import time
from enum import IntEnum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
from nibabel.processing import resample_from_to
from torch.utils import data

# Ensure repo root is on sys.path so `simulation/` is importable.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data_prep.canonical_align import extract_single_eye_lcc
from data_prep.sparsify import resolve_slice_step_axes, sparsen_volume
from simulation.degradation import degrade_thin, degrade_thick
from simulation.slice_profile import get_kernel

SPRSF_SEED = 1
SPLIT_SEED = 2
SPRSF_VAL_SEED = 3
BANK_SEED = 42


def adaptive_steps_for_bank(
    spacing_axis: float,
    target_eff_res_increment_mm: float = 2.0,
    max_eff_resolution_mm: float = 12.0,
) -> List[int]:
    """Compute per-case step list for the degradation bank (training).

    Uses ~half the density of the test sweep (increment 2.0mm vs 1.0mm)
    to keep training tractable while covering the same eff-res range.
    """
    if spacing_axis <= 0:
        return [1]
    delta_step = max(1, int(round(target_eff_res_increment_mm / float(spacing_axis))))
    steps: List[int] = [1]
    for k in range(1, 20):
        s = 1 + k * delta_step
        if s * spacing_axis > max_eff_resolution_mm:
            break
        steps.append(s)
    return steps


def _bank_cache_hash(bank_config: Dict) -> str:
    """Deterministic hash of the bank config for cache validation."""
    cfg_str = json.dumps(bank_config, sort_keys=True)
    return hashlib.md5(cfg_str.encode()).hexdigest()[:12]

# Physical side length of the inner crop the MLP actually sees. Disk patches
# from canonical_align are 80 mm; this 64 mm sub-patch is centred on the
# sparsify-time visible-LCC centroid. Keep in sync with the MLP's
# ``image_size`` buffer (set when training the AutoDecoder).
INNER_PATCH_SIZE_MM = 64.0


class PhaseType(IntEnum):
    TRAIN = 0
    VAL = 1
    INF = 2


def load_casenames(filepath: Path) -> List[str]:
    with open(filepath) as f:
        return [line.strip() for line in f if line.strip()]


def load_orbital_volumes(labels_dir: Path, casenames: List[str]):
    volumes, spacings = [], []
    for cn in casenames:
        nii_path = labels_dir / f"{cn}.nii.gz"
        if not nii_path.exists():
            raise FileNotFoundError(f"Not found: {nii_path}")
        img = nib.load(str(nii_path))
        vol = np.asarray(img.dataobj, dtype=np.float32)
        aff = img.affine
        # Column norms over diagonal: robust to residual off-diagonal terms;
        # equals the diagonal magnitude when the affine is properly axis-aligned.
        spacing = np.sqrt((aff[:3, :3] ** 2).sum(axis=0)).astype(np.float32)
        volumes.append(torch.from_numpy(vol))
        spacings.append(torch.from_numpy(spacing))
    return volumes, spacings


def compute_centroid_mm(volume: torch.Tensor, spacing: torch.Tensor,
                        offset: torch.Tensor) -> torch.Tensor:
    """
    Center of mass of foreground voxels (volume > 0) in physical (mm)
    coordinates of the patch-local frame: `coord = voxel_idx * spacing + offset`.

    Cheap: O(N_fg) on the sparse volume's foreground voxels.
    Returns a [3] tensor; NaN per axis if no foreground exists.
    """
    fg = (volume > 0).nonzero(as_tuple=False)  # [N_fg, 3], voxel indices
    if fg.numel() == 0:
        return torch.full((3,), float("nan"), dtype=torch.float32)
    centroid_voxel = fg.to(torch.float32).mean(dim=0)
    return centroid_voxel * spacing + offset


def compute_visible_lcc_centroid_mm(
    volume_sparse: torch.Tensor,
    spacing_sparse: torch.Tensor,
    offset_sparse: torch.Tensor,
) -> Tuple[Optional[torch.Tensor], int, int]:
    """Centroid of the sparsified volume's largest connected component.

    Used to position the 64 mm inner crop centre. Falls back gracefully when
    no foreground is visible (returns ``None`` so the caller can centre on
    the disk patch's geometric centre instead).

    Returns
    -------
    centroid_mm : torch.Tensor[3] or None
        Centroid in disk-patch-local mm: ``voxel_idx * spacing + offset``.
    lcc_voxel_count : int
        Voxels in the LCC (for QC).
    total_fg_count : int
        Voxels in the full visible foreground (for QC).
    """
    fg_mask = (volume_sparse > 0).cpu().numpy()
    _, lcc_centroid_voxel, lcc_count, total_fg = extract_single_eye_lcc(fg_mask)
    if lcc_centroid_voxel is None:
        return None, 0, total_fg
    cv = torch.from_numpy(lcc_centroid_voxel.astype(np.float32))
    return cv * spacing_sparse + offset_sparse, lcc_count, total_fg


def pad_or_crop_to_voxel_bbox(
    volume: torch.Tensor, lo_vox: np.ndarray, shape_vox: np.ndarray,
) -> torch.Tensor:
    """Extract a fixed-shape sub-volume at voxel position ``lo_vox`` from
    ``volume``, zero-padding when the bbox spills past the volume bounds.

    The returned tensor always has shape ``tuple(shape_vox)``; out-of-bounds
    regions read as 0 (background sentinel). This lets the inner crop hold
    its fixed 64 mm physical extent even when the visible-LCC centroid sits
    near a disk-patch edge.
    """
    lo_vox = np.asarray(lo_vox, dtype=np.int64)
    shape_vox = np.asarray(shape_vox, dtype=np.int64)
    hi_vox = lo_vox + shape_vox
    vol_shape = np.asarray(volume.shape, dtype=np.int64)
    # Clamp the source slice into the input volume's bounds.
    src_lo = np.maximum(lo_vox, 0)
    src_hi = np.minimum(hi_vox, vol_shape)
    # Destination indices into the output (sub-patch) frame.
    dst_lo = src_lo - lo_vox
    dst_hi = dst_lo + (src_hi - src_lo)
    out = torch.zeros(tuple(shape_vox.tolist()), dtype=volume.dtype)
    if np.all(src_hi > src_lo):
        out[dst_lo[0]:dst_hi[0], dst_lo[1]:dst_hi[1], dst_lo[2]:dst_hi[2]] = \
            volume[src_lo[0]:src_hi[0], src_lo[1]:src_hi[1], src_lo[2]:src_hi[2]]
    return out


def inner_crop_64mm(
    volume_sparse: torch.Tensor,
    spacing_sparse: torch.Tensor,
    offset_sparse: torch.Tensor,
    volume_dense: torch.Tensor,
    spacing_dense: torch.Tensor,
    offset_dense: torch.Tensor,
    inner_size_mm: float = INNER_PATCH_SIZE_MM,
) -> Dict[str, object]:
    """Crop a ``inner_size_mm`` cubic sub-patch around the visible-LCC centroid.

    Both ``volume_sparse`` and ``volume_dense`` live in the SAME disk-patch
    coordinate frame (they differ only in spacing along the through-plane
    axis). We compute the centroid on the sparse view (= what the MLP
    actually sees), then express the same physical sub-patch in both the
    sparse and dense voxel grids so they remain aligned voxel-for-voxel in
    physical mm.

    Returns a dict with everything callers need:
        sub_sparse                : [Nx_s, Ny_s, Nz_s] inner crop of sparse
        sub_dense                 : [Nx_d, Ny_d, Nz_d] inner crop of dense
        sub_offset_sparse_local   : [3] mm, sparse voxel-0 centre expressed in
                                    the SHARED dense-corner frame (so the
                                    sparse-fit latent decodes correctly on the
                                    dense grid; equals spacing_dense/2 off the
                                    through-plane axis)
        sub_offset_dense_local    : [3] mm, sub-patch-local origin in dense
                                    (= spacing_dense / 2, voxel-center conv.)
        sub_crop_lo_vox_dense     : [3] int, dense-frame lo voxel of the
                                    sub-patch within the 80 mm disk patch
                                    (used by inference unmap)
        sub_crop_shape_vox_dense  : [3] int, dense-frame voxel shape
        sub_origin_mm_in_disk     : [3] mm, disk-patch-local mm origin of
                                    the sub-patch (for diagnostics)
        visible_lcc_voxel_count   : int
        visible_total_fg_count    : int
    """
    sp = spacing_sparse.numpy().astype(np.float32)
    of_s = offset_sparse.numpy().astype(np.float32)
    sd = spacing_dense.numpy().astype(np.float32)
    of_d = offset_dense.numpy().astype(np.float32)

    centroid_mm, lcc_count, total_fg = compute_visible_lcc_centroid_mm(
        volume_sparse, spacing_sparse, offset_sparse,
    )
    if centroid_mm is None:
        # No visible foreground: fall back to the disk patch's geometric
        # centre. Disk patch's local mm extent along axis k is
        # vol_dense.shape[k] * spacing_dense[k] (== 80 mm by construction).
        disk_extent_mm = np.array(volume_dense.shape, dtype=np.float32) * sd
        centroid_np = disk_extent_mm / 2.0
    else:
        centroid_np = centroid_mm.numpy().astype(np.float32)

    sub_origin_mm = centroid_np - inner_size_mm / 2.0  # disk-frame mm

    sub_shape_vox_sparse = np.maximum(
        np.round(inner_size_mm / sp).astype(np.int64), 1
    )
    sub_lo_vox_sparse = np.round((sub_origin_mm - of_s) / sp).astype(np.int64)
    sub_sparse = pad_or_crop_to_voxel_bbox(
        volume_sparse, sub_lo_vox_sparse, sub_shape_vox_sparse,
    )

    sub_shape_vox_dense = np.maximum(
        np.round(inner_size_mm / sd).astype(np.int64), 1
    )
    sub_lo_vox_dense = np.round((sub_origin_mm - of_d) / sd).astype(np.int64)
    sub_dense = pad_or_crop_to_voxel_bbox(
        volume_dense, sub_lo_vox_dense, sub_shape_vox_dense,
    )

    # ── Shared-origin sub-patch frame (fixes the through-plane drift) ──
    # The latent is fitted on the SPARSE grid and decoded on the DENSE grid,
    # so both grids MUST express the same physical point with the same local
    # coordinate. We anchor both to the dense sub-patch's lower-corner origin
    #   O_disk = sub_lo_vox_dense * sd + of_d - sd/2          (disk-frame mm)
    # Dense keeps the voxel-centre convention (local origin = sd/2). The sparse
    # voxel-0 centre sits at disk-mm (sub_lo_vox_sparse*sp + of_s); its local
    # offset is that minus O_disk.
    #
    # The OLD code used ``spacing_sparse / 2`` for the sparse offset too. Along
    # the through-plane axis spacing_sparse = step * sd, so that convention put
    # the sparse origin at step*sd/2 instead of the physically-correct corner,
    # drifting the sparse and dense frames apart by ~sd*(step-1)/2 (plus the
    # sub_lo rounding mismatch). The fitted latent then decoded onto a
    # translated dense grid, shifting every step>1 prediction along the
    # sparsified axis (step=1 is unaffected because step*sd == sd). In-plane
    # axes are unaffected because spacing is unchanged there, so this reduces
    # to spacing_dense/2 (== old behaviour) off the through-plane axis.
    sub_lo_sparse_t = torch.from_numpy(sub_lo_vox_sparse).to(spacing_sparse.dtype)
    sub_lo_dense_t = torch.from_numpy(sub_lo_vox_dense).to(spacing_dense.dtype)
    sub_offset_sparse_local = (
        sub_lo_sparse_t * spacing_sparse + offset_sparse
        - (sub_lo_dense_t * spacing_dense + offset_dense)
        + spacing_dense / 2.0
    )
    sub_offset_dense_local = spacing_dense / 2.0

    return {
        "sub_sparse": sub_sparse,
        "sub_dense": sub_dense,
        # Sparse + dense sub-patches share the dense lower-corner origin so a
        # latent fitted on the sparse grid decodes correctly on the dense grid.
        "sub_offset_sparse_local": sub_offset_sparse_local,
        "sub_offset_dense_local": sub_offset_dense_local,
        "sub_crop_lo_vox_dense": sub_lo_vox_dense.tolist(),
        "sub_crop_shape_vox_dense": sub_shape_vox_dense.tolist(),
        "sub_origin_mm_in_disk": sub_origin_mm.tolist(),
        "visible_lcc_voxel_count": int(lcc_count),
        "visible_total_fg_count": int(total_fg),
    }


def coframe_dense_gt_into_obs_window(
    gt_patch_path: Path,
    obs_patch_path: Path,
    step: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Resample the dense GT disk patch into the nnUNet-obs patch's window.

    The nnUNet observation patch (``obs_patch_path``) is a separate
    canonical-align of the SAME original scan as the dense GT patch
    (``gt_patch_path``), centred on nnUNet's (drifted) globe centroid and
    carrying the sparse through-plane spacing (= ``step`` × dense). Both
    affines live in the same canonical (RAS, OS→OD-flipped) world, so we can
    build a DENSE-spacing grid that

      * shares the obs patch's voxel-0 centre and orientation (so the obs
        and the resampled dense target sit in one shared patch-local frame,
        exactly like the GT-degraded items), and
      * covers the obs 80 mm window,

    then world-coordinate resample the dense GT onto it (``order=0``, nearest
    for labels). Crucially we do NOT recentre/register: the dense GT lands at
    its TRUE position within the obs-centred window, i.e. offset from the
    window centre by nnUNet's localisation drift. That drifted dense target
    is what v6 supervises against.

    Returns ``(dense_volume, dense_spacing, step_axis)`` where
    ``dense_volume`` is a float32 tensor on the dense grid and
    ``dense_spacing`` its [3] column-norm spacing.

    Caveat: the OS→OD flip in ``canonical_align`` mirrors array axis 0 with a
    shape-dependent translation. For the standard axial acquisition the
    through-plane axis is NOT axis 0, so obs and GT share an identical flip
    and the affines co-register exactly. A sagittal acquisition whose
    through-plane axis maps to axis 0 would break this assumption; such cases
    are flagged by the near-empty-overlap guard in the bank ingest.
    """
    gt_img = nib.load(str(gt_patch_path))
    obs_img = nib.load(str(obs_patch_path))
    obs_aff = np.asarray(obs_img.affine, dtype=np.float64)
    obs_shape = np.asarray(obs_img.shape[:3], dtype=np.int64)
    obs_cols = obs_aff[:3, :3]
    obs_spacing = np.sqrt((obs_cols ** 2).sum(axis=0))
    step_axis = int(np.argmax(obs_spacing))

    # Dense columns: undo the through-plane ×step stretch (others unchanged).
    dense_cols = obs_cols.copy()
    dense_cols[:, step_axis] = obs_cols[:, step_axis] / float(step)

    target_shape = obs_shape.copy()
    target_shape[step_axis] = max(1, int(round(obs_shape[step_axis] * step)))

    target_aff = np.eye(4, dtype=np.float64)
    target_aff[:3, :3] = dense_cols
    # Voxel-0 centre coincides with the obs patch's voxel-0 centre so the obs
    # grid and the dense grid express the same physical point with the same
    # patch-local mm (matching the GT-degraded subsample convention, where
    # sparse voxel 0 == dense voxel 0).
    target_aff[:3, 3] = obs_aff[:3, 3]

    out = resample_from_to(
        gt_img,
        (tuple(int(x) for x in target_shape), target_aff),
        order=0, mode="constant", cval=0,
    )
    dense_arr = np.asarray(out.dataobj).astype(np.float32)
    dense_spacing = np.sqrt((dense_cols ** 2).sum(axis=0)).astype(np.float32)
    return (
        torch.from_numpy(dense_arr),
        torch.from_numpy(dense_spacing),
        step_axis,
    )


class OrbitalImplicitDataset(data.Dataset):
    """
    Each __getitem__ returns:
        coords:   [N,1,1,3] or [D1,D2,D3,3]  physical coordinates (mm)
        labels:   [N,1,1] or [D1,D2,D3]       integer class labels
        spacings: [3]
        offsets:  [3]
        casenames: str
        caseids:  int     (unique per training item, indexes into latent table)
        scan_ids: int     (shared across offsets of the same scan)
        observed_centroid_mm: [3]
            Center of mass of the sparse-observed foreground, in the same
            patch-local mm frame as `coords`. Drifts up to step*spacing/2 with
            slice_start_id; the true (dense) centroid sits at ≈ patch_size/2
            because canonical_align centers each crop on the whole-eye centroid.
    Plus (INF only): labels_hr, spacings_hr, offsets_hr
    """

    def __init__(self, labels_dir, casenames,
                 num_points_per_dim,
                 slice_step_size=None, slice_step_axis="auto",
                 use_thick_slices=False,
                 num_sparsify_offsets=1,
                 val_fraction=0.15,
                 phase_type=PhaseType.TRAIN,
                 verbose=True,
                 degradation_bank=None,
                 items_per_epoch=None,
                 point_sample_fraction=None,
                 train_supervision="observation"):
        super().__init__()
        if verbose:
            print(f"Loading {len(casenames)} orbital patches "
                  f"(offsets={num_sparsify_offsets}, phase={phase_type.name})...")
        t0 = time.time()

        self.phase_type = phase_type
        self.items_per_epoch = items_per_epoch
        self.point_sample_fraction = point_sample_fraction
        # ``train_supervision`` selects what the latent is fit against during
        # TRAIN/VAL:
        #   "observation" (default): the per-item SPARSE observation (the
        #       original Amiranashvili auto-decoder objective — the latent
        #       can encode degraded/partial shapes).
        #   "dense": the per-item DENSE sub-patch GT (``labels_dense_sub``).
        #       Each item keeps its own latent AND its own sparse-centroid
        #       64 mm frame (so drift handling is unchanged), but the latent
        #       is supervised to produce the TRUE dense shape under that
        #       frame. This tightens the prior so the latent space only
        #       contains plausible shapes — at inference a noisy nnUNet
        #       observation then gets pulled back toward a real shape
        #       instead of being reproduced faithfully.
        # INF always fits the sparse observation regardless of this flag.
        self.train_supervision = str(train_supervision)
        self._epoch_item_indices = None  # set per epoch if items_per_epoch

        # ── Load dense volumes (kept for diagnostics + INF) ───────
        labels_dense, spacings_dense = load_orbital_volumes(labels_dir, casenames)
        offsets_dense = [s / 2.0 for s in spacings_dense]

        # Per-case axes: int -> uniform list; "auto" -> argmax(patch_spacing).
        self.slice_step_axes: List[int] = resolve_slice_step_axes(
            slice_step_axis, spacings_dense
        )

        # The MLP sees a 64 mm physical extent regardless of disk patch size;
        # the inner-crop step below makes every sample physically 64 mm.
        self.image_size = torch.tensor(
            [INNER_PATCH_SIZE_MM] * 3, dtype=torch.float32,
        )

        # ── Strategy dispatch ─────────────────────────────────────
        if degradation_bank is not None:
            # Strategy C: mixed thin/thick degradation bank
            self.slice_step_size = None
            self.slice_step_axis = slice_step_axis
            self.num_sparsify_offsets = None
            self._init_degradation_bank(
                labels_dense, spacings_dense, offsets_dense, casenames,
                self.slice_step_axes, degradation_bank, phase_type,
            )
        else:
            # Legacy strategies A/B
            if slice_step_size is None or slice_step_size < 2:
                raise ValueError("slice_step_size must be >= 2 for legacy mode")
            self.slice_step_size = slice_step_size
            self.slice_step_axis = slice_step_axis
            self.num_sparsify_offsets = num_sparsify_offsets

            use_legacy_split = (
                num_sparsify_offsets == 1 and phase_type != PhaseType.INF
            )
            if use_legacy_split:
                self._init_strategy_a(
                    labels_dense, spacings_dense, offsets_dense, casenames,
                    slice_step_size, self.slice_step_axes, use_thick_slices,
                    val_fraction, phase_type,
                )
            else:
                self._init_multi_offset(
                    labels_dense, spacings_dense, offsets_dense, casenames,
                    slice_step_size, self.slice_step_axes, use_thick_slices,
                    num_sparsify_offsets, phase_type,
                )

        # ── Inner crop: from 80 mm disk patch to 64 mm sub-patch around
        # the visible-LCC centroid. Applies to every item AFTER all sparsi-
        # fication settles. Populates per-item sub-patch tensors + dense
        # sub-patches + sub_crop_in_disk bookkeeping for inference unmap.
        self._apply_inner_crop_to_all_items(
            labels_dense, spacings_dense, offsets_dense,
        )
        # observed centroid is computed on the 64 mm sub-patch (post-crop)
        # so it sits in sub-patch-local mm just like the coords downstream
        # consumers see.
        self._cache_observed_centroids()

        self.num_points = num_points_per_dim ** 3 if num_points_per_dim > 0 else -1
        self.yield_full_res = (phase_type == PhaseType.INF)

        if verbose:
            voxel_shapes = [list(v.shape) for v in self.labels_sparse]
            n_items = len(self)
            n_fallback = sum(
                1 for c in self.visible_lcc_voxel_counts if c == 0
            )
            print(f"  {n_items} items, {len(set(self.scan_ids))} scans "
                  f"in {time.time()-t0:.1f}s")
            print(f"  image_size (mm): {self.image_size.tolist()}")
            if voxel_shapes:
                print(f"  sub-patch voxel shapes range: "
                      f"{[min(s[i] for s in voxel_shapes) for i in range(3)]} to "
                      f"{[max(s[i] for s in voxel_shapes) for i in range(3)]}")
            if n_fallback > 0:
                # Visible-LCC centroid fell back to disk-patch geometric
                # centre because the sparsified view had no foreground. This
                # is rare (only happens when sparsify drops ALL fg slices)
                # but surface it for awareness.
                print(f"  WARN: {n_fallback}/{n_items} items had no visible "
                      f"foreground after sparsification; inner crop fell back "
                      f"to disk-patch geometric centre.")

    # ── Strategy A init (backward compatible) ─────────────────────

    def _init_strategy_a(self, labels_dense, spacings_dense, offsets_dense,
                         casenames, step_size, step_axes, thick_slices,
                         val_fraction, phase_type):
        """Original Amiranashvili logic: 1 offset per scan, val split from train.

        ``step_axes`` is a per-case list (length == len(casenames)); under
        legacy ``slice_step_axis`` int configs all entries are the same.
        """
        n = len(casenames)

        # Initial sparsification (random start per scan)
        gen = torch.Generator().manual_seed(SPRSF_SEED)
        starts = torch.randint(0, step_size, [n], generator=gen).tolist()
        res = [sparsen_volume(v, s, o, ax, step_size, st, thick_slices)
               for v, s, o, ax, st in zip(
                   labels_dense, spacings_dense, offsets_dense,
                   step_axes, starts,
               )]

        self.labels_sparse = [r[0] for r in res]
        self.spacings_sparse = [r[1] for r in res]
        self.offsets_sparse = [r[2] for r in res]

        # Secondary val split
        _, val_ids = self._split_ids(n, val_fraction)
        gen_v = torch.Generator().manual_seed(SPRSF_VAL_SEED)
        starts_v = torch.randint(0, 2, [len(val_ids)], generator=gen_v).tolist()
        if phase_type == PhaseType.VAL:
            starts_v = [(x + 1) % 2 for x in starts_v]
        for cid, sid in zip(val_ids, starts_v):
            self.labels_sparse[cid], self.spacings_sparse[cid], \
                self.offsets_sparse[cid] = sparsen_volume(
                self.labels_sparse[cid], self.spacings_sparse[cid],
                self.offsets_sparse[cid], step_axes[cid], 2, sid, False)

        # Dense references (for INF / diagnostics)
        self.labels_dense = labels_dense
        self.spacings_dense = spacings_dense
        self.offsets_dense = offsets_dense

        if phase_type == PhaseType.VAL:
            self._filter_to_ids(val_ids, casenames, labels_dense,
                                spacings_dense, offsets_dense)
            self.casenames = [casenames[i] for i in val_ids]
            self.caseids = list(val_ids)
            self.scan_ids = list(val_ids)
            self.sparsify_offsets_used = [starts[i] for i in val_ids]
        else:
            self.casenames = list(casenames)
            self.caseids = list(range(n))
            self.scan_ids = list(range(n))
            self.sparsify_offsets_used = starts

        # observed centroid is cached AFTER __init__ runs the inner crop,
        # so labels_sparse holds the 64 mm sub-patch -- the mm frame the
        # downstream model expects.

    # ── Strategy B init (multi-offset) ────────────────────────────

    def _init_multi_offset(self, labels_dense, spacings_dense, offsets_dense,
                           casenames, step_size, step_axes, thick_slices,
                           num_offsets, phase_type):
        """Each scan × each offset → one training item with its own latent.

        ``step_axes`` is a per-case list; under legacy ``slice_step_axis``
        int configs all entries are the same.
        """
        self.labels_sparse = []
        self.spacings_sparse = []
        self.offsets_sparse = []
        self.casenames = []
        self.caseids = []
        self.scan_ids = []
        self.sparsify_offsets_used = []

        # Dense references
        self.labels_dense = labels_dense
        self.spacings_dense = spacings_dense
        self.offsets_dense = offsets_dense

        offsets_to_use = list(range(num_offsets))
        latent_idx = 0

        for scan_idx in range(len(casenames)):
            v, s, o = labels_dense[scan_idx], spacings_dense[scan_idx], offsets_dense[scan_idx]
            step_axis = step_axes[scan_idx]
            for off in offsets_to_use:
                sv, ss, so = sparsen_volume(v, s, o, step_axis, step_size,
                                            off, thick_slices)
                self.labels_sparse.append(sv)
                self.spacings_sparse.append(ss)
                self.offsets_sparse.append(so)
                suffix = f"_off{off}" if num_offsets > 1 else ""
                self.casenames.append(f"{casenames[scan_idx]}{suffix}")
                self.caseids.append(latent_idx)
                self.scan_ids.append(scan_idx)
                self.sparsify_offsets_used.append(off)
                latent_idx += 1

        # observed centroid is cached AFTER __init__ runs the inner crop,
        # so labels_sparse holds the 64 mm sub-patch -- the mm frame the
        # downstream model expects.

    # ── Strategy C init (degradation bank — mixed thin/thick) ─────

    def _init_degradation_bank(
        self, labels_dense, spacings_dense, offsets_dense,
        casenames, step_axes, bank_config, phase_type,
    ):
        """Static degradation bank: each (scan, mode, step, offset) = one item.

        Items are materialized once at init. Thick labels are computed and
        cached to disk (expensive one-hot conv); thin items use cheap
        index_select. The bank is deterministic given the config, so
        resume rebuilds the same item list → latent indices are stable.
        """
        self.labels_sparse = []
        self.spacings_sparse = []
        self.offsets_sparse = []
        self.casenames = []
        self.caseids = []
        self.scan_ids = []
        self.sparsify_offsets_used = []
        self.bank_modes = []  # "dense" | "thin" | "thick" per item
        # ``obs_source`` per item: "gt" (degrade(GT_mask)) or "nnunet"
        # (nnUNet sparse pred on the degraded CT). ``item_dense_override`` is
        # None for gt items (the per-SCAN dense GT is used) and a per-item
        # (volume, spacing, offset) for nnUNet items (the co-framed dense GT
        # in that item's drifted obs window).
        self.bank_obs_source: List[str] = []
        self.item_dense_override: List[
            Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        ] = []

        self.labels_dense = labels_dense
        self.spacings_dense = spacings_dense
        self.offsets_dense = offsets_dense

        modality = bank_config.get("modality", "ct")
        modes = bank_config.get("modes", ["thin", "thick"])
        target_inc = bank_config.get("target_eff_res_increment_mm", 2.0)
        max_eff = bank_config.get("max_eff_resolution_mm", 12.0)
        offsets_per = bank_config.get("offsets_per_setting", 1)
        num_classes = bank_config.get("num_classes", 5)
        cache_dir = bank_config.get("cache_dir")

        # ── nnUNet-obs mixing (v6) ────────────────────────────────
        # obs_sources controls which observation types populate the bank.
        # "gt": degrade the GT mask (default, original Strategy C). "nnunet":
        # additionally ingest nnUNet's sparse prediction on the degraded CT,
        # co-framed against the dense GT (drift preserved). nnUNet items are
        # only added where the on-disk obs patch exists, so atlas cases (no
        # train obs patches) silently fall back to gt-only.
        obs_sources = bank_config.get("obs_sources", ["gt"])
        use_nnunet_obs = "nnunet" in obs_sources
        # When "gt" is absent from obs_sources (e.g. v6-5: obs_sources=[nnunet]),
        # GT is NOT used as an input observation at all -- no degraded-GT items
        # and no dense (step==1) GT item are added to the bank. GT still serves
        # as the dense LOSS target for nnUNet items (via the co-frame path
        # below). Existing configs all contain "gt", so this is a no-op for them.
        use_gt_obs = "gt" in obs_sources
        nnunet_prefix_tmpl = bank_config.get(
            "nnunet_patch_prefix", "labels_dataset835_{exp}_train_step_"
        )
        aligned_dir = bank_config.get("_aligned_dir")
        labels_dir = bank_config.get("_labels_dir")
        nnunet_modes = [m for m in modes if m in ("thin", "thick")]
        n_nnunet_added = 0
        n_nnunet_missing = 0
        n_nnunet_empty = 0
        nnunet_issues: List[str] = []
        if use_nnunet_obs and (aligned_dir is None or labels_dir is None):
            raise ValueError(
                "degradation_bank.obs_sources includes 'nnunet' but "
                "_aligned_dir/_labels_dir were not injected; "
                "create_data_loader must pass them."
            )

        # Auto cache dir: default to {labels_dir}/../degraded_bank/
        if cache_dir is None and "thick" in modes:
            cache_dir = str(
                Path(bank_config.get("_labels_dir", ".")).parent / "degraded_bank"
            )
        if cache_dir is not None:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)

        latent_idx = 0

        for scan_idx in range(len(casenames)):
            v = labels_dense[scan_idx]
            s = spacings_dense[scan_idx]
            o = offsets_dense[scan_idx]
            ax = step_axes[scan_idx]
            spacing_ax = float(s[ax])

            steps = adaptive_steps_for_bank(spacing_ax, target_inc, max_eff)

            for step in steps:
                if step == 1:
                    # Dense item (no degradation) -- a GT-input item, so only
                    # when "gt" is in obs_sources.
                    if use_gt_obs:
                        self.labels_sparse.append(v)
                        self.spacings_sparse.append(s.clone())
                        self.offsets_sparse.append(o.clone())
                        self.casenames.append(f"{casenames[scan_idx]}_dense")
                        self.caseids.append(latent_idx)
                        self.scan_ids.append(scan_idx)
                        self.sparsify_offsets_used.append(0)
                        self.bank_modes.append("dense")
                        self.bank_obs_source.append("gt")
                        self.item_dense_override.append(None)
                        latent_idx += 1
                    continue

                if use_gt_obs:
                    for mode in modes:
                        for off in range(offsets_per):
                            if mode == "thin":
                                sv, ss, so = degrade_thin(
                                    v, s, o, ax, step, start=off
                                )
                            elif mode == "thick":
                                sv, ss, so = self._get_thick_cached(
                                    v, s, o, ax, step, off,
                                    modality, num_classes, cache_dir,
                                    casenames[scan_idx],
                                )
                            else:
                                raise ValueError(f"Unknown mode: {mode}")

                            suffix = f"_{mode}_s{step}"
                            if offsets_per > 1:
                                suffix += f"_o{off}"
                            self.labels_sparse.append(sv)
                            self.spacings_sparse.append(ss)
                            self.offsets_sparse.append(so)
                            self.casenames.append(
                                f"{casenames[scan_idx]}{suffix}"
                            )
                            self.caseids.append(latent_idx)
                            self.scan_ids.append(scan_idx)
                            self.sparsify_offsets_used.append(off)
                            self.bank_modes.append(mode)
                            self.bank_obs_source.append("gt")
                            self.item_dense_override.append(None)
                            latent_idx += 1

                # ── nnUNet-obs items for this (scan, step) ─────────
                # One item per (mode in {thin,thick}) where the on-disk obs
                # patch exists. The observation is nnUNet's sparse pred on the
                # degraded CT (loaded with the inference offset convention);
                # the supervision target is the dense GT co-framed into that
                # obs window (drift preserved).
                if not use_nnunet_obs:
                    continue
                gt_patch_path = Path(labels_dir) / f"{casenames[scan_idx]}.nii.gz"
                for mode in nnunet_modes:
                    obs_dir = Path(aligned_dir) / (
                        nnunet_prefix_tmpl.format(exp=mode) + f"{step:02d}"
                    )
                    obs_patch_path = obs_dir / f"{casenames[scan_idx]}.nii.gz"
                    if not obs_patch_path.exists():
                        n_nnunet_missing += 1
                        continue
                    if not gt_patch_path.exists():
                        nnunet_issues.append(
                            f"{casenames[scan_idx]} {mode} s{step}: GT patch "
                            f"missing for co-framing ({gt_patch_path})"
                        )
                        continue
                    try:
                        obs_img = nib.load(str(obs_patch_path))
                        obs_arr = np.asarray(obs_img.dataobj, dtype=np.float32)
                        obs_aff = np.asarray(obs_img.affine, dtype=np.float64)
                        obs_spacing = np.sqrt(
                            (obs_aff[:3, :3] ** 2).sum(axis=0)
                        ).astype(np.float32)
                        labels_obs = torch.from_numpy(obs_arr)
                        spacing_obs = torch.from_numpy(obs_spacing)
                        # Inference offset convention: offset = spacing/2, then
                        # through-plane offset /= step so sparse voxel 0 centre
                        # coincides with dense voxel 0 centre (drift fix).
                        offset_obs = spacing_obs / 2.0
                        obs_axis = int(torch.argmax(spacing_obs))
                        offset_obs[obs_axis] = spacing_obs[obs_axis] / (2.0 * step)

                        dense_v, dense_s, _ = coframe_dense_gt_into_obs_window(
                            gt_patch_path, obs_patch_path, step,
                        )
                    except Exception as e:  # noqa: BLE001
                        nnunet_issues.append(
                            f"{casenames[scan_idx]} {mode} s{step}: obs/coframe "
                            f"failed ({type(e).__name__}: {e})"
                        )
                        continue

                    if int((dense_v > 0).sum()) == 0:
                        # Co-framed GT carries no foreground in the obs window:
                        # a gross frame mismatch (e.g. sagittal OS-flip edge
                        # case) or nnUNet dropped the globe far from GT. Skip.
                        n_nnunet_empty += 1
                        nnunet_issues.append(
                            f"{casenames[scan_idx]} {mode} s{step}: co-framed "
                            f"GT empty in obs window -- skipped"
                        )
                        continue

                    dense_off = dense_s / 2.0
                    self.labels_sparse.append(labels_obs)
                    self.spacings_sparse.append(spacing_obs)
                    self.offsets_sparse.append(offset_obs)
                    self.casenames.append(
                        f"{casenames[scan_idx]}_{mode}_s{step}_nnunet"
                    )
                    self.caseids.append(latent_idx)
                    self.scan_ids.append(scan_idx)
                    self.sparsify_offsets_used.append(0)
                    self.bank_modes.append(mode)
                    self.bank_obs_source.append("nnunet")
                    self.item_dense_override.append(
                        (dense_v, dense_s, dense_off)
                    )
                    latent_idx += 1
                    n_nnunet_added += 1

        if use_nnunet_obs:
            print(f"  [bank] nnUNet-obs items added: {n_nnunet_added} "
                  f"(missing obs patch: {n_nnunet_missing}, "
                  f"empty co-frame: {n_nnunet_empty})")
            if nnunet_issues:
                for line in nnunet_issues[:15]:
                    print(f"    - {line}")
                if len(nnunet_issues) > 15:
                    print(f"    ... and {len(nnunet_issues) - 15} more")

    @staticmethod
    def _get_thick_cached(
        volume, spacing, offset, axis, step, start,
        modality, num_classes, cache_dir, casename,
    ):
        """Load thick-degraded label from cache, or compute + save."""
        kernel = get_kernel(modality, step)
        fname = f"{casename}_thick_s{step}_o{start}.pt"

        if cache_dir is not None:
            cache_path = cache_dir / fname
            if cache_path.exists():
                cached = torch.load(cache_path, weights_only=True)
                from simulation.affine_ops import compute_sparse_affine
                ss, so = compute_sparse_affine(spacing, offset, axis, step, start)
                return cached, ss, so

        sv, ss, so = degrade_thick(
            volume, spacing, offset, axis, step, start=start,
            kernel=kernel, is_label=True, num_classes=num_classes,
        )

        if cache_dir is not None:
            cache_path = cache_dir / fname
            torch.save(sv, cache_path)

        return sv, ss, so

    # ── Helpers ───────────────────────────────────────────────────

    def _cache_observed_centroids(self):
        """
        Precompute per-item centroid of the OBSERVED (sparsified) foreground
        in patch-local physical (mm) coordinates.

        After the 64 mm inner crop step this centroid is in sub-patch-local
        mm and should sit near ``INNER_PATCH_SIZE_MM / 2`` (= 32 mm) by
        construction. Sub-patch is centred on the visible-LCC centroid, so
        the OBSERVED centroid landing far from 32 mm is itself a QC signal
        (e.g. the sub-patch was edge-clamped against the 80 mm disk patch
        boundary, so the LCC ended up off-centre after zero-padding).
        """
        self.observed_centroids_mm = [
            compute_centroid_mm(
                self.labels_sparse[i],
                self.spacings_sparse[i],
                self.offsets_sparse[i],
            )
            for i in range(len(self.labels_sparse))
        ]

    # ── Inner crop (80 mm disk patch → 64 mm sub-patch) ─────────
    def _apply_inner_crop_to_all_items(self, labels_dense, spacings_dense,
                                       offsets_dense):
        """Replace each item's 80 mm sparsified disk patch with the 64 mm
        inner crop around its visible-LCC centroid; populate matching
        per-item dense sub-patches and the sub_crop bookkeeping inference
        needs to unmap predictions back to full-volume space.

        Called once at the end of __init__ (after every sparsification —
        including Strategy A's secondary val split — has settled), so
        per-item self.labels_sparse / spacings_sparse / offsets_sparse are
        already final.

        Inputs ``labels_dense / spacings_dense / offsets_dense`` are per-
        SCAN; this method fans them out per-item using ``self.scan_ids``.
        """
        self.labels_dense_sub: List[torch.Tensor] = []
        self.spacings_dense_sub: List[torch.Tensor] = []
        self.offsets_dense_sub: List[torch.Tensor] = []
        # ``sub_crop_lo_vox_dense[i]`` / ``sub_crop_shape_vox_dense[i]`` are
        # voxel coordinates of item ``i``'s sub-patch INSIDE its source
        # 80 mm disk patch (dense voxel grid). Native unmap composes this
        # with ``crop_slices`` from the disk patch's metadata json to get
        # full-volume coordinates.
        self.sub_crop_lo_vox_dense: List[list] = []
        self.sub_crop_shape_vox_dense: List[list] = []
        self.sub_origin_mm_in_disk: List[list] = []
        self.visible_lcc_voxel_counts: List[int] = []
        self.visible_total_fg_counts: List[int] = []

        # Per-item dense override (nnUNet obs items): the dense target is the
        # GT co-framed into that item's drifted obs window, NOT the per-scan
        # GT disk patch. None for gt items / legacy strategies.
        dense_overrides = getattr(self, "item_dense_override", None)
        for item_idx in range(len(self.labels_sparse)):
            scan_id = self.scan_ids[item_idx]
            override = dense_overrides[item_idx] if dense_overrides else None
            if override is not None:
                vol_d, sp_d, of_d = override
            else:
                vol_d = labels_dense[scan_id]
                sp_d = spacings_dense[scan_id]
                of_d = offsets_dense[scan_id]
            info = inner_crop_64mm(
                volume_sparse=self.labels_sparse[item_idx],
                spacing_sparse=self.spacings_sparse[item_idx],
                offset_sparse=self.offsets_sparse[item_idx],
                volume_dense=vol_d,
                spacing_dense=sp_d,
                offset_dense=of_d,
            )
            self.labels_sparse[item_idx] = info["sub_sparse"]
            self.offsets_sparse[item_idx] = info["sub_offset_sparse_local"]
            self.labels_dense_sub.append(info["sub_dense"])
            self.spacings_dense_sub.append(sp_d)
            self.offsets_dense_sub.append(info["sub_offset_dense_local"])
            self.sub_crop_lo_vox_dense.append(info["sub_crop_lo_vox_dense"])
            self.sub_crop_shape_vox_dense.append(info["sub_crop_shape_vox_dense"])
            self.sub_origin_mm_in_disk.append(info["sub_origin_mm_in_disk"])
            self.visible_lcc_voxel_counts.append(info["visible_lcc_voxel_count"])
            self.visible_total_fg_counts.append(info["visible_total_fg_count"])

    @staticmethod
    def _split_ids(n, val_fraction):
        n_val = max(1, round(n * val_fraction))
        gen = torch.Generator().manual_seed(SPLIT_SEED)
        perm = torch.randperm(n, generator=gen).tolist()
        return perm[:-n_val], perm[-n_val:]

    def _filter_to_ids(self, ids, casenames, labels_dense,
                       spacings_dense, offsets_dense):
        # Per-item filter — applies to every list whose i-th entry is item i.
        # (Inner-crop bookkeeping is populated AFTER this call, so it isn't
        # filtered here; only the source-scan lists are.)
        for attr in ["labels_sparse", "spacings_sparse", "offsets_sparse"]:
            setattr(self, attr, [getattr(self, attr)[i] for i in ids])
        # ``labels_dense`` is the source per-scan list; in Strategy A item ==
        # scan, so the same index set applies. After this call the disk-patch
        # references stored on self are also restricted to the val scans,
        # which is fine because Strategy A val never indexes by raw scan_id
        # outside the val pool.
        self.labels_dense = [labels_dense[i] for i in ids]
        self.spacings_dense = [spacings_dense[i] for i in ids]
        self.offsets_dense = [offsets_dense[i] for i in ids]

    # ── Backward-compatible aliases (used by infer.py) ────────────

    @property
    def spacings(self):
        return self.spacings_sparse

    @property
    def labels(self):
        return self.labels_sparse

    @property
    def offsets(self):
        return self.offsets_sparse

    def __len__(self):
        return len(self.labels_sparse)

    def set_epoch_subset(self, epoch: int):
        """Legacy hook (no-op). Epoch subsetting is now handled by
        EpochSubsetSampler in create_data_loader."""
        pass

    def __getitem__(self, item):
        # Supervision-target selection. INF always fits the sparse
        # observation (the deployment input); TRAIN/VAL can instead fit the
        # dense sub-patch GT when train_supervision == "dense" (tight-prior
        # objective). The dense sub-patch shares the same 64 mm sub-patch
        # origin as the sparse view (see inner_crop_64mm), so the latent
        # frame is identical either way — only the labels/coords sampled
        # differ.
        use_dense_target = (
            self.train_supervision == "dense"
            and self.phase_type in (PhaseType.TRAIN, PhaseType.VAL)
        )
        if use_dense_target:
            label = self.labels_dense_sub[item]
            sample_spacing = self.spacings_dense_sub[item]
            sample_offset = self.offsets_dense_sub[item]
        else:
            label = self.labels_sparse[item]
            sample_spacing = self.spacings_sparse[item]
            sample_offset = self.offsets_sparse[item]

        if self.num_points > 0:
            # Point count is FIXED across items to allow batch collation.
            # point_sample_fraction scales the base num_points (from config's
            # num_points_per_dim^3), NOT the per-item voxel count — this
            # ensures all items in a batch produce the same tensor shape.
            if self.point_sample_fraction is not None:
                num_pts = max(1, int(self.num_points * self.point_sample_fraction))
            else:
                num_pts = self.num_points

            voxel_ids = torch.empty(num_pts, 3, dtype=torch.int64)
            for d in range(3):
                voxel_ids[:, d] = torch.randint(0, label.shape[d], [num_pts])
            label_values = label[voxel_ids[:, 0], voxel_ids[:, 1], voxel_ids[:, 2]]
            voxel_ids = voxel_ids.unsqueeze(1).unsqueeze(1)
            label_values = label_values.unsqueeze(1).unsqueeze(1)
        else:
            individual = [torch.arange(label.shape[d]) for d in range(3)]
            meshed = torch.meshgrid(individual, indexing="ij")
            voxel_ids = torch.stack(meshed, dim=-1)
            label_values = label

        spacing = sample_spacing
        offset = sample_offset
        coords = voxel_ids.float() * spacing + offset

        result = {
            "coords": coords, "labels": label_values,
            "spacings": spacing, "offsets": offset,
            "casenames": self.casenames[item],
            "caseids": self.caseids[item],
            "scan_ids": self.scan_ids[item],
            # Centroid of observed sparse foreground in the sub-patch-local
            # mm frame (post inner crop). By construction the sub-patch is
            # centred on the visible-LCC centroid, so this should land near
            # ``image_size / 2 = 32 mm`` along every axis. Significant drift
            # away from 32 mm indicates edge-clamping at the disk-patch
            # boundary.
            "observed_centroid_mm": self.observed_centroids_mm[item],
            # Sub-patch position inside the 80 mm disk patch (dense voxel
            # frame). Inference uses these to compose
            #   pred (64 mm sub-patch) → disk-patch (80 mm) → full volume.
            # Stored as plain lists so the default DataLoader collate
            # works without a custom collate_fn.
            "sub_crop_lo_vox_dense": self.sub_crop_lo_vox_dense[item],
            "sub_crop_shape_vox_dense": self.sub_crop_shape_vox_dense[item],
        }
        if self.yield_full_res:
            # Per-item dense sub-patch: same 64 mm physical region as the
            # sparsified input, sampled at the original dense voxel grid.
            # This is what inference compares predictions against.
            result["labels_hr"] = self.labels_dense_sub[item]
            result["spacings_hr"] = self.spacings_dense_sub[item]
            result["offsets_hr"] = self.offsets_dense_sub[item]
        return result


# ── Epoch subset sampler (Strategy C) ─────────────────────────────

class EpochSubsetSampler(data.Sampler):
    """Yields a random subset of indices each epoch (for degradation bank).

    Call `set_epoch(epoch)` before each epoch to regenerate the subset.
    The full dataset `__len__` stays constant (for latent table sizing).
    """

    def __init__(self, dataset_size: int, items_per_epoch: int):
        self.dataset_size = dataset_size
        self.items_per_epoch = min(items_per_epoch, dataset_size)
        self._indices: List[int] = list(range(self.items_per_epoch))

    def set_epoch(self, epoch: int):
        gen = torch.Generator().manual_seed(BANK_SEED + epoch)
        perm = torch.randperm(self.dataset_size, generator=gen)
        self._indices = perm[:self.items_per_epoch].tolist()

    def __iter__(self):
        # Shuffle within the epoch subset for batch diversity
        order = torch.randperm(len(self._indices)).tolist()
        return iter([self._indices[i] for i in order])

    def __len__(self):
        return self.items_per_epoch


# ── DataLoader factory ────────────────────────────────────────────

def _scan_disjoint_split(casenames, val_fraction, seed):
    """Scan-disjoint train/valid split of a casename pool.

    Groups casenames by ``source_id`` (= casename without the _OD/_OS
    suffix), shuffles the scans with ``seed``, and holds out
    ``round(val_fraction * n_scans)`` scans for validation. Every eye/variant
    of a held-out scan goes to valid -> no shape leakage across the split.
    Returns ``(train_casenames, val_casenames)`` (each sorted).
    """
    by_scan: Dict[str, List[str]] = {}
    for cn in casenames:
        sid = cn[:-3] if (cn.endswith("_OD") or cn.endswith("_OS")) else cn
        by_scan.setdefault(sid, []).append(cn)
    scans = sorted(by_scan)
    gen = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(len(scans), generator=gen).tolist()
    n_val = max(1, round(len(scans) * float(val_fraction))) if scans else 0
    val_scan_set = {scans[i] for i in perm[:n_val]}
    train_names, val_names = [], []
    for sid in scans:
        (val_names if sid in val_scan_set else train_names).extend(by_scan[sid])
    return sorted(train_names), sorted(val_names)


def create_data_loader(params, phase_type, verbose=True):
    labels_dir = Path(params["aligned_dir"]) / params.get("labels_dirname", "labels")
    casefiles_dir = Path(params["casefiles_dir"])
    is_training = (phase_type == PhaseType.TRAIN)

    # ── Determine casefile ────────────────────────────────────────
    val_casefile = params.get("val_casefile")  # None = Strategy A
    num_offsets = params.get("num_sparsify_offsets", 1)

    bank_cfg_raw = params.get("degradation_bank")
    # Strategy C v6: when the bank carries a ``split`` block, TRAIN and VAL
    # draw from ONE combined modeling pool (train_cases ∪ val_cases) split
    # scan-disjoint by ``split_seed`` -- replacing the fixed train/val casefile
    # boundary so the constructed 4-type item pool is what gets shuffled+split.
    bank_split = (bank_cfg_raw or {}).get("split") if bank_cfg_raw else None
    use_pool_split = (
        bank_split is not None and phase_type in (PhaseType.TRAIN, PhaseType.VAL)
    )

    if phase_type == PhaseType.INF:
        casenames = load_casenames(casefiles_dir / params["test_casefile"])
        num_offsets_ds = 1  # inference: single view
    elif use_pool_split:
        # Combined modeling pool -> deterministic scan-disjoint split.
        pool = load_casenames(casefiles_dir / params["train_casefile"])
        if val_casefile:
            pool = sorted(set(pool) | set(
                load_casenames(casefiles_dir / val_casefile)
            ))
        granularity = bank_split.get("split_granularity", "scan")
        if granularity != "scan":
            print(f"  [bank] split_granularity={granularity!r} not supported "
                  f"with the per-dataset latent table; falling back to "
                  f"scan-disjoint split.")
        train_names, val_names = _scan_disjoint_split(
            pool,
            bank_split.get("val_fraction", params.get("val_fraction", 0.15)),
            bank_split.get("split_seed", SPLIT_SEED),
        )
        casenames = train_names if is_training else val_names
        num_offsets_ds = 1
        if verbose:
            print(f"  [bank] pool split (seed={bank_split.get('split_seed', SPLIT_SEED)}): "
                  f"{len(train_names)} train / {len(val_names)} valid casenames "
                  f"(this phase: {len(casenames)})")
    elif phase_type == PhaseType.VAL and val_casefile:
        # Strategy B: separate val scans
        casenames = load_casenames(casefiles_dir / val_casefile)
        num_offsets_ds = 1  # val: single view (like inference)
    else:
        # TRAIN (both strategies), or VAL with Strategy A (split from train)
        casenames = load_casenames(casefiles_dir / params["train_casefile"])
        num_offsets_ds = num_offsets if is_training else 1

    # ── Create dataset ────────────────────────────────────────────
    bank_cfg = params.get("degradation_bank")
    if bank_cfg is not None:
        # Inject labels_dir path for auto cache_dir resolution + the
        # aligned_dir so the bank can locate nnUNet-obs patches.
        bank_cfg = dict(bank_cfg)  # copy to avoid mutating params
        bank_cfg.setdefault("_labels_dir", str(labels_dir))
        bank_cfg.setdefault("_aligned_dir", str(Path(params["aligned_dir"])))

    # For INF/VAL under Strategy C, the dataset still needs a
    # slice_step_size for the legacy _init_multi_offset path (which
    # creates a single-step sparsified view for latent initialization).
    # The actual multi-resolution sweep is handled by resolution_sweep.py
    # which sparsifies on-the-fly, so this value only matters as a
    # baseline starting step for the single-item INF dataset.
    step_size = params.get("slice_step_size")
    if step_size is None and bank_cfg is not None and not is_training:
        step_size = 2  # minimum valid step for INF dataset baseline

    ds = OrbitalImplicitDataset(
        labels_dir=labels_dir,
        casenames=casenames,
        num_points_per_dim=(
            params.get("num_points_per_example_per_dim_train", 64) if is_training else -1
        ),
        slice_step_size=step_size,
        slice_step_axis=params.get("slice_step_axis", "auto"),
        use_thick_slices=params.get("use_thick_slices", False),
        num_sparsify_offsets=num_offsets_ds,
        val_fraction=params.get("val_fraction", 0.15),
        phase_type=phase_type,
        verbose=verbose,
        degradation_bank=bank_cfg if is_training else None,
        items_per_epoch=params.get("items_per_epoch") if is_training else None,
        point_sample_fraction=params.get("point_sample_fraction") if is_training else None,
        train_supervision=params.get("train_supervision", "observation"),
    )

    # ── Sampler / shuffle ─────────────────────────────────────────
    sampler = None
    shuffle = is_training
    items_per_epoch = params.get("items_per_epoch")
    if is_training and bank_cfg is not None and items_per_epoch is not None:
        sampler = EpochSubsetSampler(len(ds), items_per_epoch)
        shuffle = False  # sampler handles ordering

    batch_size = params["batch_size_train"] if is_training else params.get("batch_size_val", 1)
    return data.DataLoader(
        ds, batch_size, shuffle=shuffle,
        sampler=sampler,
        num_workers=params.get("num_workers", 2),
        drop_last=is_training,
    )