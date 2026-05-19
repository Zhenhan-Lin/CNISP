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
    - image_size = max(shape * spacing) across all training cases
"""

import time
from enum import IntEnum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
from torch.utils import data

from data_prep.sparsify import sparsen_volume

SPRSF_SEED = 1
SPLIT_SEED = 2
SPRSF_VAL_SEED = 3


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
        spacing = np.abs(np.diagonal(aff)[:3]).astype(np.float32)
        volumes.append(torch.from_numpy(vol))
        spacings.append(torch.from_numpy(spacing))
    return volumes, spacings


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
    Plus (INF only): labels_hr, spacings_hr, offsets_hr
    """

    def __init__(self, labels_dir, casenames,
                 num_points_per_dim,
                 slice_step_size, slice_step_axis, use_thick_slices,
                 num_sparsify_offsets=1,
                 val_fraction=0.15,
                 phase_type=PhaseType.TRAIN,
                 verbose=True):
        super().__init__()
        if verbose:
            print(f"Loading {len(casenames)} orbital patches "
                  f"(offsets={num_sparsify_offsets}, phase={phase_type.name})...")
        t0 = time.time()

        if slice_step_size < 2:
            raise ValueError("slice_step_size must be >= 2")

        self.slice_step_size = slice_step_size
        self.slice_step_axis = slice_step_axis
        self.num_sparsify_offsets = num_sparsify_offsets
        self.phase_type = phase_type

        # ── Load dense volumes (kept for diagnostics + INF) ───────
        labels_dense, spacings_dense = load_orbital_volumes(labels_dir, casenames)
        offsets_dense = [s / 2.0 for s in spacings_dense]

        # image_size = max physical extent across all patches
        image_sizes = [
            torch.tensor(v.shape, dtype=torch.float32) * s
            for v, s in zip(labels_dense, spacings_dense)
        ]
        self.image_size = torch.stack(image_sizes).max(dim=0)[0]

        # ── Strategy A: Amiranashvili val split ───────────────────
        # Applied only when num_sparsify_offsets == 1 (Strategy A) and
        # phase is TRAIN or VAL.  Strategy B skips this entirely.
        use_legacy_split = (num_sparsify_offsets == 1 and phase_type != PhaseType.INF)

        if use_legacy_split:
            self._init_strategy_a(
                labels_dense, spacings_dense, offsets_dense, casenames,
                slice_step_size, slice_step_axis, use_thick_slices,
                val_fraction, phase_type,
            )
        else:
            self._init_multi_offset(
                labels_dense, spacings_dense, offsets_dense, casenames,
                slice_step_size, slice_step_axis, use_thick_slices,
                num_sparsify_offsets, phase_type,
            )

        self.num_points = num_points_per_dim ** 3 if num_points_per_dim > 0 else -1
        self.yield_full_res = (phase_type == PhaseType.INF)

        if verbose:
            voxel_shapes = [list(v.shape) for v in self.labels_sparse]
            print(f"  {len(self)} items, {len(set(self.scan_ids))} scans "
                  f"in {time.time()-t0:.1f}s")
            print(f"  image_size (mm): {self.image_size.tolist()}")
            if voxel_shapes:
                print(f"  sparse voxel shapes range: "
                      f"{[min(s[i] for s in voxel_shapes) for i in range(3)]} to "
                      f"{[max(s[i] for s in voxel_shapes) for i in range(3)]}")

    # ── Strategy A init (backward compatible) ─────────────────────

    def _init_strategy_a(self, labels_dense, spacings_dense, offsets_dense,
                         casenames, step_size, step_axis, thick_slices,
                         val_fraction, phase_type):
        """Original Amiranashvili logic: 1 offset per scan, val split from train."""
        n = len(casenames)

        # Initial sparsification (random start per scan)
        gen = torch.Generator().manual_seed(SPRSF_SEED)
        starts = torch.randint(0, step_size, [n], generator=gen).tolist()
        res = [sparsen_volume(v, s, o, step_axis, step_size, st, thick_slices)
               for v, s, o, st in zip(labels_dense, spacings_dense, offsets_dense, starts)]

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
                self.offsets_sparse[cid], step_axis, 2, sid, False)

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

    # ── Strategy B init (multi-offset) ────────────────────────────

    def _init_multi_offset(self, labels_dense, spacings_dense, offsets_dense,
                           casenames, step_size, step_axis, thick_slices,
                           num_offsets, phase_type):
        """Each scan × each offset → one training item with its own latent."""
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

    # ── Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _split_ids(n, val_fraction):
        n_val = max(1, round(n * val_fraction))
        gen = torch.Generator().manual_seed(SPLIT_SEED)
        perm = torch.randperm(n, generator=gen).tolist()
        return perm[:-n_val], perm[-n_val:]

    def _filter_to_ids(self, ids, casenames, labels_dense,
                       spacings_dense, offsets_dense):
        for attr in ["labels_sparse", "spacings_sparse", "offsets_sparse"]:
            setattr(self, attr, [getattr(self, attr)[i] for i in ids])
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

    def __getitem__(self, item):
        label = self.labels_sparse[item]

        if self.num_points > 0:
            voxel_ids = torch.empty(self.num_points, 3, dtype=torch.int64)
            for d in range(3):
                voxel_ids[:, d] = torch.randint(0, label.shape[d], [self.num_points])
            label_values = label[voxel_ids[:, 0], voxel_ids[:, 1], voxel_ids[:, 2]]
            voxel_ids = voxel_ids.unsqueeze(1).unsqueeze(1)
            label_values = label_values.unsqueeze(1).unsqueeze(1)
        else:
            individual = [torch.arange(label.shape[d]) for d in range(3)]
            meshed = torch.meshgrid(individual, indexing="ij")
            voxel_ids = torch.stack(meshed, dim=-1)
            label_values = label

        spacing = self.spacings_sparse[item]
        offset = self.offsets_sparse[item]
        coords = voxel_ids.float() * spacing + offset

        result = {
            "coords": coords, "labels": label_values,
            "spacings": spacing, "offsets": offset,
            "casenames": self.casenames[item],
            "caseids": self.caseids[item],
            "scan_ids": self.scan_ids[item],
        }
        if self.yield_full_res:
            scan_id = self.scan_ids[item]
            result["labels_hr"] = self.labels_dense[scan_id]
            result["spacings_hr"] = self.spacings_dense[scan_id]
            result["offsets_hr"] = self.offsets_dense[scan_id]
        return result


# ── DataLoader factory ────────────────────────────────────────────

def create_data_loader(params, phase_type, verbose=True):
    labels_dir = Path(params["aligned_dir"]) / params.get("labels_dirname", "labels")
    casefiles_dir = Path(params["casefiles_dir"])
    is_training = (phase_type == PhaseType.TRAIN)

    # ── Determine casefile ────────────────────────────────────────
    val_casefile = params.get("val_casefile")  # None = Strategy A
    num_offsets = params.get("num_sparsify_offsets", 1)

    if phase_type == PhaseType.INF:
        casenames = load_casenames(casefiles_dir / params["test_casefile"])
        num_offsets_ds = 1  # inference: single view
    elif phase_type == PhaseType.VAL and val_casefile:
        # Strategy B: separate val scans
        casenames = load_casenames(casefiles_dir / val_casefile)
        num_offsets_ds = 1  # val: single view (like inference)
    else:
        # TRAIN (both strategies), or VAL with Strategy A (split from train)
        casenames = load_casenames(casefiles_dir / params["train_casefile"])
        num_offsets_ds = num_offsets if is_training else 1

    # ── Create dataset ────────────────────────────────────────────
    ds = OrbitalImplicitDataset(
        labels_dir=labels_dir,
        casenames=casenames,
        num_points_per_dim=(
            params.get("num_points_per_example_per_dim_train", 64) if is_training else -1
        ),
        slice_step_size=params["slice_step_size"],
        slice_step_axis=params["slice_step_axis"],
        use_thick_slices=params.get("use_thick_slices", False),
        num_sparsify_offsets=num_offsets_ds,
        val_fraction=params.get("val_fraction", 0.15),
        phase_type=phase_type,
        verbose=verbose,
    )

    batch_size = params["batch_size_train"] if is_training else params.get("batch_size_val", 1)
    return data.DataLoader(
        ds, batch_size, shuffle=is_training,
        num_workers=params.get("num_workers", 2),
        drop_last=is_training,
    )