"""Prior-channel augmentations for the nnUNet-C corrector (design §1.2.2–1.2.3).

These are the CUSTOM prior-channel transforms that supplement nnUNet's stock
cascade morphological aug (§1.2.1 — `ApplyRandomBinaryOperatorTransform` +
`RemoveRandomConnectedComponentFromOneHotEncodingTransform`, which the trainer
already emits via the `is_cascaded` branch). They run on the ONE-HOT prior
channels **after** `MoveSegAsOneHotToDataTransform` has appended them to `image`.

Layout at this point (batchgeneratorsv2, sample-level, CPU torch):
    data_dict['image'] : torch.Tensor (C, *spatial), C = 1 CT + 4 one-hot prior
                         -> prior channels are indices (1, 2, 3, 4)
    data_dict['segmentation'] : (1, *spatial) = GT target (untouched here)

Both transforms subclass `batchgeneratorsv2.transforms.base.basic_transform.
BasicTransform`, sample their randomness in `get_parameters`, and modify ONLY the
prior channels of `image` in `_apply_to_image` (ch0 CT and the GT are untouched).
Training-time only — the trainer inserts them before deep-supervision downsampling
and NOT into the validation pipeline (clean priors at val).

Deps: torch, numpy, scipy.ndimage, batchgeneratorsv2. Installed into nnunetv2
site-packages alongside the trainer by run_train.sh / run_corrector_predict.sh.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from scipy.ndimage import shift as _ndi_shift

from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform


class PriorCentroidJitterTransform(BasicTransform):
    """Rigid, degradation-decoupled translation of the prior channels.

    Replaces the accidental step-correlated centroid drift with a controlled,
    step-INDEPENDENT jitter so the corrector cannot treat ch1–4 as pixel-accurate
    truth. The SAME integer voxel shift δ is applied to every prior channel
    (preserving inter-structure spatial relationships) with zero fill (order 0, so
    the one-hot channels stay binary). ch0 and the GT are never shifted.

    ``max_shift_voxels`` is per spatial axis, in the patch's (plan-transposed) axis
    order — calibrate the largest component to the longitudinal drift magnitude
    measured from the original bug (design §1.2.2). e.g. (4, 2, 2).
    """

    def __init__(self, prior_channel_indices: Sequence[int] = (1, 2, 3, 4),
                 max_shift_voxels: Sequence[int] = (4, 2, 2)):
        super().__init__()
        self.prior_channel_indices = tuple(int(c) for c in prior_channel_indices)
        self.max_shift_voxels = tuple(int(m) for m in max_shift_voxels)

    def get_parameters(self, **data_dict) -> dict:
        img = data_dict["image"]
        nsp = img.ndim - 1                       # spatial dims
        maxes = list(self.max_shift_voxels)[:nsp]
        maxes += [0] * (nsp - len(maxes))        # pad if fewer maxes than axes
        shift = [int(np.random.randint(-m, m + 1)) if m > 0 else 0 for m in maxes]
        return {"shift": tuple(shift)}

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        shift = params["shift"]
        if not any(shift):
            return img
        for c in self.prior_channel_indices:
            if c >= img.shape[0]:
                continue
            arr = img[c].cpu().numpy()
            moved = _ndi_shift(arr, shift, order=0, mode="constant", cval=0)
            img[c] = torch.from_numpy(moved).to(img.dtype)
        return img

    # priors are in `image`; leave GT / regression untouched.
    def _apply_to_segmentation(self, segmentation, **params):
        return segmentation


class PriorChannelDropoutTransform(BasicTransform):
    """Drop shape-prior channels (ch1–4) to prevent corrector over-reliance (§1.2.3).

    Two modes, checked in order per patch:
      1. Full-prior dropout: zero ALL prior channels with prob ``p_all`` — forces
         image-only segmentation (strongest regularization).
      2. Per-structure dropout: else, independently zero each prior channel with
         prob ``p_each`` — simulates a partial CNISP failure.
    Only ``image`` prior channels are zeroed; ch0 and the GT are untouched.
    """

    def __init__(self, prior_channel_indices: Sequence[int] = (1, 2, 3, 4),
                 p_all: float = 0.1, p_each: float = 0.25):
        super().__init__()
        self.prior_channel_indices = tuple(int(c) for c in prior_channel_indices)
        self.p_all = float(p_all)
        self.p_each = float(p_each)

    def get_parameters(self, **data_dict) -> dict:
        if np.random.rand() < self.p_all:
            zero = list(self.prior_channel_indices)                 # full dropout
        else:
            zero = [c for c in self.prior_channel_indices
                    if np.random.rand() < self.p_each]              # per-structure
        return {"zero_channels": zero}

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        for c in params["zero_channels"]:
            if c < img.shape[0]:
                img[c] = 0
        return img

    def _apply_to_segmentation(self, segmentation, **params):
        return segmentation
