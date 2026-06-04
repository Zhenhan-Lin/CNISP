"""
Sparse observation container.

Encapsulates the output of a degradation operation with full provenance.
The container deliberately has NO densification / upsampling method —
densification is exclusively the shape model's responsibility.
"""

from dataclasses import dataclass

import torch


@dataclass
class SparseObservation:
    """A degraded (sparse) volume with its honest affine and provenance."""

    volume: torch.Tensor    # [S1, S2, S3] sparse label (int) or image (float)
    spacing: torch.Tensor   # [3] voxel spacing after sparsification
    offset: torch.Tensor    # [3] first-voxel-center position after sparsification
    mode: str               # "thin" | "thick" | "dense"
    step: int               # subsampling factor (1 = dense)
    axis: int               # sparsification axis (0/1/2)
    modality: str           # "ct" | "mri"
    label_source: str       # "gt" | "nnunet" | "reconnet" | ...
