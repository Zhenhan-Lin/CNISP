"""
Shared simulation module for CNISP.

Provides degradation operators (thin/thick), slice profile kernels,
and affine helpers used by both the shape prior training/test pipeline
and the nnUNet deployment pipeline.
"""

from simulation.degradation import degrade_thin, degrade_thick  # noqa: F401
from simulation.slice_profile import get_kernel  # noqa: F401
from simulation.observation import SparseObservation  # noqa: F401
from simulation.registration import register_mask_to_gt  # noqa: F401
