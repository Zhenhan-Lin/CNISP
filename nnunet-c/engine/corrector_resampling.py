"""Per-channel data resampling for the nnUNet-C corrector (ch0 vs binary ch1..N).

nnUNet's default `resampling_fn_data` resamples ALL image channels with one order
(cubic spline, order 3). For the corrector that corrupts the binary prelabel
channels (ch1..N) into continuous values. This drop-in replacement resamples:

  * channel 0 (the CT)            with order 3  (proper intensity resampling),
  * channels 1..N (binary masks)  with order 0  (nearest -> stays {0,1}).

Both go to the SAME target grid, so the channel stack stays co-registered; only
the interpolation order differs per channel.

INSTALL: nnUNet discovers resampling functions by name under
``nnunetv2/preprocessing/resampling/``, so this file must live there. Copy it in:

    python - <<'PY'
    import shutil, os, nnunetv2.preprocessing.resampling as r
    dst = os.path.join(os.path.dirname(r.__file__), "corrector_resampling.py")
    shutil.copyfile("nnunet-c/engine/corrector_resampling.py", dst)
    print("installed ->", dst)
    PY

Then set the plan's ``resampling_fn_data`` to ``resample_corrector_data_to_shape``
(build_finetune_plan.py does this with --binary-resampling, the default).
"""

from __future__ import annotations

import numpy as np

from nnunetv2.preprocessing.resampling.default_resampling import (
    resample_data_or_seg_to_shape,
)


def resample_corrector_data_to_shape(
    data,
    new_shape,
    current_spacing,
    new_spacing,
    is_seg: bool = False,
    order: int = 3,
    order_z: int = 0,
    force_separate_z=None,
):
    """Resample ch0 with order 3, ch1..N with order 0 (binary-preserving).

    Signature matches nnUNet's resampling_fn_data contract; the incoming ``order``
    is intentionally ignored in favour of the per-channel policy above.
    """
    ch0 = resample_data_or_seg_to_shape(
        data[0:1], new_shape, current_spacing, new_spacing,
        is_seg=False, order=3, order_z=order_z, force_separate_z=force_separate_z,
    )
    if data.shape[0] == 1:
        return ch0
    rest = resample_data_or_seg_to_shape(
        data[1:], new_shape, current_spacing, new_spacing,
        is_seg=False, order=0, order_z=0, force_separate_z=force_separate_z,
    )
    return np.concatenate([ch0, rest], axis=0)
