#!/usr/bin/env python3
"""DEPRECATED -- replaced by ``nnunet/engine/predict_sparse_iso.py``.

This script used to nearest-neighbour *duplicate* the sparse-grid
predictions along the through-plane axis to fake a native-grid mask for
Dice. That was wrong: nnUNet had already resampled its fine plan-spacing
(iso 0.5) output back down to the sparse input grid before saving, so the
NN-duplicated mask only ever carried sparse-resolution content on a dense
grid -- the genuine iso prediction was gone.

The sweep now runs ``engine/predict_sparse_iso.py`` instead, which keeps
the plan-spacing logits and resamples them onto the native grid with
nnUNet's own segmentation resampler. See that script (and the Phase 1b
section of ``nnunet/README.md``) for details.
"""

import sys


def main() -> int:
    sys.stderr.write(
        "[upsample_sparse_preds] DEPRECATED. This NN slice-duplication step "
        "was removed because it discarded nnUNet's iso-0.5 prediction.\n"
        "  Use instead:\n"
        "    python nnunet/engine/predict_sparse_iso.py --config nnunet/configs.yaml\n"
        "  It writes sparse_step_XX/ (sparse grid), sparse_step_XX_upsampled/ "
        "(iso 0.5), and sparse_step_XX_native/ (iso resampled to the native "
        "grid via nnUNet's resampler; the Dice target).\n"
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
