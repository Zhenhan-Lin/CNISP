"""Patch-size CLI helper shared by the canonical-align builders.

``build_dataset835_canonical_patches.py`` and
``build_dataset835_sparse_patches.py`` both need to:

1. Default the canonical-aligned cubic patch size (mm) to whatever the
   CNISP MLP was trained on (read from
   ``aligned_dir/metadata/*.json::patch_size_mm`` via
   ``orbital_shape_prior_st1.data_prep.canonical_align.infer_patch_size_mm``).
2. Allow ``--patch-size`` to override, but loudly warn when the
   override disagrees with the training-time value -- mismatched
   patches translate the predicted globe by
   ``(training_patch - this_patch) / 2 mm`` per axis.

This file centralises the resolution + warning so the two builders
can't drift from each other. ``infer_fn`` is passed in (rather than
imported here) so this module stays import-cheap; both builders must
``add_cnisp_src_to_syspath(__file__)`` and then
``from data_prep.canonical_align import infer_patch_size_mm`` before
calling ``resolve_patch_size_mm``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Optional


def resolve_patch_size_mm(
    cli_patch_size: Optional[float],
    train_meta_dir: Path,
    *,
    log_prefix: str,
    infer_fn: Callable[[Path], float],
) -> float:
    """Pick the canonical-align patch size in mm, with a training-mismatch warning.

    Parameters
    ----------
    cli_patch_size : the user's ``--patch-size`` (None means "auto").
    train_meta_dir : directory holding the CNISP training-time
        ``metadata/*.json`` (``patch_size_mm`` is read from any of
        them).
    log_prefix : ``"dataset835_canonical"`` / ``"dataset835_sparse"``,
        used in the print/warn banners so logs stay readable.
    infer_fn : callable that returns the training patch size given
        ``train_meta_dir`` (in practice
        ``data_prep.canonical_align.infer_patch_size_mm``). Passing it
        in keeps this helper free of CNISP imports.

    Returns
    -------
    The resolved patch size (mm) -- the training-time value when
    ``cli_patch_size is None``, otherwise the user's value (after
    warning on disagreement).
    """
    if cli_patch_size is None:
        patch_size = float(infer_fn(train_meta_dir))
        print(
            f"[{log_prefix}] patch_size_mm auto-detected from "
            f"{train_meta_dir} -> {patch_size:.3f} mm"
        )
        return patch_size

    patch_size = float(cli_patch_size)
    try:
        detected: Optional[float] = float(infer_fn(train_meta_dir))
    except (FileNotFoundError, ValueError):
        detected = None
    if detected is not None and abs(detected - patch_size) > 1e-3:
        print(
            f"[{log_prefix}] WARNING: --patch-size {patch_size:.3f} mm "
            f"differs from the training-time patch_size_mm="
            f"{detected:.3f} mm recorded in {train_meta_dir}. The MLP's "
            f"latent_coords were learned at {detected:.3f} mm/2, so this "
            f"patch will sit at a different physical offset.",
            file=sys.stderr,
        )
    return patch_size


__all__ = ["resolve_patch_size_mm"]
