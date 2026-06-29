"""Effective-resolution bucket + per-method label constants.

Every comparison artefact written by ``compare_native.py``,
``build_method_summary.py``, and ``build_paired_summary.py`` shares the
same:

* foreground structure ordering (``STRUCT_ORDER``),
* nnUNet method label string in the paired CSV (``NNUNET_METHOD_LABEL``),
* default effective-resolution bucket edges (``DEFAULT_BUCKET_EDGES_MM``),
* eff_res -> bucket-label mapping (``assign_bucket``),
* bucket-label -> sort key (``bucket_sort_key`` -- 'unknown' last).

Having them as constants/functions in one module guarantees the per-
method PNGs and the paired PNGs never drift apart in axis labels,
sort order, or what counts as ``mean`` Dice. If you need to add a new
foreground class or shift a bucket boundary, do it here ONCE.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple


STRUCT_ORDER: List[str] = ["ON", "Globe", "Fat", "Recti"]
"""Foreground class order shared by paired CSV writers + summary plots."""


NNUNET_METHOD_LABEL: str = "nnUNet-sparse"
"""Method label written by ``compare_native.py`` for the nnUNet rows.

The CNISP rows are written as ``CNISP-atlasGT`` / ``CNISP-nnUNetPred``
(see ``compare_native._lookup_method_label``).
"""


NNUNET_INTERP_METHOD_LABEL: str = "nnUNet-interp"
"""Method label for the nnUNet Taubin post-processing control.

The control Taubin-smooths the degraded-grid nnUNet prediction on its own
grid, resamples it (order=0) onto the native CT grid, and Dices it against
the native GT. It is reported both as a standalone summary
(``interpolate_native.summarize``) and as an extra column in
``compare_native``. Like ``NNUNET_METHOD_LABEL`` it is nnUNet-only.
"""


NNUNET_C_METHOD_LABEL: str = "nnUNet-C"
"""Method label for the nnUNet-C "corrector" (control C) rows.

nnUNet-C refines a CNISP prelabel with a finetuned 5-channel nnUNet. Its
per-case Dice is produced by ``nnunet-c/diagnostics/eval_corrector.py`` (the
prediction is resampled onto each source's native GT grid, order 0, and
Dice is computed there -- the SAME convention compare_native uses), then
merged into the paired CSV by ``simulation.comparison.nnunet_c`` so it sits
on the shared eff_res buckets next to nnUNet-sparse and CNISP.
"""


DEFAULT_BUCKET_EDGES_MM: Tuple[float, ...] = (
    1.0, 2.0, 3.0, 4.0, 5.0, 6.5, 8.5, 11.0, 13.0,
)
"""Default effective-resolution bucket edges (mm, through-plane).

Used as the YAML fallback for ``summary_bucket_edges_mm`` in
``configs.yaml`` so the per-method and paired summaries fall back to
the same edges when the YAML key is missing.
"""


def assign_bucket(
    eff_res: Optional[float],
    edges: List[float],
) -> Tuple[int, str]:
    """Map an effective-resolution value to ``(index, label)``.

    The label format ``"(lo, ub]"`` is the half-open right-inclusive
    convention used by every CNISP/nnUNet by-eff_res table on disk so
    consumers can re-parse the edges without rebuilding them. ``None``
    or ``NaN`` returns ``(-1, "unknown")``.
    """
    if eff_res is None or (
        isinstance(eff_res, float) and math.isnan(eff_res)
    ):
        return -1, "unknown"
    for i, ub in enumerate(edges):
        if eff_res <= ub + 1e-6:
            lo = 0.0 if i == 0 else edges[i - 1]
            return i, f"({lo:.1f}, {ub:.1f}]"
    return len(edges), f"({edges[-1]:.1f}, inf]"


def bucket_sort_key(label: str) -> float:
    """Sort key that puts 'unknown' last and other buckets by lower bound.

    Returns the bucket's lower-bound float (or ``1e9`` for ``unknown``
    / malformed labels) so ``sorted(labels, key=bucket_sort_key)`` is
    stable across renames.
    """
    if label == "unknown":
        return 1e9
    try:
        return float(label.split(",")[0].lstrip("("))
    except ValueError:
        return 1e9


__all__ = [
    "STRUCT_ORDER",
    "NNUNET_METHOD_LABEL",
    "NNUNET_INTERP_METHOD_LABEL",
    "NNUNET_C_METHOD_LABEL",
    "DEFAULT_BUCKET_EDGES_MM",
    "assign_bucket",
    "bucket_sort_key",
]
