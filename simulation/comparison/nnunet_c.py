#!/usr/bin/env python3
"""Turn the nnUNet-C ``eval_corrector`` per-case CSV into paired rows.

nnUNet-C (control C) refines a CNISP prelabel with a finetuned 5-channel
nnUNet. ``nnunet-c/diagnostics/eval_corrector.py`` already computes its
per-case Dice -- resampling the prediction onto each source's native GT grid
(order 0) and Dicing on that grid, the SAME convention ``compare_native``
uses for nnUNet-sparse -- and writes::

    case_id, source_id, step, dice_ON, dice_Recti, dice_Globe, dice_Fat, dice_mean

This module reads that CSV and emits rows in the EXACT shape
``compare_native`` writes into ``paired_per_source__<tag>__<exp>.csv`` so
nnUNet-C lands on the shared eff_res buckets next to nnUNet-sparse and CNISP.

The eval CSV has no ``eff_res_mm`` column (it is keyed by sparsification
``step``), so we join the effective resolution from the SAME per-(source,
step) ``eff_res_idx`` that ``compare_native`` builds from CNISP's
``sweep_results.pkl``. This guarantees byte-identical bucketing across the
three methods. A (source, step) with no eff_res entry falls into the
"unknown" bucket (eff_res cell left blank), mirroring the other methods.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ``dice_{name}`` column name per shared structure label.
_STRUCT_COL = {
    "ON": "dice_ON",
    "Globe": "dice_Globe",
    "Fat": "dice_Fat",
    "Recti": "dice_Recti",
    "mean": "dice_mean",
}


def load_nnunet_c_rows(
    eval_csv: Path,
    method_label: str,
    eff_res_idx: Dict[Tuple[str, int], float],
    struct_order: List[str],
    *,
    gt_source_by_sid: Optional[Dict[str, str]] = None,
) -> List[Dict[str, str]]:
    """Read an nnUNet-C eval CSV -> rows matching compare_native's schema.

    Parameters
    ----------
    eval_csv : path to ``eval_<control>_fold<N>.csv`` written by
        ``nnunet-c/diagnostics/eval_corrector.py``.
    method_label : method string to stamp on every row (e.g. ``nnUNet-C``).
    eff_res_idx : ``(source_id, step) -> effective_resolution_mm`` lookup,
        as built by ``nnunet.lib.metrics.build_eff_res_index`` from the
        CNISP run's ``sweep_results.pkl``. Used so nnUNet-C shares the exact
        eff_res values (and therefore buckets) of the other two methods.
    struct_order : foreground structure order (``STRUCT_ORDER``); the per-
        structure rows plus a trailing ``mean`` row are emitted per case.
    gt_source_by_sid : optional ``source_id -> gt_source`` map (e.g. from
        the resolved test sources) so the written rows carry the same
        ``gt_source`` tag as the nnUNet/CNISP rows. Defaults to blank.

    Returns
    -------
    list of dicts with the same keys compare_native writes::

        source_id, gt_source, method, step_size, slice_start_id,
        eff_res_mm, structure, dice
    """
    p = Path(eval_csv)
    if not p.exists():
        print(f"[nnunet_c] eval CSV not found: {p}; skipping nnUNet-C rows.",
              file=sys.stderr)
        return []

    names = list(struct_order) + ["mean"]
    rows: List[Dict[str, str]] = []
    n_cases = 0
    n_no_eff = 0
    with open(p) as f:
        reader = csv.DictReader(f)
        for r in reader:
            sid = (r.get("source_id") or "").strip()
            step_raw = r.get("step", "")
            try:
                step = int(float(step_raw))
            except (TypeError, ValueError):
                continue
            if not sid:
                continue
            n_cases += 1
            eff_res = eff_res_idx.get((sid, step))
            if eff_res is None:
                n_no_eff += 1
            eff_cell = "" if eff_res is None else f"{eff_res:.4f}"
            gt_source = (gt_source_by_sid or {}).get(sid, "")
            for name in names:
                col = _STRUCT_COL.get(name)
                val = r.get(col, "") if col else ""
                try:
                    dice = float(val)
                except (TypeError, ValueError):
                    continue
                rows.append({
                    "source_id": sid,
                    "gt_source": gt_source,
                    "method": method_label,
                    "step_size": str(step),
                    "slice_start_id": "0",
                    "eff_res_mm": eff_cell,
                    "structure": name,
                    "dice": f"{dice:.6f}",
                })

    print(f"[nnunet_c] {p.name}: {n_cases} case(s) -> {len(rows)} row(s) "
          f"(method={method_label!r}; {n_no_eff} case(s) without an eff_res "
          f"match fell into the 'unknown' bucket).")
    return rows


__all__ = ["load_nnunet_c_rows"]
