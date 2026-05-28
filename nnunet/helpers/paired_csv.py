"""Readers/filters for ``comparison/paired_per_source__<run_tag>.csv``.

The paired CSV is the single source of truth for per-method Dice plots
(``build_method_summary.py``) and the head-to-head paired plots
(``build_paired_summary.py``). Both files used to carry their own copy
of:

* numeric coercion of ``step_size`` (int), ``dice`` (float), and the
  optional ``eff_res_mm`` (float -- empty cells become NaN),
* source-prefix include/exclude filtering against the ``source_id``
  field,
* the CLI -> config fallback for the include/exclude prefix lists.

This module centralises all three so the two summary scripts can never
disagree on what rows they consume or how they're filtered.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

from nnunet.helpers.config import csv_list


PathLike = Union[str, Path]


def read_paired_csv(
    path: PathLike,
    methods: Union[str, Iterable[str]],
) -> List[Dict]:
    """Read paired_per_source__<run_tag>.csv, keeping only requested methods.

    Parameters
    ----------
    path : path to the CSV.
    methods : a single method label (``str``) or an iterable of method
        labels to keep (e.g. ``"nnUNet-sparse"`` or
        ``["nnUNet-sparse", "CNISP-atlasGT"]``). Other rows are dropped.

    Returns
    -------
    list of dicts with keys::

        source_id, gt_source, method, step_size (int),
        eff_res_mm (float; NaN if absent), structure, dice (float)

    Rows that fail numeric coercion (``step_size``/``dice`` missing or
    non-numeric) are silently skipped, matching the legacy behaviour of
    the two summary scripts.

    Raises
    ------
    FileNotFoundError : when ``path`` does not exist.
    SystemExit       : when ZERO rows survive the method filter (the
        helper prints a hint pointing the user to ``compare_native.py``
        / ``run_pipeline.sh::compare``).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Run `nnunet/compare_native.py` first "
            f"(or the `compare` phase of run_pipeline.sh)."
        )
    if isinstance(methods, str):
        keep = {methods}
    else:
        keep = set(methods)

    rows: List[Dict] = []
    with open(p) as f:
        for r in csv.DictReader(f):
            m = r.get("method")
            if m not in keep:
                continue
            try:
                step = int(float(r["step_size"]))
                dice = float(r["dice"])
            except (KeyError, ValueError):
                continue
            eff_str = r.get("eff_res_mm", "")
            try:
                eff = float(eff_str) if eff_str else float("nan")
            except ValueError:
                eff = float("nan")
            rows.append({
                "source_id": r.get("source_id", ""),
                "gt_source": r.get("gt_source", ""),
                "method": m,
                "step_size": step,
                "eff_res_mm": eff,
                "structure": r.get("structure", ""),
                "dice": dice,
            })

    if not rows:
        raise SystemExit(
            f"{p}: no rows matched methods={sorted(keep)!r}. "
            f"Did `compare_native.py` write these methods' rows?"
        )

    # Warn (but don't fail) when one of multiple requested methods has
    # zero rows -- matches the legacy ``build_paired_summary`` behaviour.
    seen = {r["method"] for r in rows}
    missing = [m for m in sorted(keep) if m not in seen]
    if missing:
        print(
            f"[paired_csv] WARN: no rows for method(s) {missing!r} in {p}",
            file=sys.stderr,
        )
    return rows


def apply_source_filter(
    rows: List[Dict],
    include_prefixes: Iterable[str],
    exclude_prefixes: Iterable[str],
) -> List[Dict]:
    """Restrict rows by ``source_id`` prefix.

    ``include_prefixes`` (if non-empty) keeps only sources whose id
    starts with one of the listed prefixes. ``exclude_prefixes`` then
    drops any matching sources -- evaluated AFTER include, so an
    explicit deny can carve a hole out of an include set if both are
    passed.

    Used by the compare phase to keep paired plots focused on the
    human-labelled atlas cohort, excluding ``chk_*`` deployment cases
    whose chk_pseudo GT in ``test_label_source=nnunet_pred`` mode is
    the same Dataset835 dense prediction that nnUNet-sparse at step=1
    IS (producing a structural identity-1.0 row that inflates the
    deployment curve).
    """
    inc = tuple(p for p in include_prefixes if p)
    exc = tuple(p for p in exclude_prefixes if p)
    if not inc and not exc:
        return list(rows)
    out: List[Dict] = []
    for r in rows:
        sid = r.get("source_id", "")
        if inc and not sid.startswith(inc):
            continue
        if exc and sid.startswith(exc):
            continue
        out.append(r)
    return out


def resolve_source_prefix_filters(
    cli_include: Optional[str],
    cli_exclude: Optional[str],
    cfg: Dict,
    *,
    include_key: str = "viz_include_source_prefixes",
    exclude_key: str = "viz_exclude_source_prefixes",
) -> Tuple[List[str], List[str]]:
    """Resolve include/exclude prefix CLI flags against config defaults.

    Convention used by both summary scripts:

    * ``--include-source-prefixes`` / ``--exclude-source-prefixes``
      accept a comma-separated string; passing ``None`` falls back to
      the matching YAML key (``viz_include_source_prefixes`` /
      ``viz_exclude_source_prefixes``).
    * Passing the empty string explicitly disables the filter (returns
      ``[]``).
    """
    if cli_include is None:
        include = list(cfg.get(include_key, []))
    else:
        include = csv_list(cli_include)
    if cli_exclude is None:
        exclude = list(cfg.get(exclude_key, []))
    else:
        exclude = csv_list(cli_exclude)
    return include, exclude


__all__ = [
    "read_paired_csv",
    "apply_source_filter",
    "resolve_source_prefix_filters",
]
