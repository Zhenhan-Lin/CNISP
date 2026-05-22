"""Resolve per-source paths and label schemes for the CNISP test set.

This module is the single source of truth for:
  * which 31 ``source_id``s back the 62 entries in ``test_cases.txt``,
  * how to find each source's CT image (for nnUNet input),
  * how to find each source's native-head GT label NIfTI,
  * which canonical label scheme each GT uses.

Everything downstream (``prepare_inputs.py``, ``build_smore_test_images.py``,
``compare_native.py``) calls into this module so the resolution logic
cannot drift.

Conventions
-----------
casename format: ``{source_id}_{eye}`` with ``eye in {OD, OS}``.

source_id formats:
  * ``atlas_<filename_stem>`` for atlas manual GT cases.
  * ``chk_<subject_id>``      for QA-kept nnUNet-prediction GT cases.

Both eyes of the same source share the SAME native-head GT NIfTI, so the
metadata for either eye is sufficient to recover ``original_nifti_path``.

GT label schemes (per orbital_shape_prior_st1/data_prep/canonical_align.py):
  * ``labelfusion``  -> {1: ON, 3: Recti, 5: Globe, 7: Fat}   (atlas)
  * ``nnunet``       -> {1: ON, 2: Recti, 3: Globe, 4: Fat}   (chk_*)

Atlas labels may carry a -1000 offset (so e.g. BG=-1000, ON=-999, ...);
``detect_label_scheme`` handles that case, and ``original_label_map`` in
the returned dataclass already accounts for it.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── CNISP label conventions (must match data_prep.canonical_align) ─

# Forward map: canonical structure name -> input-scheme label value.
LABELFUSION_LABELS = {"ON": 1, "Recti": 3, "Globe": 5, "Fat": 7}
NNUNET_LABELS = {"ON": 1, "Recti": 2, "Globe": 3, "Fat": 4}


@dataclass
class SourceInfo:
    """Everything we need to know about one of the 31 unique source scans."""

    source_id: str
    casenames: List[str]            # e.g. ["chk_14455_OD", "chk_14455_OS"]
    ct_image_path: Optional[Path]   # may be None until prepare_inputs runs
    gt_label_path: Path             # native-head GT NIfTI (original)
    gt_scheme: str                  # "labelfusion" or "nnunet"
    gt_label_offset: int            # 0, or -1000 for atlas-offset volumes
    gt_struct_to_value: Dict[str, int]  # incl. offset, ready for ==-style masking
    gt_source: str                  # "atlas" or "chk_pseudo"
    metadata_json_paths: List[Path] # per-eye metadata json (1 or 2 entries)


# ── test-cases parsing ────────────────────────────────────────────


def parse_test_cases(test_cases_path: Path) -> Dict[str, List[str]]:
    """Read test_cases.txt and group casenames by source_id."""
    if not test_cases_path.exists():
        raise FileNotFoundError(f"test_cases.txt not found: {test_cases_path}")

    sources: Dict[str, List[str]] = {}
    with open(test_cases_path) as f:
        for raw in f:
            casename = raw.strip()
            if not casename:
                continue
            if not (casename.endswith("_OD") or casename.endswith("_OS")):
                raise ValueError(
                    f"Unexpected casename (no _OD/_OS suffix): {casename!r}"
                )
            source_id = casename[:-3]
            sources.setdefault(source_id, []).append(casename)
    return sources


# ── per-eye metadata ──────────────────────────────────────────────


def load_metadata(meta_dir: Path, casename: str) -> Dict:
    p = meta_dir / f"{casename}.json"
    if not p.exists():
        raise FileNotFoundError(
            f"alignment metadata not found for {casename}: {p}"
        )
    with open(p) as f:
        return json.load(f)


def _detect_gt_offset(gt_label_path: Path) -> int:
    """Detect the atlas-style -1000 offset by sampling the minimum label.

    Mirrors ``canonical_align.detect_label_scheme`` for the offset case
    without re-running its full scheme detection (we already know the
    scheme from ``input_label_scheme`` in the metadata JSON).
    """
    try:
        import nibabel as nib  # local import keeps import-time deps light
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("nibabel is required to detect GT label offset") from exc

    img = nib.load(str(gt_label_path))
    arr = np.asarray(img.dataobj, dtype=np.int32)
    mn = int(arr.min())
    return mn if mn < 0 else 0


def build_struct_to_value(scheme: str, offset: int) -> Dict[str, int]:
    base = LABELFUSION_LABELS if scheme == "labelfusion" else NNUNET_LABELS
    return {name: val + offset for name, val in base.items()}


# ── per-source assembly ───────────────────────────────────────────


def resolve_sources(
    test_cases_path: Path,
    meta_dir: Path,
    atlas_image_dir: Optional[Path] = None,
    pivot_csv: Optional[Path] = None,
    pivot_subject_column: str = "subject",
    pivot_image_path_columns: Optional[List[str]] = None,
    detect_atlas_offset: bool = True,
    require_ct: bool = False,
) -> Tuple[List[SourceInfo], List[str]]:
    """Build a SourceInfo per unique source_id.

    Parameters
    ----------
    detect_atlas_offset
        If True, peek into each atlas GT NIfTI to detect the -1000 offset.
        Disable when you only need the canonical scheme (e.g. for the
        SMORE prep script which doesn't touch labels).
    require_ct
        If True, fail with a clear list of missing CTs when any cannot be
        resolved. Set False for stages that don't yet need CT paths.

    Returns
    -------
    sources, missing
        ``sources``: list of SourceInfo (sorted by source_id).
        ``missing``: list of human-readable strings, one per source whose
        CT could not be located. Empty when every source resolved.
    """
    grouped = parse_test_cases(test_cases_path)

    # Lazy CSV read for chk_* lookups.
    pivot_index: Dict[str, Dict[str, str]] = {}
    pivot_columns_available: List[str] = []
    if pivot_csv is not None and pivot_csv.exists():
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pandas is required to read the pivot CSV") from exc
        df = pd.read_csv(pivot_csv)
        pivot_columns_available = list(df.columns)
        if pivot_subject_column not in df.columns:
            raise KeyError(
                f"pivot_csv {pivot_csv} has no column {pivot_subject_column!r}. "
                f"Available columns: {pivot_columns_available}"
            )
        for _, row in df.iterrows():
            subj = str(row[pivot_subject_column])
            # Each subject may have multiple rows (sessions). Keep the
            # first row that exposes an existing image-path column.
            if subj in pivot_index:
                continue
            pivot_index[subj] = {k: ("" if pd.isna(row[k]) else str(row[k]))
                                 for k in df.columns}

    missing: List[str] = []
    out: List[SourceInfo] = []
    probe_cols = pivot_image_path_columns or [
        "ct_image_path", "image_path", "image", "ct", "t1"
    ]

    for source_id in sorted(grouped):
        casenames = sorted(grouped[source_id])
        # Both eyes share the same source GT; reading either metadata works.
        meta_paths = [meta_dir / f"{c}.json" for c in casenames]
        meta = load_metadata(meta_dir, casenames[0])
        gt_label_path = Path(meta["original_nifti_path"])
        scheme = meta["input_label_scheme"]

        if scheme not in ("labelfusion", "nnunet"):
            missing.append(
                f"{source_id}: unrecognised input_label_scheme={scheme!r} "
                f"in {meta_paths[0]}"
            )
            continue

        offset = 0
        if scheme == "labelfusion" and detect_atlas_offset and gt_label_path.exists():
            offset = _detect_gt_offset(gt_label_path)

        gt_source = "atlas" if source_id.startswith("atlas_") else "chk_pseudo"

        # ── locate CT image ─────────────────────────────────────────
        ct_image_path: Optional[Path] = None
        if source_id.startswith("atlas_"):
            if atlas_image_dir is None:
                missing.append(f"{source_id}: atlas_image_dir not set in config")
            else:
                # atlas source_id = "atlas_" + stem; image dir holds {stem}.nii.gz
                stem = source_id[len("atlas_"):]
                candidate = Path(atlas_image_dir) / f"{stem}.nii.gz"
                if candidate.exists():
                    ct_image_path = candidate
                else:
                    # Fall back: try the GT label's parent .. /atlas_images/
                    sibling = gt_label_path.parent.parent / "atlas_images" / f"{stem}.nii.gz"
                    if sibling.exists():
                        ct_image_path = sibling
                    else:
                        missing.append(
                            f"{source_id}: atlas CT not found. Tried\n"
                            f"  {candidate}\n  {sibling}"
                        )
        elif source_id.startswith("chk_"):
            subj = source_id[len("chk_"):]
            row = pivot_index.get(subj)
            if row is None:
                missing.append(
                    f"{source_id}: subject {subj!r} not found in pivot CSV "
                    f"(searched column {pivot_subject_column!r}). "
                    f"Available columns: {pivot_columns_available}"
                )
            else:
                found = None
                tried = []
                for col in probe_cols:
                    if col not in row:
                        continue
                    val = row[col]
                    tried.append(f"{col}={val}")
                    if val and Path(val).exists():
                        found = Path(val)
                        break
                if found is not None:
                    ct_image_path = found
                else:
                    missing.append(
                        f"{source_id}: no readable CT in pivot row. "
                        f"Probed: {tried or probe_cols}"
                    )
        else:
            missing.append(
                f"{source_id}: unknown source prefix (expected atlas_ or chk_)"
            )

        if require_ct and ct_image_path is None:
            # missing already recorded above
            continue

        out.append(
            SourceInfo(
                source_id=source_id,
                casenames=casenames,
                ct_image_path=ct_image_path,
                gt_label_path=gt_label_path,
                gt_scheme=scheme,
                gt_label_offset=offset,
                gt_struct_to_value=build_struct_to_value(scheme, offset),
                gt_source=gt_source,
                metadata_json_paths=meta_paths,
            )
        )

    return out, missing


def fail_on_missing(missing: List[str], context: str) -> None:
    """Convenience: print missing list and sys.exit on non-empty input."""
    if not missing:
        return
    print(
        f"[{context}] could not resolve {len(missing)} source(s):",
        file=sys.stderr,
    )
    for m in missing:
        print(f"  - {m}", file=sys.stderr)
    sys.exit(1)
