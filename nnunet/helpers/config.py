"""Tiny config/IO/sys-path helpers shared across nnUNet scripts.

Most files under ``nnunet/`` are run as plain scripts (``python
nnunet/foo.py``) rather than as ``-m`` modules, and used to carry small
copy-pasted blocks for:

* loading the YAML config (``_load_yaml``),
* parsing a comma-separated CLI value into a clean list (``_csv_list``),
* stripping a NIfTI extension to get a "stem" (``_stem_of``),
* prepending ``orbital_shape_prior_st1`` to ``sys.path`` so the script
  can ``from data_prep.canonical_align import ...`` regardless of CWD.

The "make ``nnunet.*`` importable" bootstrap can't live here (chicken-
and-egg: this file IS inside ``nnunet/``), so every script still does a
single ``sys.path.insert`` for the repo root before importing from
``nnunet.helpers``. Once that one line is in place,
``add_cnisp_src_to_syspath`` handles the rest.

Callers do::

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from nnunet.helpers.config import (
        add_cnisp_src_to_syspath, load_yaml,
    )
    add_cnisp_src_to_syspath(__file__)
    ...
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Union

import yaml


PathLike = Union[str, Path]


# ── sys.path bootstrapping ───────────────────────────────────────────


def _find_repo_root(caller_file: PathLike) -> Path:
    """Walk up from ``caller_file`` to the directory containing both
    ``nnunet/`` and ``orbital_shape_prior_st1/``.

    Raises ``RuntimeError`` if no such ancestor exists -- which would
    mean the script was moved out of this repository.
    """
    p = Path(caller_file).resolve().parent
    while True:
        if (p / "nnunet").is_dir() and (p / "orbital_shape_prior_st1").is_dir():
            return p
        if p == p.parent:
            break
        p = p.parent
    raise RuntimeError(
        f"Could not locate repo root from {caller_file!r}: no ancestor "
        f"contains both 'nnunet/' and 'orbital_shape_prior_st1/'."
    )


def add_cnisp_src_to_syspath(caller_file: PathLike) -> Path:
    """Prepend ``<repo_root>/orbital_shape_prior_st1`` to ``sys.path``.

    Callers can then do ``from data_prep.canonical_align import ...``
    (the historical import shape used inside ``orbital_shape_prior_st1``)
    without having to install it as a package. Returns the CNISP source
    directory path.
    """
    root = _find_repo_root(caller_file)
    cnisp_src = root / "orbital_shape_prior_st1"
    s = str(cnisp_src)
    if s not in sys.path:
        sys.path.insert(0, s)
    return cnisp_src


# ── YAML / CSV / path utilities ──────────────────────────────────────


def load_yaml(path: PathLike) -> Dict:
    """Load a YAML config file. ``None`` (empty file) becomes ``{}``."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


def csv_list(s: str) -> List[str]:
    """Parse a comma-separated CLI value into a clean list of tokens.

    Empty / whitespace-only tokens are dropped, surrounding whitespace
    is stripped. ``csv_list("")`` returns ``[]``.
    """
    if not s:
        return []
    return [t.strip() for t in s.split(",") if t.strip()]


def stem_of(p: PathLike) -> str:
    """Return the filename stem, stripping both ``.nii`` and ``.nii.gz``.

    Mirrors the convention used in ``orbital_shape_prior_st1`` so paths
    written by ``map_results_to_native`` (which writes
    ``"{stem}{suffix}.nii.gz"``) round-trip cleanly.
    """
    name = Path(p).name
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    if name.endswith(".nii"):
        return name[: -len(".nii")]
    return Path(name).stem


__all__ = [
    "PathLike",
    "add_cnisp_src_to_syspath",
    "load_yaml",
    "csv_list",
    "stem_of",
]
