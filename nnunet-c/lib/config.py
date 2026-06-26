"""Config loading + repo/sys.path bootstrap for the nnUNet-C experiment.

Callers (scripts/, engine/, diagnostics/) typically do::

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # nnunet-c/
    from lib.config import load_corrector_config, add_repo_to_syspath
    add_repo_to_syspath(__file__)   # makes `nnunet.*` importable

This module reuses ``nnunet/helpers/config.py`` conventions but keeps the
corrector config self-contained. It also merges the shared nnUNet comparison
config (``nnunet_config_yaml``) so paths like ``work_dir`` / ``cnisp_paths_yaml``
have a single source of truth.

Only depends on PyYAML.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml

PathLike = Union[str, Path]


# ── repo-root + sys.path bootstrap ───────────────────────────────────


def find_repo_root(caller_file: PathLike) -> Path:
    """Walk up from ``caller_file`` to the dir containing both ``nnunet/`` and
    ``orbital_shape_prior_st1/`` (the CNISP repo root)."""
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


def add_repo_to_syspath(caller_file: PathLike) -> Path:
    """Prepend the repo root to ``sys.path`` so ``from nnunet... import`` works.

    Returns the repo root path. Intentionally does NOT add
    ``orbital_shape_prior_st1`` to avoid `engine`/`lib` top-level name clashes;
    CNISP inference is invoked out-of-process via its own shell wrapper.
    """
    root = find_repo_root(caller_file)
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)
    return root


# ── YAML helpers ─────────────────────────────────────────────────────


def load_yaml(path: PathLike) -> Dict:
    """Load a YAML file. Empty file -> ``{}``."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _resolve_under_root(root: Path, value: PathLike) -> Path:
    """Resolve a possibly-relative config path against the repo root."""
    p = Path(value)
    return p if p.is_absolute() else (root / p)


# ── corrector config ─────────────────────────────────────────────────


def load_corrector_config(
    config_path: PathLike, caller_file: Optional[PathLike] = None
) -> Dict:
    """Load corrector.yaml, merge shared nnUNet config, and resolve paths.

    The returned dict carries everything downstream needs, plus a ``_resolved``
    sub-dict with concrete absolute paths:

        repo_root, work_dir, aligned_dir, casefiles_dir, metadata_dir,
        cnisp_output_basedir, cnisp_model_basedir, nnunet_pred_root,
        degraded_ct_root, corrector_train_split, source_to_path_json.
    """
    config_path = Path(config_path)
    root = find_repo_root(caller_file or config_path)
    cfg = load_yaml(config_path)

    # Merge the shared nnUNet comparison config for paths.
    nnunet_cfg_path = _resolve_under_root(root, cfg["nnunet_config_yaml"])
    nnunet_cfg = load_yaml(nnunet_cfg_path)
    cfg["_nnunet_config"] = nnunet_cfg

    # CNISP paths.yaml (work_dir lives in the nnUNet config; aligned_dir/
    # casefiles_dir/output_basedir/model_basedir live in CNISP paths.yaml).
    cnisp_paths_yaml = Path(nnunet_cfg["cnisp_paths_yaml"])
    cnisp_paths = load_yaml(cnisp_paths_yaml)
    cfg["_cnisp_paths"] = cnisp_paths

    work_dir = Path(nnunet_cfg["work_dir"])
    aligned_dir = Path(cnisp_paths["aligned_dir"])
    casefiles_dir = Path(cnisp_paths["casefiles_dir"])

    nnunet_pred_root = cfg.get("nnunet_pred_root") or (work_dir / "prediction")
    degraded_ct_root = cfg.get("degraded_ct_root") or (work_dir / "input")

    cfg["_resolved"] = {
        "repo_root": root,
        "config_path": config_path,
        "work_dir": work_dir,
        "aligned_dir": aligned_dir,
        "casefiles_dir": casefiles_dir,
        "metadata_dir": aligned_dir / "metadata",
        "cnisp_paths_yaml": cnisp_paths_yaml,
        "cnisp_output_basedir": Path(cnisp_paths["output_basedir"]),
        "cnisp_model_basedir": Path(cnisp_paths["model_basedir"]),
        "nnunet_pred_root": Path(nnunet_pred_root),
        "degraded_ct_root": Path(degraded_ct_root),
        "source_to_path_json": work_dir / "source_to_path.json",
        "corrector_train_split": _resolve_under_root(
            root, cfg["corrector_train_split"]
        ),
        "nnunet_c_root": root / "nnunet-c",
        "staging_root": root / "nnunet-c" / "staging",
    }
    return cfg


def get_control(cfg: Dict, control: str) -> Dict:
    """Return the controls[<control>] block, validated."""
    control = control.upper()
    controls = cfg.get("controls", {})
    if control not in controls:
        raise KeyError(
            f"control {control!r} not in corrector.yaml controls "
            f"{sorted(controls)}"
        )
    return controls[control]


def structures(cfg: Dict) -> List[str]:
    """Fixed ch1..chN structure order (e.g. [ON, Recti, Globe, Fat])."""
    return list(cfg["structures"])
