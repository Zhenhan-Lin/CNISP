"""Resolve the ch1..chN prelabel mask for controls B and C from ONE nnUNet pred.

Detail 2: for a fair B-vs-C comparison both prelabels must trace back to the
SAME nnUNet sparse prediction per (source_id, step):

  * raw nnUNet pred  : {nnunet_pred_root}/{exp}/sparse_step_{XX}/{sid}.nii.gz
  * B prelabel       : {nnunet_pred_root}/{exp}/sparse_step_{XX}_native/{sid}.nii.gz
                       (native-grid form of that same prediction)
  * C prelabel       : {cnisp_output_basedir}/{model}/runs/{exp}/{run_tag}/
                       native_space_step_{XX}/<gtstem>_cnisp_step{XX}.nii.gz
                       (CNISP latent-opt fit to the canonical form of that pred)

`source_prediction_dir` (the raw sparse_step dir) is recorded for both so the
staging manifest can assert they coincide.

Requires the repo root on sys.path (for resolve_gt label scheme).
Depends on stdlib + json.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from nnunet.data_prep.resolve_gt import NNUNET_LABELS  # noqa: E402

_STEP_DIR_RE = re.compile(r"^sparse_step_(\d+)$")


def step_dir_name(step: int) -> str:
    return f"sparse_step_{int(step):02d}"


def degraded_ct_path(cfg: Dict, sid: str, step: int) -> Path:
    """ch0 source: degraded sparse CT (channel-0 naming)."""
    root: Path = cfg["_resolved"]["degraded_ct_root"]
    exp = cfg["experiment"]
    return root / exp / step_dir_name(step) / f"{sid}_0000.nii.gz"


def source_prediction_dir(cfg: Dict, step: int) -> Path:
    """The single raw nnUNet sparse-prediction dir for this step (Detail 2)."""
    root: Path = cfg["_resolved"]["nnunet_pred_root"]
    exp = cfg["experiment"]
    return root / exp / step_dir_name(step)


def _b_prelabel_path(cfg: Dict, sid: str, step: int) -> Path:
    root: Path = cfg["_resolved"]["nnunet_pred_root"]
    exp = cfg["experiment"]
    return root / exp / f"{step_dir_name(step)}_native" / f"{sid}.nii.gz"


def _cnisp_run_dir(cfg: Dict) -> Path:
    base: Path = cfg["_resolved"]["cnisp_output_basedir"]
    return base / cfg["cnisp_model_name"] / "runs" / cfg["experiment"] / cfg["run_tag"]


def _c_prelabel_path(cfg: Dict, sid: str, step: int) -> Path:
    """CNISP native-space mask for this source/step, resolved via manifest.json."""
    native_dir = _cnisp_run_dir(cfg) / f"native_space_step_{int(step):02d}"
    manifest = native_dir / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"CNISP native manifest missing: {manifest}. Run Stage 3 "
            f"(gen_prelabels) for control C first."
        )
    with open(manifest) as f:
        mf = json.load(f)
    by_sid = mf.get("by_source_id", mf)
    if sid not in by_sid:
        raise KeyError(
            f"source {sid!r} not in {manifest} (has {len(by_sid)} sources)"
        )
    return native_dir / by_sid[sid]


def resolve_prelabel(
    cfg: Dict, control: Dict, sid: str, step: int, source_info
) -> Optional[Dict]:
    """Resolve prelabel {path, struct_to_value, scheme, source_prediction_dir}.

    Returns None for control A (no prelabel channels). `source_info` is the
    resolve_gt SourceInfo for `sid` (used for control C's original label scheme).
    """
    src = control["prelabel_source"]
    if src == "none":
        return None
    if src == "nnunet":
        return {
            "path": _b_prelabel_path(cfg, sid, step),
            "struct_to_value": dict(NNUNET_LABELS),     # Dataset835 output scheme
            "scheme": "nnunet",
            "source_prediction_dir": source_prediction_dir(cfg, step),
        }
    if src == "cnisp":
        return {
            "path": _c_prelabel_path(cfg, sid, step),
            # CNISP native mask is remapped to the source's ORIGINAL scheme.
            "struct_to_value": dict(source_info.gt_struct_to_value),
            "scheme": source_info.gt_scheme,
            "source_prediction_dir": source_prediction_dir(cfg, step),
        }
    raise ValueError(f"unknown prelabel_source {src!r}")


def available_steps(cfg: Dict, sid: str, control: Dict, source_info=None) -> List[int]:
    """Steps (>1) where ch0 degraded CT AND the prelabel both exist for `sid`.

    For control A only ch0 is required.
    """
    root: Path = cfg["_resolved"]["degraded_ct_root"]
    exp_dir = root / cfg["experiment"]
    if not exp_dir.is_dir():
        return []
    steps: List[int] = []
    for d in sorted(exp_dir.iterdir()):
        m = _STEP_DIR_RE.match(d.name)
        if not m:
            continue
        step = int(m.group(1))
        if not degraded_ct_path(cfg, sid, step).is_file():
            continue
        if control["prelabel_source"] != "none":
            try:
                pre = resolve_prelabel(cfg, control, sid, step, source_info)
            except (FileNotFoundError, KeyError):
                continue
            if pre is None or not Path(pre["path"]).exists():
                continue
        steps.append(step)
    return steps
