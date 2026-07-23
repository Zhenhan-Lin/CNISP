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
    """ch0 source: degraded sparse CT (channel-0 naming).

    step==1 is the DENSE baseline: it is never sparsified, so it lives in the
    shared ``input/native/`` dir, NOT under ``input/<exp>/sparse_step_01/``
    (which the sweep never creates). Routing step 1 there is what lets the
    corrector test set include the step_size=1 (dense) point for B and C.
    """
    root: Path = cfg["_resolved"]["degraded_ct_root"]
    if int(step) == 1:
        return root / "native" / f"{sid}_0000.nii.gz"
    exp = cfg["experiment"]
    return root / exp / step_dir_name(step) / f"{sid}_0000.nii.gz"


def source_prediction_dir(cfg: Dict, step: int) -> Path:
    """The single raw nnUNet sparse-prediction dir for this step (Detail 2)."""
    root: Path = cfg["_resolved"]["nnunet_pred_root"]
    exp = cfg["experiment"]
    return root / exp / step_dir_name(step)


def _b_prelabel_path(cfg: Dict, sid: str, step: int) -> Path:
    root: Path = cfg["_resolved"]["nnunet_pred_root"]
    # step==1 dense baseline: the native pred lives in the shared prediction/
    # native/ dir (prediction/<exp>/sparse_step_01_native/ is only a symlink to
    # it). Read the shared one directly so step 1 works even without the symlink.
    if int(step) == 1:
        return root / "native" / f"{sid}.nii.gz"
    exp = cfg["experiment"]
    return root / exp / f"{step_dir_name(step)}_native" / f"{sid}.nii.gz"


def _cnisp_run_dir(cfg: Dict) -> Path:
    base: Path = cfg["_resolved"]["cnisp_output_basedir"]
    return base / cfg["cnisp_model_name"] / "runs" / cfg["experiment"] / cfg["run_tag"]


def _corrector_data_root(cfg: Dict) -> Path:
    """Resolve ``corrector_data.data_root`` the same way every other corrector-data
    consumer does (``build_corrector_dataset.py:196`` / ``align_corrector_data.py:102``):
    relative paths are anchored at the repo root. Default ``nnunet-c/data``.

    Every corrector artifact "lives under data_root" (corrector.yaml), so the iso
    prelabel roots must follow it too -- otherwise a config that moves data_root
    (e.g. the FOV tree ``nnunet-c/data_fov_min_retain``) silently leaves its iso
    prelabels behind in the default ``nnunet-c/data`` tree. For the default
    ``nnunet-c/data`` this is byte-identical to the old ``nnunet_c_root / "data"``.
    """
    cd = cfg.get("corrector_data", {}) or {}
    dr = Path(cd.get("data_root", "nnunet-c/data"))
    return dr if dr.is_absolute() else (cfg["_resolved"]["repo_root"] / dr)


def _cnisp_iso_root(cfg: Dict) -> Path:
    """Root of the iso-0.5 prelabels CNISP emitted for the corrector.

    These are written by ``03_infer.py --emit-iso-prelabel-dir`` (see
    ``run_corrector_predict.sh`` EMIT_ISO). Layout mirrors native_space:
        <root>/native_space_step_XX/<stem>_cnisp_iso_stepXX.nii.gz + manifest.json
    Default: ``nnunet-c/data/cnisp_pred_test_iso`` (follows corrector_data.data_root).
    """
    name = (cfg.get("corrector_data", {}) or {}).get(
        "cnisp_iso_pred_dirname", "cnisp_pred_test_iso"
    )
    return _corrector_data_root(cfg) / name


def _cnisp_train_iso_root(cfg: Dict) -> Path:
    """Root of the iso prelabels emitted for the corrector TRAIN set.

    Written by ``032_cnisp_infer_corrector.py --emit-iso-prelabel-dir`` (via
    ``run_corrector_cnisp.sh`` EMIT_ISO). Same layout as the test iso root:
        <root>/native_space_step_XX/<stem>_cnisp_iso_stepXX.nii.gz + manifest.json
    Default: ``nnunet-c/data/cnisp_pred_train_iso`` (follows corrector_data.data_root).
    """
    name = (cfg.get("corrector_data", {}) or {}).get(
        "cnisp_train_iso_pred_dirname", "cnisp_pred_train_iso"
    )
    return _corrector_data_root(cfg) / name


def _c_train_iso_prelabel_path(cfg: Dict, sid: str, step: int) -> Path:
    """CNISP iso head mask for a TRAIN source/step (same iso-decode path as test).

    Reads the PER-SOURCE manifest ``manifest_by_source/<sid>.json`` that 032's
    iso emit writes (per-source so concurrent shard workers don't race a shared
    manifest). So the corrector's train ch1..4 come from the SAME iso decode as
    its test ch1..4.
    """
    step_dir = _cnisp_train_iso_root(cfg) / f"native_space_step_{int(step):02d}"
    mf = step_dir / "manifest_by_source" / f"{sid}.json"
    if not mf.is_file():
        raise FileNotFoundError(
            f"CNISP train iso prelabel missing for {sid!r} step {step}: {mf}. "
            f"Emit train iso prelabels first: EMIT_ISO=1 run_corrector_cnisp.sh."
        )
    name = json.load(open(mf))["file"]
    return step_dir / name


def _c_iso_prelabel_path(cfg: Dict, sid: str, step: int) -> Path:
    """CNISP iso-0.5 head mask for this source/step, resolved via manifest.json."""
    step_dir = _cnisp_iso_root(cfg) / f"native_space_step_{int(step):02d}"
    manifest = step_dir / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"CNISP iso manifest missing: {manifest}. Run the CNISP test with "
            f"iso emit first (EMIT_ISO=1 run_corrector_predict.sh, or "
            f"03_infer.py --emit-iso-prelabel-dir)."
        )
    with open(manifest) as f:
        mf = json.load(f)
    by_sid = mf.get("by_source_id", mf)
    if sid not in by_sid:
        raise KeyError(
            f"source {sid!r} not in {manifest} (has {len(by_sid)} sources)"
        )
    return step_dir / by_sid[sid]


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
