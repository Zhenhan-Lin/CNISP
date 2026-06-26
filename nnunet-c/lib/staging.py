"""Softlink each case's ct/prelabel/gt into a single flat tree (Detail 1).

Instead of chasing paths across work_dir/, CNISP runs/, and aligned_dir/ during
assembly, we first stage every file a case needs as a symlink under:

    nnunet-c/staging/{control}/{split}/corr_{sid}_step{XX}/
        ct.nii.gz        -> degraded sparse CT (pinned)
        prelabel.nii.gz  -> B native nnUNet pred | C CNISP native mask  (5ch only)
        gt.nii.gz        -> resolved native GT

Every link target is verified to exist at staging time (fail loud, fail early),
and a staging_manifest.json records the resolved sources + Detail-2 provenance.

Requires repo root on sys.path (for nnunet.helpers.fs). Depends on stdlib + json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from nnunet.helpers.fs import safe_symlink  # noqa: E402

from lib import prelabel as _pl


def case_id(sid: str, step: int) -> str:
    return f"corr_{sid}_step{int(step):02d}"


def _require(path: Path, what: str, case: str) -> Path:
    if not Path(path).exists():
        raise FileNotFoundError(f"[{case}] {what} not found: {path}")
    return Path(path)


def stage_case(
    cfg: Dict,
    control_name: str,
    control: Dict,
    split: str,
    sid: str,
    step: int,
    source_info,
) -> Dict:
    """Create the per-case softlinks; return the manifest entry."""
    cid = case_id(sid, step)
    case_dir = cfg["_resolved"]["staging_root"] / control_name / split / cid
    case_dir.mkdir(parents=True, exist_ok=True)

    ct_src = _require(_pl.degraded_ct_path(cfg, sid, step), "degraded CT (ch0)", cid)
    gt_src = _require(Path(source_info.gt_label_path), "GT label", cid)
    safe_symlink(ct_src, case_dir / "ct.nii.gz")
    safe_symlink(gt_src, case_dir / "gt.nii.gz")

    entry: Dict = {
        "case_id": cid,
        "source_id": sid,
        "step": int(step),
        "ct": str(ct_src),
        "gt": str(gt_src),
        "gt_scheme": source_info.gt_scheme,
        "n_channels": control["n_channels"],
    }

    if control["prelabel_source"] != "none":
        pre = _pl.resolve_prelabel(cfg, control, sid, step, source_info)
        pre_src = _require(Path(pre["path"]), "prelabel mask", cid)
        # Detail 2: the single nnUNet prediction this prelabel traces back to.
        src_pred_dir = _require(
            Path(pre["source_prediction_dir"]),
            "source nnUNet prediction dir (Detail 2)", cid,
        )
        safe_symlink(pre_src, case_dir / "prelabel.nii.gz")
        entry.update({
            "prelabel": str(pre_src),
            "prelabel_source": control["prelabel_source"],
            "prelabel_scheme": pre["scheme"],
            "source_prediction_dir": str(src_pred_dir),
        })
    return entry


def stage_split(
    cfg: Dict,
    control_name: str,
    control: Dict,
    split: str,
    sources: List[str],
    source_infos: Dict,
) -> List[Dict]:
    """Stage every (source, step) case for one split; return manifest entries."""
    entries: List[Dict] = []
    for sid in sources:
        si = source_infos[sid]
        steps = _pl.available_steps(cfg, sid, control, si)
        if not steps:
            print(f"  [stage] {sid}: no (ch0+prelabel) steps found; skipping")
            continue
        for step in steps:
            entries.append(
                stage_case(cfg, control_name, control, split, sid, step, si)
            )
        print(f"  [stage] {sid}: staged steps {steps}")
    return entries


def write_manifest(
    cfg: Dict, control_name: str, entries_by_split: Dict[str, List[Dict]]
) -> Path:
    """Write staging_manifest.json under staging/{control}/."""
    out = cfg["_resolved"]["staging_root"] / control_name / "staging_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "control": control_name,
        "experiment": cfg["experiment"],
        "run_tag": cfg.get("run_tag"),
        "n_cases": sum(len(v) for v in entries_by_split.values()),
        "splits": entries_by_split,
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    return out
