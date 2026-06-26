"""Assemble a control's nnUNet raw dataset (imagesTr/labelsTr/dataset.json).

Control-aware: A is external (Dataset835, no build); B/C build 5-channel raw
datasets from the staged ct/prelabel/gt softlinks. ch0 = degraded CT (order 3),
ch1..ch4 = binary prelabel channels (order 0), label = GT remapped to {1,2,3,4}
(order 0). Everything is resampled to the 835 plan-spacing grid so nnUNet's
preprocessing resample is a no-op (pothole-2 a-ii).

Called by scripts/build_dataset.py. The actual nnUNet plan/preprocess/train run
on the GPU box (this only writes the raw dataset + dataset.json).

Depends on numpy + nibabel + lib.*.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from lib import caselist as _cl
from lib import channels as _ch
from lib import config as _cfg
from lib import labels as _lab
from lib import resample as _rs
from lib import staging as _st


def _raw_root(raw_root: Optional[str]) -> Path:
    root = raw_root or os.environ.get("nnUNet_raw")
    if not root:
        raise RuntimeError(
            "nnUNet_raw is unset and --raw-root not given; cannot locate the "
            "nnUNet raw datasets dir. Export nnUNet_raw on the GPU/data host."
        )
    return Path(root)


def _dataset_dir(raw_root: Path, control: Dict) -> Path:
    name = f"Dataset{int(control['dataset_id']):03d}_{control['dataset_name']}"
    return raw_root / name


def _write_dataset_json(
    ds_dir: Path, control: Dict, cfg: Dict, num_training: int
) -> Path:
    n = int(control["n_channels"])
    if n == 1:
        channel_names = {"0": "CT"}
    else:
        channel_names = {"0": "CT"}
        for i in range(1, n):
            channel_names[str(i)] = "noNorm"
    labels = {k: int(v) for k, v in cfg["labels"].items()}
    dataset_json = {
        "channel_names": channel_names,
        "labels": labels,
        "numTraining": int(num_training),
        "file_ending": ".nii.gz",
        "name": control["dataset_name"],
        "description": (
            f"nnUNet-C corrector control; prelabel_source="
            f"{control['prelabel_source']}, experiment={cfg['experiment']}"
        ),
    }
    out = ds_dir / "dataset.json"
    with open(out, "w") as f:
        json.dump(dataset_json, f, indent=2)
    return out


def build_control(
    config_path: str,
    control_name: str,
    caller_file: str,
    splits: List[str] = ("train",),
    raw_root: Optional[str] = None,
) -> Dict:
    """Build the raw dataset for one control. Returns a summary dict."""
    cfg = _cfg.load_corrector_config(config_path, caller_file=caller_file)
    control = _cfg.get_control(cfg, control_name)
    control_name = control_name.upper()

    if control.get("external"):
        raise RuntimeError(
            f"control {control_name} is external (Dataset"
            f"{control['dataset_id']}); nothing to build."
        )

    # Hard leakage gate before we touch any data.
    _cl.assert_no_leakage(cfg)

    target_spacing = _rs.resolve_target_spacing(cfg)
    structures = _cfg.structures(cfg)
    raw = _raw_root(raw_root)
    ds_dir = _dataset_dir(raw, control)
    images_dir = ds_dir / "imagesTr"
    labels_dir = ds_dir / "labelsTr"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    print(f"[build] control={control_name} -> {ds_dir}")
    print(f"[build] target_spacing(835 plan)={target_spacing}")

    # Sources per split. Only 'train' lands in imagesTr/labelsTr (finetune);
    # 'test' staging is for predict-time provenance / debugging.
    split_sources = {
        "train": _cl.corrector_train_sources(cfg),
        "test": _cl.test_sources(cfg),
    }

    entries_by_split: Dict[str, List[Dict]] = {}
    assembled: List[Dict] = []
    for split in splits:
        sources = split_sources[split]
        source_infos = _lab.resolve_source_infos(cfg, sources)
        print(f"[build] split={split}: {len(sources)} source(s)")
        entries = _st.stage_split(
            cfg, control_name, control, split, sources, source_infos
        )
        entries_by_split[split] = entries

        if split != "train":
            continue  # only train is assembled into imagesTr/labelsTr
        for e in entries:
            si = source_infos[e["source_id"]]
            case_dir = (cfg["_resolved"]["staging_root"] / control_name
                        / split / e["case_id"])
            prelabel_path = (case_dir / "prelabel.nii.gz"
                             if control["prelabel_source"] != "none" else None)
            prelabel_stv = None
            if prelabel_path is not None:
                pre = _import_resolve(cfg, control, e["source_id"], e["step"], si)
                prelabel_stv = pre["struct_to_value"]
            summary = _ch.assemble_case(
                case_id=e["case_id"],
                ct_path=case_dir / "ct.nii.gz",
                gt_path=case_dir / "gt.nii.gz",
                target_spacing=target_spacing,
                n_channels=int(control["n_channels"]),
                structures=structures,
                gt_struct_to_value=dict(si.gt_struct_to_value),
                images_dir=images_dir,
                labels_dir=labels_dir,
                experiment=cfg["experiment"],
                prelabel_path=prelabel_path,
                prelabel_struct_to_value=prelabel_stv,
            )
            assembled.append(summary)
            print(f"  [assemble] {e['case_id']}: shape={summary['shape']} "
                  f"labels={summary['label_values']}")

    _st.write_manifest(cfg, control_name, entries_by_split)
    ds_json = _write_dataset_json(ds_dir, control, cfg, num_training=len(assembled))
    build_manifest = ds_dir / "corrector_build_manifest.json"
    with open(build_manifest, "w") as f:
        json.dump({"control": control_name, "cases": assembled}, f, indent=2)

    print(f"[build] wrote {len(assembled)} training case(s); dataset.json -> {ds_json}")
    return {
        "control": control_name,
        "dataset_dir": str(ds_dir),
        "num_training": len(assembled),
        "dataset_json": str(ds_json),
        "build_manifest": str(build_manifest),
    }


def _import_resolve(cfg, control, sid, step, si):
    """Lazy import to keep prelabel resolution local to the assembly loop."""
    from lib import prelabel as _pl
    return _pl.resolve_prelabel(cfg, control, sid, step, si)
