"""nnUNet raw-dataset path + dataset.json helpers for the corrector builders.

This used to also hold a full staging-flow train builder (``build_control``); that
legacy path bypassed the single ``engine/convert.py::convert_case`` converter and
has been removed. Only the small raw-dataset helpers remain, shared by the active
train builder (``scripts/build_corrector_dataset.py``).

The per-(case,step) channel conversion lives in ``engine/convert.py`` (the one
converter used by BOTH the train builder and the test builder
``scripts/build_corrector_testset.py``).

Depends only on stdlib.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional


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
