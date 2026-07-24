"""
Resolve the exact target spacing (and voxel volume) from the corrector's finetune
nnU-Net plan — the SINGLE source of truth for the FOV-completion grid.

The corrector does NOT use nnU-Net's default plan; `build_finetune_plan.py` merges
Dataset835's target spacing/architecture into the 855/845 plan and writes
`nnUNetPlansFinetune.json` (see engine/plan_merge.py). Target spacing therefore
lives at:

    plan["configurations"][<configuration>]["spacing"]   # [z, y, x], post-transpose

Read exactly like the rest of the corrector reads plans — plain JSON, no
PlansManager import (nnunetv2 isn't importable off the GPU box). Never infer
spacing from "iso-0.5", filenames, or a hand-typed value (review §2.1).

Axis convention (review §2.2): preprocessed arrays, spacing, and the region
masks all use z, y, x; visible_box = [(z_lo,z_hi),(y_lo,y_hi),(x_lo,x_hi)],
half-open.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Optional, Tuple


def _preprocessed_dataset_dir(dataset_id: int, dataset_name: str) -> Path:
    preproc = os.environ.get("nnUNet_preprocessed")
    if not preproc:
        raise RuntimeError("$nnUNet_preprocessed is unset (needed to locate the plan).")
    return Path(preproc) / f"Dataset{int(dataset_id):03d}_{dataset_name}"


def resolve_target_spacing_from_plan(
    plans_file: str | Path,
    configuration: str,
) -> Tuple[float, float, float]:
    """Return ``(sz, sy, sx)`` = the exact target spacing from a plan JSON.

    ``plans_file``: path to e.g. ``.../Dataset847_.../nnUNetPlansFinetune.json``.
    ``configuration``: e.g. ``"3d_fullres"``.
    """
    plan = json.loads(Path(plans_file).read_text())
    cfgs = plan.get("configurations", {})
    if configuration not in cfgs:
        raise KeyError(f"{plans_file}: no configuration {configuration!r} "
                       f"(has {sorted(cfgs)})")
    spacing = cfgs[configuration].get("spacing")
    if not spacing or len(spacing) != 3:
        raise ValueError(f"{plans_file}[{configuration}].spacing missing/!=3: {spacing}")
    return tuple(float(v) for v in spacing)  # type: ignore[return-value]


def resolve_from_corrector_config(
    corrector_cfg: dict,
    control: dict,
    out_plan_name: str = "nnUNetPlansFinetune",
    configuration: Optional[str] = None,
) -> Tuple[Tuple[float, float, float], Path, str]:
    """Locate the finetune plan for a control from the corrector config + env, and
    return ``(spacing_zyx, plans_file, configuration)``.

    Mirrors build_finetune_plan.py's dataset-dir resolution so the FOV pipeline
    reads the SAME plan the corrector trains on.
    """
    configuration = configuration or corrector_cfg["configuration"]
    ddir = _preprocessed_dataset_dir(control["dataset_id"], control["dataset_name"])
    plans_file = ddir / f"{out_plan_name}.json"
    if not plans_file.is_file():
        raise FileNotFoundError(
            f"finetune plan not found: {plans_file}. Run build_finetune_plan.py "
            f"(+ nnUNetv2_preprocess) for this control first.")
    return resolve_target_spacing_from_plan(plans_file, configuration), plans_file, configuration


def voxel_volume_mm3(spacing_zyx: Tuple[float, float, float]) -> float:
    return float(spacing_zyx[0] * spacing_zyx[1] * spacing_zyx[2])


def mm3_to_voxels(volume_mm3: float, spacing_zyx: Tuple[float, float, float]) -> int:
    """Ceil-convert a physical volume floor to a voxel count (review §5.5/12.6)."""
    return int(math.ceil(float(volume_mm3) / voxel_volume_mm3(spacing_zyx)))


def mm_to_voxels(width_mm: float, spacing_zyx: Tuple[float, float, float]) -> Tuple[int, int, int]:
    return tuple(int(round(float(width_mm) / s)) for s in spacing_zyx)  # type: ignore[return-value]


def _selftest() -> int:
    import tempfile
    plan = {"plans_name": "nnUNetPlansFinetune",
            "configurations": {"3d_fullres": {"spacing": [0.5, 0.4765625, 0.4765625]}}}
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "nnUNetPlansFinetune.json"
        p.write_text(json.dumps(plan))
        sp = resolve_target_spacing_from_plan(p, "3d_fullres")
        assert sp == (0.5, 0.4765625, 0.4765625), sp
        vv = voxel_volume_mm3(sp)
        assert abs(vv - 0.5 * 0.4765625 ** 2) < 1e-9
        # 20 mm^3 floor at this spacing
        n = mm3_to_voxels(20.0, sp)
        assert n == math.ceil(20.0 / vv), n
        # seam 12 vox physical width along each axis
        assert mm_to_voxels(12 * 0.5, sp)[0] == 12
        try:
            resolve_target_spacing_from_plan(p, "2d")
            raise AssertionError("should have raised on missing configuration")
        except KeyError:
            pass
    print(f"spacing={sp}  voxel_vol={vv:.5f}mm3  20mm3->{n}vox")
    print("PLAN-SPACING SELF-TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
