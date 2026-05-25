#!/usr/bin/env python3
"""Stage SMORE-super-resolved CTs for nnUNetv2_predict.

For every source in ``${work_dir}/source_to_path.json`` (written by
``data_prep/prepare_inputs.py``), look up the canonical SMORE output at
``${smore_out_root}/<source_id>_smore.nii.gz`` (flat layout produced by
the latest ``engine/build_smore_test_images.py``) and symlink it as
``${work_dir}/input/smore/<source_id>_0000.nii.gz`` for nnUNetv2.

Sources missing the SMORE file get listed in a warning block at the end
with a hint to run ``nnunet/engine/build_smore_test_images.py`` first.
The phase continues with whatever is available, but exits with code 2
if zero sources are stageable so the pipeline visibly fails instead of
silently producing an empty input dir.

Usage
-----
    python nnunet/data_prep/prepare_smore_inputs.py --config nnunet/configs.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import yaml


# This file lives at nnunet/data_prep/prepare_smore_inputs.py;
# repo root is two directories up.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# Mirrors ``--smore-suffix`` default in nnunet/engine/build_smore_test_images.py.
# If you change it there, change it here.
_SMORE_SUFFIX = "_smore"


def _load_yaml(path: Path) -> Dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _safe_symlink(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="nnunet/configs.yaml")
    args = ap.parse_args()

    cfg = _load_yaml(Path(args.config))

    work_dir = Path(cfg["work_dir"])
    smore_out_root = Path(cfg["smore_out_root"])
    suffix = str(cfg.get("smore_suffix", _SMORE_SUFFIX))

    source_to_path = work_dir / "source_to_path.json"
    if not source_to_path.exists():
        print(f"[prepare_smore_inputs] {source_to_path} missing -- "
              f"run nnunet/data_prep/prepare_inputs.py first.",
              file=sys.stderr)
        return 2
    with open(source_to_path) as f:
        manifest_in = json.load(f)

    smore_input_dir = work_dir / "input" / "smore"
    smore_input_dir.mkdir(parents=True, exist_ok=True)

    print(f"[prepare_smore_inputs] sources:         {len(manifest_in)}")
    print(f"[prepare_smore_inputs] smore_out_root:  {smore_out_root}")
    print(f"[prepare_smore_inputs] out_dir:         {smore_input_dir}")

    staged: List[str] = []
    missing: List[str] = []

    for sid in sorted(manifest_in):
        smore_path = smore_out_root / f"{sid}{suffix}.nii.gz"
        if not (smore_path.exists() or smore_path.is_symlink()):
            missing.append(f"{sid}: not found at {smore_path}")
            continue
        dst = smore_input_dir / f"{sid}_0000.nii.gz"
        _safe_symlink(smore_path.resolve(), dst)
        staged.append(sid)

    if missing:
        print(f"\n[prepare_smore_inputs] {len(missing)} source(s) without "
              f"SMORE output:", file=sys.stderr)
        for line in missing:
            print(f"  - {line}", file=sys.stderr)
        print(f"  Run nnunet/engine/build_smore_test_images.py to produce them.",
              file=sys.stderr)

    print(f"\n[prepare_smore_inputs] staged {len(staged)} symlink(s); "
          f"{len(missing)} missing.")

    if not staged:
        print(f"[prepare_smore_inputs] zero sources staged -- aborting.",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
