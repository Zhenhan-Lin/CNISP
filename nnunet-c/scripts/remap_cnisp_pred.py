#!/usr/bin/env python3
"""Remap CNISP native-space masks to the nnUNet {1,2,3,4} scheme into data/cnisp_pred.

CNISP writes native masks in each source's ORIGINAL label scheme (labelfusion
{1,3,5,7} possibly with a -1000 atlas offset, or nnunet {1,2,3,4}). For the
corrector's control-C prelabel we want a uniform {0,1,2,3,4} (matching
data/nnunet_pred, which is already nnUNet scheme), remapped BY STRUCTURE NAME
(ON=1, Recti=2, Globe=3, Fat=4) -- never a value shift, because CNISP's canonical
ordering differs from nnUNet's.

This is a thin, scheme-agnostic converter: point --in-dir at the CNISP native
masks; each is auto-detected and remapped into --out-dir (default data/cnisp_pred).

Usage:
    python nnunet-c/scripts/remap_cnisp_pred.py --in-dir /path/to/native_masks
    python nnunet-c/scripts/remap_cnisp_pred.py --in-dir <dir> --scheme labelfusion
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.config import add_repo_to_syspath, load_corrector_config  # noqa: E402

_REPO = add_repo_to_syspath(__file__)

import numpy as np  # noqa: E402
import nibabel as nib  # noqa: E402

from lib.labels import remap_native_to_nnunet  # noqa: E402

_DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "corrector.yaml"


def _default_cnisp_pred_dir(config_path: str) -> Path:
    """data_root/cnisp_pred from corrector.yaml (best-effort; falls back)."""
    try:
        cfg = load_corrector_config(config_path, caller_file=__file__)
        cd = cfg["corrector_data"]
        root = cfg["_resolved"]["repo_root"]
        data_root = Path(cd["data_root"])
        data_root = data_root if data_root.is_absolute() else (root / data_root)
        return data_root / cd.get("cnisp_pred_dirname", "cnisp_pred")
    except Exception:  # noqa: BLE001 -- config may reference cluster-only paths
        return _REPO / "nnunet-c" / "data" / "cnisp_pred"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(_DEFAULT_CONFIG))
    ap.add_argument("--in-dir", required=True,
                    help="directory of CNISP native masks (original scheme)")
    ap.add_argument("--out-dir", default=None,
                    help="output dir (default: corrector data_root/cnisp_pred)")
    ap.add_argument("--glob", default="*.nii.gz", help="input glob (default %(default)s)")
    ap.add_argument("--scheme", default="auto",
                    choices=["auto", "labelfusion", "nnunet"],
                    help="source scheme (default auto-detect per file)")
    ap.add_argument("--strip-suffix", default="_cnisp",
                    help="strip this token from output filenames (default %(default)s)")
    args = ap.parse_args()

    structs = ["ON", "Recti", "Globe", "Fat"]   # fixed nnUNet channel order

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir) if args.out_dir else _default_cnisp_pred_dir(args.config)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob(args.glob))
    if not files:
        print(f"[remap_cnisp_pred] no files match {in_dir}/{args.glob}", file=sys.stderr)
        return 1
    print(f"[remap_cnisp_pred] {len(files)} mask(s): {in_dir} -> {out_dir}")

    report = {"in_dir": str(in_dir), "out_dir": str(out_dir), "files": []}
    n_ok = 0
    for fp in files:
        img = nib.load(str(fp))
        arr = np.asanyarray(img.dataobj)
        try:
            remapped, scheme, offset = remap_native_to_nnunet(
                arr, structs, scheme=args.scheme,
            )
        except ValueError as e:
            print(f"  [SKIP] {fp.name}: {e}", file=sys.stderr)
            report["files"].append({"file": fp.name, "error": str(e)})
            continue
        name = fp.name.replace(args.strip_suffix, "") if args.strip_suffix else fp.name
        dst = out_dir / name
        nib.save(nib.Nifti1Image(remapped, img.affine), str(dst))
        uniq = sorted(int(v) for v in np.unique(remapped))
        print(f"  {fp.name}: scheme={scheme} offset={offset} -> {name} values={uniq}")
        report["files"].append({"file": fp.name, "out": name, "scheme": scheme,
                                "offset": offset, "values": uniq})
        n_ok += 1

    with open(out_dir / "remap_cnisp_pred_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"[remap_cnisp_pred] wrote {n_ok}/{len(files)} -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
