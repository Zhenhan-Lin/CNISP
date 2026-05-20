#!/usr/bin/env python3
"""
Map per-case primary predictions to native image space (standalone).

Reads ``inference_results.pkl`` (primary picks; one row per case) written
by ``infer_test_set`` and inverts the canonical alignment back into the
original full-head NIfTI grid.

Patch-level isotropic predictions are already saved on disk by inference
at ``output_basedir/<model>/step_XX/iso_space/``. This script does not
re-emit them; if you need an isotropic full-head merge, call
``engine.native_mapping.map_iso_results_to_native`` from a custom driver
with a results list that includes ``pred_class_map_iso``.

Usage:
    PYTHONPATH=. python3 scripts/map_to_native.py -p configs/paths.yaml -m orbital_ad_v1
"""

import argparse
import pickle
from pathlib import Path

import yaml

from engine.native_mapping import map_results_to_native


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--paths", required=True)
    parser.add_argument("-m", "--model_name", required=True)
    args = parser.parse_args()

    with open(args.paths) as f:
        paths = yaml.safe_load(f)

    pred_dir = Path(paths["output_basedir"]) / args.model_name
    results_path = pred_dir / "inference_results.pkl"

    if not results_path.exists():
        raise FileNotFoundError(
            f"Run inference first: {results_path}"
        )

    with open(results_path, "rb") as f:
        results = pickle.load(f)

    meta_dir = Path(paths["aligned_dir"]) / "metadata"
    native_dir = pred_dir / "native_space"

    print(f"Mapping {len(results)} primary predictions to native space...")
    native_paths = map_results_to_native(results, meta_dir, native_dir)
    print(f"  {len(native_paths)} volumes → {native_dir}")

    print("Done.")


if __name__ == "__main__":
    main()