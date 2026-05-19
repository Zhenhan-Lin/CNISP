#!/usr/bin/env python3
"""
Map predictions to native image space (standalone).

Generates both native-spacing and isotropic versions.

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
        raise FileNotFoundError(f"Run inference first: {results_path}")

    with open(results_path, "rb") as f:
        results = pickle.load(f)

    meta_dir = Path(paths["aligned_dir"]) / "metadata"

    # Native spacing
    native_dir = pred_dir / "native_space"
    print(f"Mapping {len(results)} predictions to native space...")
    native_paths = map_results_to_native(results, meta_dir, native_dir)
    print(f"  {len(native_paths)} volumes → {native_dir}\n")

    # Isotropic
    iso_results = [r for r in results if "pred_class_map_iso" in r]
    if iso_results:
        from engine.native_mapping import map_iso_results_to_native
        iso_dir = pred_dir / "iso_space"
        print(f"Mapping {len(iso_results)} isotropic predictions to native space...")
        iso_paths = map_iso_results_to_native(iso_results, meta_dir, iso_dir)
        print(f"  {len(iso_paths)} files → {iso_dir}\n")
    else:
        print("No isotropic predictions found in inference_results.pkl.")
        print("Re-run inference with updated infer.py to generate them.")

    print("Done.")


if __name__ == "__main__":
    main()