#!/usr/bin/env python3
"""
Step 3: Test A — Controlled Reconstruction.

Loads best val checkpoint by default. Use --checkpoint latest to override.

Usage:
    python scripts/03_infer.py \
        -p configs/paths.yaml \
        -t configs/train_default.yaml \
        -c configs/test_default.yaml \
        -m orbital_ad_v1

    # Use latest periodic checkpoint instead of best:
    python scripts/03_infer.py ... --checkpoint latest

Outputs (under output_basedir/<model_name>/):
    inference_results.pkl   per-case primary picks (one row per case,
                            picked by adaptive_step_sweep.primary_eff_res_mm)
                            -> consumed by 04_diagnose Sections 1/2 and
                            by map_to_native.py
    sweep_results.pkl       full per-(case, step) sweep
    step_XX/                NIfTI predictions, latents, viz, metadata
    test_results.csv        per-row sweep metrics
    test_summary.csv        eff_res-bucket aggregated metrics
"""

import argparse

import yaml

from engine.infer import infer_test_set


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--paths", required=True,
                        help="paths.yaml (data locations)")
    parser.add_argument("-t", "--train_config", required=True,
                        help="Training config (architecture params: latent_dim, num_classes, etc.)")
    parser.add_argument("-c", "--config", required=True,
                        help="Test config (latent_num_iters, adaptive_step_sweep, etc.)")
    parser.add_argument("-m", "--model_name", required=True,
                        help="Model directory name under model_basedir")
    parser.add_argument("--checkpoint", default="latest", choices=["best", "latest"],
                        help="Which checkpoint to load (default: best)")
    args = parser.parse_args()

    # Merge configs: paths → train (architecture) → test (overrides runtime settings)
    with open(args.paths) as f:
        params = yaml.safe_load(f)
    with open(args.train_config) as f:
        params.update(yaml.safe_load(f))
    with open(args.config) as f:
        params.update(yaml.safe_load(f))

    params["model_name"] = args.model_name
    params["checkpoint"] = args.checkpoint

    # infer_test_set writes inference_results.pkl + sweep_results.pkl itself
    infer_test_set(params)


if __name__ == "__main__":
    main()