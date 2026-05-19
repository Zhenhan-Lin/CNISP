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
"""

import argparse
import pickle
import yaml
from pathlib import Path

from engine.infer import infer_test_set


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--paths", required=True,
                        help="paths.yaml (data locations)")
    parser.add_argument("-t", "--train_config", required=True,
                        help="Training config (architecture params: latent_dim, num_classes, etc.)")
    parser.add_argument("-c", "--config", required=True,
                        help="Test config (inference params: latent_num_iters, slice_step_size, etc.)")
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

    # Run inference
    results = infer_test_set(params)

    # Save results for Step 4 (diagnostics)
    out_path = Path(params["output_basedir"]) / args.model_name / "inference_results.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()