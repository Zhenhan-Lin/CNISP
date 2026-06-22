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
    inference_results.pkl       per-case primary picks (one row per case,
                                picked by adaptive_step_sweep.primary_eff_res_mm)
                                -> consumed by map_to_native.py and by
                                scripts/04_visualization.py
    sweep_results.pkl           full per-(case, step) sweep
    step_XX/                    NIfTI predictions, latents, viz, metadata
    native_space/               primary picks mapped to native CT space
    native_space_step_XX/       every sweep step mapped to native space +
                                manifest.json indexed by source_id
    native_sweep_manifest.json  top-level index over per-step manifests
    test_results.csv            per-(case, step) sweep metrics
                                (eff_res_bucket column is included)
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
    parser.add_argument(
        "--test-label-source", default=None,
        choices=["atlas_gt", "nnunet_pred", "real_pair"],
        help=("Override test_label_source from the test yaml. "
              "atlas_gt = ceiling curve (sparsified canonical GT). "
              "nnunet_pred = deployment curve (canonical-aligned Dataset835 "
              "sparse-CT pred as latent-opt input; see test_default.yaml). "
              "real_pair = Turella sim3 (REAL low-res nnUNet pred input + "
              "separate hi-res GT, post-hoc rigid mask registration)."),
    )
    parser.add_argument(
        "--run-tag", default=None,
        help=("Override run_tag from the test yaml. Output lands at "
              "output_basedir/<model_name>/runs/<experiment>/<run_tag>/. "
              "Defaults to 'atlas_gt' which preserves the ceiling-curve layout."),
    )
    parser.add_argument(
        "--experiment", default=None, choices=["thin", "thick", "real"],
        help=("Simulation-strategy directory layer under runs/. "
              "thin = idealised point-sampling; thick = physical partial-"
              "volume degradation; real = Turella sim3 real paired data. "
              "When set for thin/thick it also drives the sweep degradation "
              "(sweep_mode) so the applied degradation matches the folder. "
              "Defaults: 'real' for real_pair, else sweep_mode (thin)."),
    )
    parser.add_argument(
        "--test-casefile", default=None,
        help=("Override test_casefile from the test yaml (a filename under "
              "casefiles_dir). Use a subset list to infer/compare only a few "
              "cases, e.g. test_cases_v7small.txt for ~20 images / 40 eyes. "
              "Downstream compare/viz only see the sources this run produced, "
              "so the head-to-head plots are restricted to the same subset."),
    )
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
    if args.test_label_source is not None:
        params["test_label_source"] = args.test_label_source
    if args.run_tag is not None:
        params["run_tag"] = args.run_tag
    if args.experiment is not None:
        params["experiment"] = args.experiment
    if args.test_casefile is not None:
        params["test_casefile"] = args.test_casefile

    # infer_test_set writes inference_results.pkl + sweep_results.pkl itself
    infer_test_set(params)


if __name__ == "__main__":
    main()