#!/usr/bin/env python3
"""
Step 4: Result visualization & summary.

Reads the reconstruction folder produced by `scripts/03_infer.py`
(output_basedir/<model_name>/) and emits the CNISP-only artifacts:

    recon_layout.txt            file-tree summary
    cross_resolution_analysis/  iso-space pairwise Dice heatmaps + CSV
                                (prior self-consistency; no GT involved)
    native_sweep_summary.json   per-step native_space_step_XX/ audit

Per-step Dice trend / per-class / per-case figures are now produced by
the `compare` phase (nnunet/engine/build_method_summary.py) so that
CNISP and nnUNet share the same source set + bucket edges. The CNISP
slice lands at output_basedir/<model_name>/viz/CNISP_*.

No diagnostic interpretation is performed (see git history for the legacy
`04_diagnose.py` if you need reconstruction QC or latent-space analysis).

Usage:
    python scripts/04_visualization.py \\
        -p configs/paths.yaml \\
        -t configs/train_sty2.yaml \\
        -c configs/test_default.yaml \\
        -m orbital_ad_v2

    # Or skip configs and point straight at a folder:
    python scripts/04_visualization.py -d /path/to/reconstructions/<model>
"""

import argparse

import yaml

from engine.visualize import visualize_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--paths", default=None,
                        help="paths.yaml; required unless -d is provided.")
    parser.add_argument("-t", "--train_config", default=None,
                        help="Training config yaml (for num_classes).")
    parser.add_argument("-c", "--config", default=None,
                        help="Test config yaml (optional).")
    parser.add_argument("-m", "--model_name", default=None,
                        help="Model directory name under output_basedir.")
    parser.add_argument("-d", "--recon_dir", default=None,
                        help="Direct path to the reconstruction folder; "
                             "overrides -p/-m.")
    args = parser.parse_args()

    params: dict = {}
    if args.paths:
        with open(args.paths) as f:
            params.update(yaml.safe_load(f) or {})
    if args.train_config:
        with open(args.train_config) as f:
            params.update(yaml.safe_load(f) or {})
    if args.config:
        with open(args.config) as f:
            params.update(yaml.safe_load(f) or {})

    if args.recon_dir:
        params["recon_dir"] = args.recon_dir
    elif args.model_name:
        params["model_name"] = args.model_name
    else:
        parser.error("Provide either -d/--recon_dir or -m/--model_name.")

    visualize_results(params)


if __name__ == "__main__":
    main()
