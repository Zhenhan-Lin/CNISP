#!/usr/bin/env python3
"""
Step 2: Train the orbital shape prior.

Usage:
    python scripts/02_train.py -p configs/paths.yaml -c configs/train_default.yaml
"""

import argparse
import yaml

from engine.train import train_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--paths", required=True)
    parser.add_argument("-c", "--config", required=True)
    args = parser.parse_args()

    with open(args.paths) as f:
        params = yaml.safe_load(f)
    with open(args.config) as f:
        params.update(yaml.safe_load(f))

    train_model(params)


if __name__ == "__main__":
    main()
