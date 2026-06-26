#!/usr/bin/env bash
# Build a corrector control's nnUNet raw dataset (imagesTr/labelsTr/dataset.json).
#
# Usage:
#   bash nnunet-c/run_build_dataset.sh B            # stacked nnUNet (Dataset855)
#   bash nnunet-c/run_build_dataset.sh C            # CNISP-conditioned (Dataset845)
#   CONFIG=/path/corrector.yaml bash nnunet-c/run_build_dataset.sh B
#
# Requires: $nnUNet_raw set; degraded CTs + prelabels already produced (Stages 1
# and, for control C, 3 of run_full_pipeline.sh).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
CONTROL="${1:?usage: run_build_dataset.sh <B|C> [splits]}"
SPLITS="${2:-train}"
CONFIG="${CONFIG:-$HERE/configs/corrector.yaml}"

echo "[run_build_dataset] control=$CONTROL splits=$SPLITS config=$CONFIG"
python3 "$HERE/scripts/build_dataset.py" \
    --config "$CONFIG" \
    --control "$CONTROL" \
    --splits "$SPLITS"
