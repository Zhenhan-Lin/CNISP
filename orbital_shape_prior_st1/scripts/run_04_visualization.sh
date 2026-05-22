#!/bin/bash
# ============================================================
# Step 4: Result Visualization & Summary
#
# Reads output_basedir/<MODEL_NAME>/ produced by Step 3 and writes
# the CNISP-only artifacts:
#   recon_layout.txt           file tree of the recon folder
#   cross_resolution_analysis/ iso-space pairwise Dice heatmaps + CSV
#                              (prior self-consistency; no GT involved)
#   native_sweep_summary.json  audit of native_space_step_XX/ outputs
#
# Dice trend / per-class / per-case figures come from the `compare`
# phase (nnunet/engine/build_method_summary.py), driven from the
# paired_per_source.csv compare_native produces. The CNISP slice of
# that bundle lands at output_basedir/<MODEL_NAME>/viz/CNISP_*.
#
# For reconstruction QC / latent analysis, recover the legacy
# `scripts/04_diagnose.py` from git history.
# ============================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

PATHS_YAML="$PROJECT_ROOT/configs/paths.yaml"
TRAIN_YAML="$PROJECT_ROOT/configs/train_sty2.yaml"
TEST_YAML="${1:-$PROJECT_ROOT/configs/test_default.yaml}"
MODEL_NAME="orbital_ad_v3"

echo "============================================================"
echo "Step 4: Result Visualization"
echo "  Paths config: $PATHS_YAML"
echo "  Train config: $TRAIN_YAML"
echo "  Test config:  $TEST_YAML"
echo "  Model:        $MODEL_NAME"
echo "============================================================"

python3 "$PROJECT_ROOT/scripts/04_visualization.py" \
    -p "$PATHS_YAML" \
    -t "$TRAIN_YAML" \
    -c "$TEST_YAML" \
    -m "$MODEL_NAME"

echo ""
echo "============================================================"
echo "Step 4 COMPLETE."
echo ""
echo "Artifacts under output_basedir/$MODEL_NAME/:"
echo "  recon_layout.txt"
echo "  cross_resolution_analysis/  (iso-space pairwise Dice + heatmaps)"
echo "  native_sweep_summary.json   (native_space_step_XX/ audit)"
echo ""
echo "Dice trend / per-class / per-case figures will appear under"
echo "  output_basedir/$MODEL_NAME/viz/   after the \`compare\` phase."
echo "============================================================"
