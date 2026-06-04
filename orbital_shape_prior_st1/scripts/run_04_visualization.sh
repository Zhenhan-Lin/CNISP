#!/bin/bash
# ============================================================
# Step 4: Result Visualization & Summary
#
# Reads output_basedir/<MODEL_NAME>/runs/<RUN_TAG>/ produced by Step 3
# and writes the CNISP-only artifacts there:
#   recon_layout.txt           file tree of the recon folder
#   cross_resolution_analysis/ iso-space pairwise Dice heatmaps + CSV
#                              (prior self-consistency; no GT involved)
#   native_sweep_summary.json  audit of native_space_step_XX/ outputs
#
# Dice trend / per-class / per-case figures come from the `compare`
# phase (nnunet/engine/build_method_summary.py), driven from the
# paired_per_source__<run_tag>.csv compare_native produces. The CNISP
# slice of that bundle lands at
# output_basedir/<MODEL_NAME>/viz/<RUN_TAG>/<METHOD_LABEL>_*.
#
# For reconstruction QC / latent analysis, recover the legacy
# `scripts/04_diagnose.py` from git history.
# ============================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$REPO_ROOT:${PYTHONPATH:-}"

PATHS_YAML="$PROJECT_ROOT/configs/paths.yaml"
TRAIN_YAML="$PROJECT_ROOT/configs/train_sty2.yaml"
TEST_YAML="${1:-$PROJECT_ROOT/configs/test_default.yaml}"
MODEL_NAME="orbital_ad_v5"
# Optional 2nd arg: run_tag override (atlas_gt / nnunet_pred / ...).
# When unset, the test yaml's run_tag wins (defaults to atlas_gt).
RUN_TAG="${2:-}"

echo "============================================================"
echo "Step 4: Result Visualization"
echo "  Paths config: $PATHS_YAML"
echo "  Train config: $TRAIN_YAML"
echo "  Test config:  $TEST_YAML"
echo "  Model:        $MODEL_NAME"
[[ -n "$RUN_TAG" ]] && echo "  run_tag:      $RUN_TAG"
echo "============================================================"

EXTRA=()
[[ -n "$RUN_TAG" ]] && EXTRA+=(--run-tag "$RUN_TAG")

python3 "$PROJECT_ROOT/scripts/04_visualization.py" \
    -p "$PATHS_YAML" \
    -t "$TRAIN_YAML" \
    -c "$TEST_YAML" \
    -m "$MODEL_NAME" \
    "${EXTRA[@]}"

echo ""
echo "============================================================"
echo "Step 4 COMPLETE."
echo ""
RUN_TAG_DISPLAY="${RUN_TAG:-<from test yaml; defaults to atlas_gt>}"
echo "Artifacts under output_basedir/$MODEL_NAME/runs/$RUN_TAG_DISPLAY/:"
echo "  recon_layout.txt"
echo "  cross_resolution_analysis/  (iso-space pairwise Dice + heatmaps)"
echo "  native_sweep_summary.json   (native_space_step_XX/ audit)"
echo ""
echo "Dice trend / per-class / per-case figures will appear under"
echo "  output_basedir/$MODEL_NAME/viz/<run_tag>/   after the \`compare\` phase."
echo "============================================================"
