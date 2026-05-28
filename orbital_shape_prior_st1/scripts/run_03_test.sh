#!/bin/bash
# ============================================================
# Step 3: Test A — Controlled Reconstruction
#
# What this does:
#   For each test case, the adaptive resolution sweep:
#     1. Load full-resolution aligned GT label patch
#     2. Compute per-case step list from configs/test_default.yaml
#        adaptive_step_sweep (steps depend on the case's through-plane
#        spacing so eff_res stays under max_eff_resolution_mm)
#     3. For each step: sparsify, optimize latent (≤1200 iters),
#        dense-sample MLP, compute Dice (dense + observed-only)
#     4. Pick per-case "primary" result closest to primary_eff_res_mm
#        and map it back to native image space
#
# This answers: "Given PERFECT sparse observations at varying acquisition
# anisotropy, how much 3D structure can the shape prior recover?"
#
# REVIEW BEFORE PROCEEDING TO STEP 4:
#   1. Check per-case dice scores in stdout + test_results.csv
#   2. Open a few step_XX/pred/*_pred.nii.gz in Slicer alongside GT
#   3. Check if failures are systematic (all ON bad?) or random
#   4. Look at primary native_space/ outputs for the down-stream pipeline
# ============================================================

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=1

PATHS_YAML="$PROJECT_ROOT/configs/paths.yaml"
# TRAIN_YAML="$PROJECT_ROOT/configs/train_default.yaml"
TRAIN_YAML="$PROJECT_ROOT/configs/train_sty2.yaml"
TEST_YAML="${1:-$PROJECT_ROOT/configs/test_default.yaml}"

# Read model_name from train config (or override here)
MODEL_NAME="orbital_ad_v4"

# Option C overrides (optional). Either pass on the command line or set
# the corresponding fields in the test yaml. When unset, the test yaml
# wins; the yaml's own default keeps the ceiling-curve behaviour.
#
#   ./run_03_test.sh [test_yaml] [test_label_source] [run_tag]
TEST_LABEL_SOURCE="${2:-}"
RUN_TAG="${3:-}"

# ── Run ───────────────────────────────────────────────────────
echo "============================================================"
echo "Step 3: Test A — Controlled Reconstruction"
echo "  Paths config:        $PATHS_YAML"
echo "  Train config:        $TRAIN_YAML"
echo "  Test config:         $TEST_YAML"
echo "  Model:               $MODEL_NAME"
[[ -n "$TEST_LABEL_SOURCE" ]] && echo "  test_label_source:   $TEST_LABEL_SOURCE"
[[ -n "$RUN_TAG"           ]] && echo "  run_tag:             $RUN_TAG"
echo "============================================================"

EXTRA=()
[[ -n "$TEST_LABEL_SOURCE" ]] && EXTRA+=(--test-label-source "$TEST_LABEL_SOURCE")
[[ -n "$RUN_TAG"           ]] && EXTRA+=(--run-tag "$RUN_TAG")

python3 "$PROJECT_ROOT/scripts/03_infer.py" \
    -p "$PATHS_YAML" \
    -t "$TRAIN_YAML" \
    -c "$TEST_YAML" \
    -m "$MODEL_NAME" \
    "${EXTRA[@]}"

echo ""
echo "============================================================"
echo "Step 3 COMPLETE."
echo ""
RUN_TAG_DISPLAY="${RUN_TAG:-<from test yaml; defaults to atlas_gt>}"
echo "Outputs saved to: output_basedir/$MODEL_NAME/runs/$RUN_TAG_DISPLAY/"
echo "  inference_results.pkl       — primary picks (one per case), feeds map_to_native + 04_visualization"
echo "  sweep_results.pkl           — full (case × step) sweep"
echo "  test_results.csv            — per-(case, step) metrics (with eff_res_bucket column)"
echo "  step_XX/pred/*.nii.gz       — reconstructed label maps (per step)"
echo "  step_XX/latents/*.npy       — optimized latents (cache resume)"
echo "  step_XX/iso_space/          — isotropic-grid predictions (for cross-resolution heatmap)"
echo "  native_space/               — primary picks mapped back to native image grid"
echo "  native_space_step_XX/       — every step mapped to native space + manifest.json"
echo "  native_sweep_manifest.json  — top-level index over per-step manifests"
echo ""
echo "REVIEW CHECKLIST before running Step 4:"
echo "  □ Mean dice > 0.75? (viable shape prior)"
echo "  □ Mean dice > 0.85? (strong shape prior, worth pursuing route 2)"
echo "  □ Which structures fail? (ON expected to be hardest)"
echo "  □ Visual check: do reconstructed shapes look anatomically plausible?"
echo "============================================================"