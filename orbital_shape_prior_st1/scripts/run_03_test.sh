#!/bin/bash
# ============================================================
# Step 3: Test A — Controlled Reconstruction
#
# What this does:
#   For each test case:
#     1. Load full-resolution aligned GT label patch
#     2. Synthetically sparsify (keep every Nth slice → simulates anisotropy)
#     3. Use sparse slices as observation → optimize latent code (1200 iters)
#     4. Dense-sample MLP on full-resolution grid → 3D reconstruction
#     5. Compare reconstruction vs full-resolution GT
#
# This answers: "Given PERFECT sparse observations, how much 3D
# information can the shape prior recover?"
#
# REVIEW BEFORE PROCEEDING TO STEP 4:
#   1. Check per-case dice scores in terminal output
#   2. Open a few *_pred.nii.gz files in Slicer alongside GT
#      to visually assess reconstruction quality
#   3. Check if failures are systematic (all ON bad?) or random
#
# OPTIONAL: run at multiple sparsification levels to see degradation:
#   for STEP in 2 4 8; do
#     sed "s/slice_step_size: .*/slice_step_size: $STEP/" \
#       configs/eval_default.yaml > /tmp/eval_step${STEP}.yaml
#     bash scripts/run_03_test.sh /tmp/eval_step${STEP}.yaml
#   done
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
MODEL_NAME="orbital_ad_v2"

# ── Run ───────────────────────────────────────────────────────
echo "============================================================"
echo "Step 3: Test A — Controlled Reconstruction"
echo "  Paths config:  $PATHS_YAML"
echo "  Train config:  $TRAIN_YAML"
echo "  Test config:   $TEST_YAML"
echo "  Model:         $MODEL_NAME"
echo "============================================================"

python3 "$PROJECT_ROOT/scripts/03_infer.py" \
    -p "$PATHS_YAML" \
    -t "$TRAIN_YAML" \
    -c "$TEST_YAML" \
    -m "$MODEL_NAME"

echo ""
echo "============================================================"
echo "Step 3 COMPLETE."
echo ""
echo "Outputs saved to: output_basedir/$MODEL_NAME/"
echo "  *_pred.nii.gz  — reconstructed label maps"
echo "  inference_results.pkl — serialized results for diagnostics"
echo ""
echo "REVIEW CHECKLIST before running Step 4:"
echo "  □ Mean dice > 0.75? (viable shape prior)"
echo "  □ Mean dice > 0.85? (strong shape prior, worth pursuing route 2)"
echo "  □ Which structures fail? (ON expected to be hardest)"
echo "  □ Visual check: do reconstructed shapes look anatomically plausible?"
echo "============================================================"