#!/bin/bash
# ============================================================
# Step 2: Train Shape Prior
#
# MONITORING DURING TRAINING:
#   In a separate terminal:
#     tensorboard --logdir=$MODEL_DIR --port=6006
#   Then open http://localhost:6006
#   Watch:
#     - loss/train should decrease steadily
#     - dice/train should increase toward 0.85+
#     - dice/val should track train (gap = overfitting)
#     - lat_norm2 should stabilize (not explode)
#
# REVIEW BEFORE PROCEEDING TO STEP 3:
#   1. Training converged? (loss plateau, dice plateau)
#   2. Train/val dice gap < 0.05? (no severe overfitting)
#   3. Final train dice > 0.80? (model has enough capacity)
#   If train dice is low, consider:
#     - Increasing num_epochs
#     - Adjusting dice_class_weights to upweight ON/Recti
#     - Checking alignment quality (go back to Step 1)
# ============================================================

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

PATHS_YAML="$PROJECT_ROOT/configs/paths.yaml"
# TRAIN_YAML="$PROJECT_ROOT/configs/train_default.yaml"
TRAIN_YAML="$PROJECT_ROOT/configs/train_sty2.yaml"

# GPU selection (change as needed)
export CUDA_VISIBLE_DEVICES=1

# ── Run ───────────────────────────────────────────────────────
echo "============================================================"
echo "Step 2: Train Shape Prior"
echo "  Paths config:    $PATHS_YAML"
echo "  Training config: $TRAIN_YAML"
echo "  GPU:             CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "============================================================"

python3 "$PROJECT_ROOT/scripts/02_train.py" \
    -p "$PATHS_YAML" \
    -c "$TRAIN_YAML"

echo ""
echo "============================================================"
echo "Step 2 COMPLETE."
echo ""
echo "REVIEW CHECKLIST before running Step 3:"
echo "  □ Check tensorboard: did loss converge?"
echo "  □ Train dice > 0.80?"
echo "  □ Train-val dice gap < 0.05?"
echo "  □ Latent norms stable (not diverging)?"
echo "============================================================"
