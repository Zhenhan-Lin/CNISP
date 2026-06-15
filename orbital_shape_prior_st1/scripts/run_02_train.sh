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
REPO_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$REPO_ROOT:${PYTHONPATH:-}"

PATHS_YAML="$PROJECT_ROOT/configs/paths.yaml"
# Train config: $1 overrides (absolute path, or a name/relative path resolved
# under configs/). Default is the v6 config train_sty2.yaml. Use
# configs/train_v5_5.yaml or configs/train_v6_5.yaml for the intermediate runs.
TRAIN_YAML="${1:-$PROJECT_ROOT/configs/train_sty2.yaml}"
if [[ ! -f "$TRAIN_YAML" ]]; then
    # Allow passing just a filename or a path relative to configs/.
    if [[ -f "$PROJECT_ROOT/configs/$TRAIN_YAML" ]]; then
        TRAIN_YAML="$PROJECT_ROOT/configs/$TRAIN_YAML"
    else
        echo "[run_02_train] train config not found: $TRAIN_YAML" >&2
        exit 2
    fi
fi

# GPU selection. Respect a GPU already chosen by a parent (e.g.
# run_pipeline.sh --gpu exports CUDA_VISIBLE_DEVICES); fall back to GPU 1
# only when invoked standalone with nothing set.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

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
