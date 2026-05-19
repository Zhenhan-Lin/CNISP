#!/bin/bash
# ============================================================
# Step 4: Diagnostics & Report
#
# Runs:
#   1. Reconstruction QC:
#      - Per-structure dice (unaligned and centroid-aligned)
#      - Position contribution = aligned_dice - unaligned_dice
#        → tells you how much error is due to position vs shape
#      - Centroid shift (mm) between prediction and GT
#      - Volume ratio (pred/GT) per structure
#
#   2. Latent space interpretability:
#      - PCA of test-time optimized latents
#      - Correlation with anatomical metrics (volume, centroid)
#
# HOW TO INTERPRET THE RESULTS:
#
#   Position contribution > 0.10:
#     Canonical alignment is insufficient. Shape prior is spending
#     capacity on position rather than shape. Next step: improve
#     alignment (PCA rotation? stronger anchor?) or add Jansen-style
#     test-time pose optimization.
#
#   Position contribution < 0.05 AND dice > 0.85:
#     Shape prior works well. Proceed to Route 2 (embed into nnUNet).
#
#   Position contribution < 0.05 AND dice < 0.75:
#     Shape is the bottleneck, not position. Consider:
#     - More training data
#     - Class-balanced sampling (upweight ON/Recti)
#     - Larger latent dim or more layers
#
#   Volume ratio consistently != 1.0:
#     Systematic over/under-segmentation. Check if specific structures
#     are biased (e.g., ON always under-segmented due to thin geometry).
# ============================================================

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

PATHS_YAML="$PROJECT_ROOT/configs/paths.yaml"
# TRAIN_YAML="$PROJECT_ROOT/configs/train_default.yaml"
TRAIN_YAML="$PROJECT_ROOT/configs/train_sty2.yaml"
TEST_YAML="${1:-$PROJECT_ROOT/configs/test_default.yaml}"
MODEL_NAME="orbital_ad_v2"

# ── Run ───────────────────────────────────────────────────────
echo "============================================================"
echo "Step 4: Diagnostics"
echo "  Model: $MODEL_NAME"
echo "============================================================"

python3 "$PROJECT_ROOT/scripts/04_diagnose.py" \
    -p "$PATHS_YAML" \
    -c "$TRAIN_YAML" \
    -t "$TEST_YAML" \
    -m "$MODEL_NAME"

echo ""
echo "============================================================"
echo "Step 4 COMPLETE."
echo ""
echo "Reports saved to: output_basedir/$MODEL_NAME/"
echo "  diagnostic_report.json       — reconstruction QC + latent analysis"
echo "  cross_resolution_analysis/   — heatmaps + pairwise Dice CSV"
echo "============================================================"
