#!/bin/bash
# ============================================================
# Step 1: Data Preparation
#   - Canonical alignment (checklist + atlas)
#   - Alignment QC (centroid std report)
#   - Train/test split
#
# REVIEW BEFORE PROCEEDING TO STEP 2:
#   1. Check terminal output for alignment QC report
#      - Globe centroid std should be < 3mm (it's the anchor)
#      - ON centroid std is the key metric: < 5mm = good, 5-10mm = marginal
#   2. Visually inspect a few patches in Slicer:
#      cd $ALIGNED_DIR/labels && ls *.nii.gz
#      Open 5-10 patches, verify structures are centered and oriented consistently
#   3. Check train/test split sizes in $CASEFILE_DIR/
# ============================================================

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

PATHS_YAML="$PROJECT_ROOT/configs/paths.yaml"
# TRAIN_YAML="$PROJECT_ROOT/configs/train_default.yaml"
TRAIN_YAML="$PROJECT_ROOT/configs/train_sty2.yaml"

PATCH_SIZE_MM=80.0
TEST_FRACTION=0.2
VAL_FRACTION="${VAL_FRACTION:-0.0}"

# ── Run ───────────────────────────────────────────────────────
echo "============================================================"
echo "Step 1: Data Preparation"
echo "  Project root: $PROJECT_ROOT"
echo "  Paths config: $PATHS_YAML"
echo "  Train config: $TRAIN_YAML"
echo "  Patch size:   ${PATCH_SIZE_MM}mm"
echo "============================================================"

python3 "$PROJECT_ROOT/scripts/01_prepare_data.py" \
    -p "$PATHS_YAML" \
    -c "$TRAIN_YAML" \
    --patch_size "$PATCH_SIZE_MM"

echo ""
echo "============================================================"
echo "Step 1 COMPLETE."
echo ""
echo "REVIEW CHECKLIST before running Step 2:"
echo "  □ Alignment QC report above — check centroid std values"
echo "  □ Open a few patches in Slicer to verify visual quality"
echo "  □ Check train_cases.txt and test_cases.txt sizes"
echo ""
echo "Aligned patches saved to the directory specified in paths.yaml"
echo "============================================================"
