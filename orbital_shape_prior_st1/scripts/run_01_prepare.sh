#!/bin/bash
# ============================================================
# Step 1: Data Preparation
#   - Canonical alignment (checklist + atlas)
#   - Train/val/test split (by patient source_id)
#
# REVIEW BEFORE PROCEEDING TO STEP 2:
#   1. Visually inspect a few patches in Slicer:
#      cd $ALIGNED_DIR/labels && ls *.nii.gz
#      Open 5-10 patches, verify structures are centered and oriented consistently
#   2. Check train/val/test split sizes in $CASEFILE_DIR/
#   3. (Optional) regenerate alignment QC report with
#      data_prep.alignment_qc.compute_alignment_stats / print_alignment_report
# ============================================================

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$REPO_ROOT:${PYTHONPATH:-}"

PATHS_YAML="$PROJECT_ROOT/configs/paths.yaml"
# TRAIN_YAML="$PROJECT_ROOT/configs/train_default.yaml"
TRAIN_YAML="$PROJECT_ROOT/configs/train_sty2.yaml"

PATCH_SIZE_MM=80.0
# Split fractions (test_fraction, val_fraction, atlas_to_test) are read
# from $TRAIN_YAML by 01_prepare_data.py — edit there to change them.

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
echo "  □ Open a few patches in Slicer to verify visual quality"
echo "  □ Check train_cases.txt / val_cases.txt / test_cases.txt sizes"
echo "  □ (Optional) run alignment_qc.compute_alignment_stats for centroid std"
echo ""
echo "Aligned patches saved to the directory specified in paths.yaml"
echo "============================================================"
