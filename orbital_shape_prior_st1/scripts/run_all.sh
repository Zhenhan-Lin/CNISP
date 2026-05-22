#!/bin/bash
# ============================================================
# Full Pipeline: Steps 1 → 2 → 3 → 4
#
# Pauses between steps for manual review.
# Alternatively, run each step individually:
#   bash scripts/run_01_prepare.sh
#   bash scripts/run_02_train.sh        (long — hours/days)
#   bash scripts/run_03_test.sh
#   bash scripts/run_04_visualization.sh
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================================"
echo "Orbital Shape Prior — Full Pipeline"
echo "============================================================"

# ── Step 1 ────────────────────────────────────────────────────
bash "$SCRIPT_DIR/run_01_prepare.sh"

echo ""
read -p "Review Step 1 outputs. Continue to training? [y/N] " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Stopped after Step 1."
    exit 0
fi

# ── Step 2 ────────────────────────────────────────────────────
bash "$SCRIPT_DIR/run_02_train.sh"

echo ""
read -p "Review training curves. Continue to testing? [y/N] " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Stopped after Step 2."
    exit 0
fi

# ── Step 3 ────────────────────────────────────────────────────
bash "$SCRIPT_DIR/run_03_test.sh"

echo ""
read -p "Review reconstruction results. Build visualization summary? [y/N] " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Stopped after Step 3."
    exit 0
fi

# ── Step 4 ────────────────────────────────────────────────────
bash "$SCRIPT_DIR/run_04_visualization.sh"

echo ""
echo "============================================================"
echo "Pipeline complete. Review recon_summary.png and the"
echo "cross_resolution_analysis/ + native_sweep_summary.json outputs."
echo "============================================================"
