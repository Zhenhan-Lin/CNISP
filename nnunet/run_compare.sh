#!/usr/bin/env bash
# ============================================================
# End-to-end driver for the Phase 1 nnUNet vs CNISP comparison.
#
# Runs the comparison ONCE for the given CNISP run_tag (default
# atlas_gt = ceiling curve). For the full Option C two-run report
# (ceiling + deployment), use ../run_pipeline.sh instead -- it
# orchestrates both stories plus their prereq phases.
#
#   1. stage CT inputs                  -> work_dir/nnunet_input/
#   2. nnUNetv2_predict (native)        -> work_dir/prediction/native/
#   3. CNISP per-step native backfill   -> output_basedir/.../runs/<tag>/native_space_step_XX/
#      (no-op if infer.py already wrote those dirs; --force to override)
#   4. paired native-space Dice         -> work_dir/comparison/paired_*__<tag>.csv|.txt
#
# Phase 1.5 (SMORE prep) is NOT invoked here; run
#   python nnunet/engine/build_smore_test_images.py --config nnunet/configs.yaml
# separately, ideally in parallel with this driver.
#
# Usage:
#   bash nnunet/run_compare.sh                    # run_tag=atlas_gt
#   bash nnunet/run_compare.sh nnunet_pred        # deployment-curve run
# ============================================================
set -euo pipefail

CONFIG="${CONFIG:-nnunet/configs.yaml}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
RUN_TAG="${1:-atlas_gt}"

cd "$REPO_ROOT"

echo "[run_compare] CONFIG=$CONFIG  RUN_TAG=$RUN_TAG"
echo "[run_compare] step 1/4: data_prep/prepare_inputs.py"
python3 nnunet/data_prep/prepare_inputs.py --config "$CONFIG"

echo "[run_compare] step 2/4: run_predict_native.sh"
CONFIG="$CONFIG" bash nnunet/run_predict_native.sh

echo "[run_compare] step 3/4: engine/build_cnisp_native_sweep.py (idempotent backfill)"
python3 nnunet/engine/build_cnisp_native_sweep.py --config "$CONFIG" --run-tag "$RUN_TAG"

echo "[run_compare] step 4/4: compare_native.py"
python3 nnunet/compare_native.py --config "$CONFIG" --cnisp-run-tag "$RUN_TAG"

echo ""
echo "[run_compare] done. Outputs:"
WORK_DIR="$(python3 - <<'PY'
import yaml, os
with open(os.environ.get("CONFIG", "nnunet/configs.yaml")) as f:
    print((yaml.safe_load(f) or {}).get("work_dir", ""))
PY
)"
echo "  ${WORK_DIR}/comparison/paired_per_source__${RUN_TAG}.csv"
echo "  ${WORK_DIR}/comparison/paired_summary__${RUN_TAG}.csv"
echo "  ${WORK_DIR}/comparison/paired_summary__${RUN_TAG}.txt"
