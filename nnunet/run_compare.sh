#!/usr/bin/env bash
# ============================================================
# End-to-end driver for the Phase 1 nnUNet vs CNISP comparison.
#
#   1. stage CT inputs                  -> work_dir/nnunet_input/
#   2. nnUNetv2_predict (native)        -> work_dir/nnunet_pred_native/
#   3. CNISP per-step native backfill   -> output_basedir/.../native_space_step_XX/
#      (no-op if infer.py already wrote those dirs; --force to override)
#   4. paired native-space Dice         -> work_dir/paired_*.csv|.txt
#
# Phase 1.5 (SMORE prep) is NOT invoked here; run
#   python nnunet/engine/build_smore_test_images.py --config nnunet/configs.yaml
# separately, ideally in parallel with this driver.
# ============================================================
set -euo pipefail

CONFIG="${CONFIG:-nnunet/configs.yaml}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

cd "$REPO_ROOT"

echo "[run_compare] CONFIG=$CONFIG"
echo "[run_compare] step 1/4: data_prep/prepare_inputs.py"
python3 nnunet/data_prep/prepare_inputs.py --config "$CONFIG"

echo "[run_compare] step 2/4: run_predict_native.sh"
CONFIG="$CONFIG" bash nnunet/run_predict_native.sh

echo "[run_compare] step 3/4: engine/build_cnisp_native_sweep.py (idempotent backfill)"
python3 nnunet/engine/build_cnisp_native_sweep.py --config "$CONFIG"

echo "[run_compare] step 4/4: compare_native.py"
python3 nnunet/compare_native.py --config "$CONFIG"

echo ""
echo "[run_compare] done. Outputs:"
WORK_DIR="$(python3 - <<'PY'
import yaml, os
with open(os.environ.get("CONFIG", "nnunet/configs.yaml")) as f:
    print((yaml.safe_load(f) or {}).get("work_dir", ""))
PY
)"
echo "  ${WORK_DIR}/paired_per_source.csv"
echo "  ${WORK_DIR}/paired_summary.csv"
echo "  ${WORK_DIR}/paired_summary.txt"
