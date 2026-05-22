#!/usr/bin/env bash
# ============================================================
# Phase 1: nnUNetv2_predict on the original (native-spacing) CT
# inputs staged by nnunet/prepare_inputs.py.
#
# Output is written at the input CT's native spacing -- nnUNetv2
# resamples internally to the iso plan during forward pass and
# resamples back to native before saving. For the iso-grid
# comparison (Phase 2) re-run this on SMORE'd inputs instead.
# ============================================================
set -euo pipefail

CONFIG="${CONFIG:-nnunet/configs.yaml}"

# Tolerant YAML peek -- only needs scalar keys.
_yaml_get() {
    python3 - "$CONFIG" "$1" <<'PY'
import sys, yaml
cfg_path, key = sys.argv[1], sys.argv[2]
with open(cfg_path) as f:
    cfg = yaml.safe_load(f) or {}
val = cfg.get(key)
if isinstance(val, list):
    print(" ".join(str(x) for x in val))
else:
    print("" if val is None else val)
PY
}

DATASET_ID="${DATASET_ID:-$(_yaml_get dataset_id)}"
CFG="${CFG_NAME:-$(_yaml_get configuration)}"
PLAN="${PLAN:-$(_yaml_get plan)}"
TRAINER="${TRAINER:-$(_yaml_get trainer)}"
FOLDS="${FOLDS:-$(_yaml_get folds)}"
GPU_ID="${GPU_ID:-$(_yaml_get gpu_id)}"
WORK_DIR="${WORK_DIR:-$(_yaml_get work_dir)}"

if [[ -z "$WORK_DIR" ]]; then
    echo "[ERROR] WORK_DIR not set (config key: work_dir)" >&2
    exit 2
fi

IN_DIR="${IN_DIR:-${WORK_DIR}/nnunet_input}"
OUT_DIR="${OUT_DIR:-${WORK_DIR}/nnunet_pred_native}"

if [[ ! -d "$IN_DIR" ]]; then
    echo "[ERROR] input dir not found: $IN_DIR (did you run prepare_inputs.py?)" >&2
    exit 2
fi

mkdir -p "$OUT_DIR"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo "[run_predict_native] dataset=${DATASET_ID} cfg=${CFG} plan=${PLAN} trainer=${TRAINER}"
echo "[run_predict_native] folds=${FOLDS}  GPU=${GPU_ID}"
echo "[run_predict_native] in:  ${IN_DIR}"
echo "[run_predict_native] out: ${OUT_DIR}"

# shellcheck disable=SC2086
nnUNetv2_predict \
    -d "${DATASET_ID}" \
    -c "${CFG}" \
    -p "${PLAN}" \
    -tr "${TRAINER}" \
    -f ${FOLDS} \
    -i "${IN_DIR}" \
    -o "${OUT_DIR}"

echo "[run_predict_native] done. predictions: ${OUT_DIR}"
