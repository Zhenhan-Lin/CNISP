#!/usr/bin/env bash
# ============================================================
# Phase 1c: nnUNetv2_predict on the SMORE-super-resolved CTs
# staged by nnunet/data_prep/prepare_smore_inputs.py.
#
# Input  : ${WORK_DIR}/nnunet_input_smore/<sid>_0000.nii.gz
# Output : ${WORK_DIR}/prediction/smore/<sid>.nii.gz
#
# Mask only. No upsampling, no compare wiring -- downstream analysis is
# TBD; this phase exists so the SMORE-grid prediction is always on hand
# when we decide what to do with it.
# ============================================================
set -euo pipefail

CONFIG="${CONFIG:-nnunet/configs.yaml}"

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

IN_DIR="${IN_DIR:-${WORK_DIR}/nnunet_input_smore}"
OUT_DIR="${OUT_DIR:-${WORK_DIR}/prediction/smore}"

if [[ ! -d "$IN_DIR" ]]; then
    echo "[ERROR] input dir not found: $IN_DIR" >&2
    echo "        run nnunet/data_prep/prepare_smore_inputs.py first." >&2
    exit 2
fi

mkdir -p "$OUT_DIR"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo "[run_predict_smore] dataset=${DATASET_ID} cfg=${CFG} plan=${PLAN} trainer=${TRAINER}"
echo "[run_predict_smore] folds=${FOLDS}  GPU=${GPU_ID}"
echo "[run_predict_smore] in:  ${IN_DIR}"
echo "[run_predict_smore] out: ${OUT_DIR}"

# shellcheck disable=SC2086
nnUNetv2_predict \
    -d "${DATASET_ID}" \
    -c "${CFG}" \
    -p "${PLAN}" \
    -tr "${TRAINER}" \
    -f ${FOLDS} \
    -i "${IN_DIR}" \
    -o "${OUT_DIR}"

echo "[run_predict_smore] done. predictions: ${OUT_DIR}"
