#!/usr/bin/env bash
# ============================================================
# Phase 1b: nnUNetv2_predict on every sparsified CT directory
# staged by nnunet/data_prep/sparsify_inputs.py.
#
# Loops over ${WORK_DIR}/nnunet_input_step_XX/ (each holds the
# 31 sources sparsified along their through-plane axis at
# step_size = XX), one predict run per step. Outputs land at
# ${WORK_DIR}/prediction/sparse_step_XX/ at the sparse CT's
# spacing (nnUNetv2 resamples to plan and back internally).
# nnunet/engine/upsample_sparse_preds.py is the next step; it NN-resamples
# each prediction back to the dense native CT grid so Dice can
# compare against the native GT.
#
# Already-complete step directories are skipped: a step is
# considered complete if every source listed in
# ${WORK_DIR}/nnunet_input_sparse_manifest.json for that step
# already has a prediction file. Partial outputs are recomputed.
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

MANIFEST="${WORK_DIR}/nnunet_input_sparse_manifest.json"
if [[ ! -f "$MANIFEST" ]]; then
    echo "[ERROR] sparse manifest missing: $MANIFEST" >&2
    echo "        run nnunet/data_prep/sparsify_inputs.py first." >&2
    exit 2
fi

mapfile -t STEP_DIRS < <(ls -d "${WORK_DIR}"/nnunet_input_step_*/ 2>/dev/null | sort)
if [[ ${#STEP_DIRS[@]} -eq 0 ]]; then
    echo "[ERROR] no nnunet_input_step_*/ directories under $WORK_DIR" >&2
    echo "        run nnunet/data_prep/sparsify_inputs.py first." >&2
    exit 2
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo "[run_predict_sparse_sweep] dataset=${DATASET_ID} cfg=${CFG} plan=${PLAN} trainer=${TRAINER}"
echo "[run_predict_sparse_sweep] folds=${FOLDS}  GPU=${GPU_ID}"
echo "[run_predict_sparse_sweep] discovered ${#STEP_DIRS[@]} step dir(s)."

for in_dir in "${STEP_DIRS[@]}"; do
    in_dir="${in_dir%/}"
    step_tag="$(basename "$in_dir" | sed 's/^nnunet_input_step_//')"
    out_dir="${WORK_DIR}/prediction/sparse_step_${step_tag}"
    mkdir -p "$out_dir"

    expected_sources=$(python3 - "$MANIFEST" "$step_tag" <<'PY'
import json, sys
manifest_path, step_tag = sys.argv[1], sys.argv[2]
with open(manifest_path) as f:
    m = json.load(f)
print("\n".join(sorted(m.get("by_step", {}).get(step_tag, {}).keys())))
PY
)
    if [[ -z "$expected_sources" ]]; then
        echo "[run_predict_sparse_sweep] step=${step_tag}: no sources in manifest -- skipping."
        continue
    fi

    complete=1
    while IFS= read -r sid; do
        [[ -z "$sid" ]] && continue
        if [[ ! -f "${out_dir}/${sid}.nii.gz" ]]; then
            complete=0
            break
        fi
    done <<< "$expected_sources"
    if [[ $complete -eq 1 ]]; then
        echo "[run_predict_sparse_sweep] step=${step_tag}: all predictions already exist -- skipping."
        continue
    fi

    echo "[run_predict_sparse_sweep] step=${step_tag}: in=${in_dir} out=${out_dir}"
    # shellcheck disable=SC2086
    nnUNetv2_predict \
        -d "${DATASET_ID}" \
        -c "${CFG}" \
        -p "${PLAN}" \
        -tr "${TRAINER}" \
        -f ${FOLDS} \
        -i "${in_dir}" \
        -o "${out_dir}"
done

echo "[run_predict_sparse_sweep] done."
