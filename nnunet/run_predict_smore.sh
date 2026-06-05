#!/usr/bin/env bash
# ============================================================
# Phase 1c: nnUNetv2_predict on the SMORE-super-resolved CTs
# staged by nnunet/data_prep/prepare_smore_inputs.py.
#
# Input  : ${WORK_DIR}/input/smore/<sid>_0000.nii.gz
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

# Resolve folds (best|all|list) against trained checkpoints.
_resolve_folds() {
    python3 - "$CONFIG" <<'PY'
import json, os, sys, yaml
from pathlib import Path
cfg = yaml.safe_load(open(sys.argv[1])) or {}
mf = (Path(os.environ["nnUNet_results"])
      / f"Dataset{int(cfg['dataset_id']):03d}_{cfg.get('dataset_name','')}"
      / f"{cfg.get('trainer','nnUNetTrainer')}__{cfg.get('plan','nnUNetPlans')}__{cfg.get('configuration','3d_fullres')}")
ckpt = cfg.get("checkpoint_name", "checkpoint_final.pth")
avail = sorted(int(d.name[5:]) for d in mf.glob("fold_*") if (d/ckpt).is_file())
if not avail:
    sys.exit(f"[folds] no trained fold under {mf}")
f = cfg.get("folds", [0])
if f == "all":
    sel = avail
elif f == "best":
    sel = [max(avail, key=lambda k: (json.loads((mf/f"fold_{k}"/"validation"/"summary.json").read_text())["foreground_mean"]["Dice"] if (mf/f"fold_{k}"/"validation"/"summary.json").is_file() else -1))]
else:
    f = [f] if isinstance(f, int) else f
    sel = [int(x) for x in f if int(x) in avail]
    if not sel:
        sys.exit(f"[folds] requested {f} not trained; available {avail}")
print(" ".join(map(str, sel)))
PY
}

DATASET_ID="${DATASET_ID:-$(_yaml_get dataset_id)}"
CFG="${CFG_NAME:-$(_yaml_get configuration)}"
PLAN="${PLAN:-$(_yaml_get plan)}"
TRAINER="${TRAINER:-$(_yaml_get trainer)}"
FOLDS="${FOLDS:-$(_resolve_folds)}"
GPU_ID="${GPU_ID:-$(_yaml_get gpu_id)}"
WORK_DIR="${WORK_DIR:-$(_yaml_get work_dir)}"

if [[ -z "$WORK_DIR" ]]; then
    echo "[ERROR] WORK_DIR not set (config key: work_dir)" >&2
    exit 2
fi

IN_DIR="${IN_DIR:-${WORK_DIR}/input/smore}"
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
