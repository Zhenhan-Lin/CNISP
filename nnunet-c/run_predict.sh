#!/usr/bin/env bash
# Run the corrector inference cascade for a control on the test set:
#   (Stages 1 & 3 must have produced the degraded CTs + prelabels already)
#   1. assemble 5-channel imagesTs (predict_cascade.py)
#   2. nnUNetv2_predict with the control's finetuned model
#
# Control A (external Dataset835) predicts the single-channel degraded CT with
# the original 835 model (no prelabel channels).
#
# Usage:
#   bash nnunet-c/run_predict.sh C /path/out_preds            # test split
#   OUT_IMAGES=/tmp/imgs bash nnunet-c/run_predict.sh B /path/out_preds
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
CONTROL="${1:?usage: run_predict.sh <A|B|C> <out_pred_dir> [split]}"
OUT_PRED="${2:?usage: run_predict.sh <A|B|C> <out_pred_dir> [split]}"
SPLIT="${3:-test}"
CONFIG="${CONFIG:-$HERE/configs/corrector.yaml}"
PLAN_NAME="${PLAN_NAME:-nnUNetPlansFinetune}"
CHECKPOINT="${CHECKPOINT:-checkpoint_final.pth}"
OUT_IMAGES="${OUT_IMAGES:-$HERE/staging/_predict/${CONTROL}_${SPLIT}_imagesTs}"

eval "$(python3 "$HERE/scripts/corrector_env.py" --config "$CONFIG" --control "$CONTROL")"

echo "[run_predict] control=$CONTROL split=$SPLIT dataset=$CTRL_DATASET_ID"
echo "[run_predict] (1) assemble imagesTs -> $OUT_IMAGES"
python3 "$HERE/scripts/predict_cascade.py" \
    --config "$CONFIG" --control "$CONTROL" --split "$SPLIT" \
    --out-images-dir "$OUT_IMAGES"

# Control A uses the original 835 plan; B/C use the merged finetune plan.
PRED_PLAN="$PLAN_NAME"
PRED_FOLD="${FOLD:-0}"
if [[ "$EXTERNAL" == "1" ]]; then
    PRED_PLAN="$REF_PLAN"
    PRED_FOLD="${FOLD:-$REF_FOLD}"
fi

echo "[run_predict] (2) nnUNetv2_predict d=$CTRL_DATASET_ID plan=$PRED_PLAN fold=$PRED_FOLD"
nnUNetv2_predict \
    -i "$OUT_IMAGES" \
    -o "$OUT_PRED" \
    -d "$CTRL_DATASET_ID" \
    -c "$CONFIGURATION" \
    -p "$PRED_PLAN" \
    -tr "$TRAINER" \
    -f "$PRED_FOLD" \
    -chk "$CHECKPOINT"

echo "[run_predict] done -> $OUT_PRED"
