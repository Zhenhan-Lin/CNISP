#!/usr/bin/env bash
# Predict a trained nnUNet-C corrector (B=855 or C=845) on the CNISP TEST set.
#
# Assumes the test cases already have degraded CTs + Dataset835 sparse preds +
# canonical-aligned patches from the earlier work_dir/run_pipeline sweep.
#
# Stages:
#   1. (control C, RUN_CNISP=1) CNISP test inference via the existing 032 launcher
#      -> <CNISP_TEST_DIR>/<gtstem>_step{XX}.nii.gz   ({1,2,3,4} native masks)
#   2. install the per-channel resampler into nnunetv2 (ch0 order3, ch1-4 order0)
#   3. build_corrector_testset.py -> nnunet-c/test_input/<name>/imagesTs (5-ch)
#   4. nnUNetv2_predict -d <id> -p nnUNetPlansFinetune -chk <CHK> -f <fold>
#
# Usage:
#   bash nnunet-c/run_corrector_predict.sh C 0
#   RUN_CNISP=0 bash nnunet-c/run_corrector_predict.sh B 0          # B has no CNISP step
#   GPUS="0 1" CHK=checkpoint_best.pth bash nnunet-c/run_corrector_predict.sh C 0
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
CONTROL="${1:?usage: run_corrector_predict.sh <B|C> <fold>}"
FOLD="${2:?usage: run_corrector_predict.sh <B|C> <fold>}"
CONFIG="${CONFIG:-$HERE/configs/corrector.yaml}"
PLAN_NAME="${PLAN_NAME:-nnUNetPlansFinetune}"
# Two DIFFERENT checkpoints (do not conflate):
#   CHK       = nnUNet-C predict checkpoint -> best (the finetuned corrector)
#   CNISP_CHK = CNISP test-inference checkpoint -> latest, to MATCH the training
#               prelabels (those were generated with CNISP 'latest').
CHK="${CHK:-checkpoint_best.pth}"
CNISP_CHK="${CNISP_CHK:-latest}"
# TEST uses CNISP's established ADAPTIVE sweep (adaptive_step_sweep in the test
# yaml) -- NOT the training grid 3/6/9/12 (that was the self-degraded train set).
# The assembler then DISCOVERS whatever (source,step) CNISP produced.
STEPS="${STEPS:-adaptive}"
CASEFILE="${CASEFILE:-test_cases.txt}"
CNISP_TEST_DIR="${CNISP_TEST_DIR:-$HERE/data/cnisp_pred_test}"
RUN_CNISP="${RUN_CNISP:-auto}"        # auto: 1 for control C, 0 otherwise
GPUS="${GPUS:-0 1}"
export nnUNet_compile="${nnUNet_compile:-f}"

echo "================================================================"
echo "[predict] control=$CONTROL fold=$FOLD casefile=$CASEFILE"
echo "[predict] nnUNet-C ckpt=$CHK   CNISP ckpt=$CNISP_CHK   sweep=$STEPS"
echo "================================================================"

eval "$(python3 "$HERE/scripts/corrector_env.py" --config "$CONFIG" --control "$CONTROL")"

if [[ "$EXTERNAL" == "1" ]]; then
    echo "[predict] control $CONTROL is external (Dataset$CTRL_DATASET_ID = pure nnUNet"
    echo "          on the degraded test CTs); predict it with the stock 835 model, e.g.:"
    echo "  nnUNetv2_predict -d $CTRL_DATASET_ID -c $CONFIGURATION -tr $TRAINER -p $REF_PLAN -f $REF_FOLD \\"
    echo "    -i <degraded test CTs> -o <out>"
    exit 0
fi
: "${nnUNet_results:?export nnUNet_results}"

if [[ "$RUN_CNISP" == "auto" ]]; then
    [[ "$PRELABEL_SOURCE" == "cnisp" ]] && RUN_CNISP=1 || RUN_CNISP=0
fi

# ── 1. CNISP test inference (control C) via the existing launcher ─────
if [[ "$RUN_CNISP" == "1" ]]; then
    echo "[predict] (1) CNISP test inference -> $CNISP_TEST_DIR"
    OUT_DIR="$CNISP_TEST_DIR" ALIGNED_DIR="$ALIGNED_DIR" \
    CASEFILE="$CASEFILE" STEPS="$STEPS" MAX_SAMPLES=0 \
    CHECKPOINT="$CNISP_CHK" GPUS="$GPUS" \
        bash "$HERE/run_corrector_cnisp.sh"
else
    echo "[predict] (1) skip CNISP test inference (RUN_CNISP=0)"
fi

# ── 2. install per-channel resampler (predict-time preprocess needs it) ─
echo "[predict] (2) install per-channel resampler into nnunetv2"
python3 - "$HERE/engine/corrector_resampling.py" <<'PY'
import sys, shutil, os
import nnunetv2.preprocessing.resampling as r
dst = os.path.join(os.path.dirname(r.__file__), "corrector_resampling.py")
shutil.copyfile(sys.argv[1], dst)
print(f"[predict] installed resampler -> {dst}")
PY

# ── 3. assemble 5-channel test inputs ────────────────────────────────
echo "[predict] (3) build_corrector_testset -> nnunet-c/test_input"
python3 "$HERE/scripts/build_corrector_testset.py" \
    --config "$CONFIG" --control "$CONTROL" \
    --casefile "$CASEFILE" --steps "${BUILD_STEPS:-auto}" \
    --cnisp-test-dir "$CNISP_TEST_DIR"

IMAGES_TS="$HERE/test_input/$CTRL_DATASET_NAME/imagesTs"
OUT_DIR_PRED="${OUT_DIR_PRED:-$HERE/predictions/$CTRL_DATASET_NAME/fold_${FOLD}}"
mkdir -p "$OUT_DIR_PRED"

# ── 4. nnUNetv2_predict with the finetuned corrector ─────────────────
echo "[predict] (4) nnUNetv2_predict d=$CTRL_DATASET_ID p=$PLAN_NAME chk=$CHK"
echo "          in=$IMAGES_TS"
echo "          out=$OUT_DIR_PRED"
nnUNetv2_predict \
    -i "$IMAGES_TS" -o "$OUT_DIR_PRED" \
    -d "$CTRL_DATASET_ID" -c "$CONFIGURATION" -tr "$TRAINER" \
    -p "$PLAN_NAME" -f "$FOLD" -chk "$CHK"

echo "[predict] done: predictions -> $OUT_DIR_PRED"

# ── 5. shared eval (same code/resample for A/B/C) ────────────────────
MAP_JSON="$HERE/test_input/$CTRL_DATASET_NAME/test_cases_map.json"
EVAL_CSV="${EVAL_CSV:-$HERE/predictions/$CTRL_DATASET_NAME/eval_${CONTROL}_fold${FOLD}.csv}"
if [[ "${RUN_EVAL:-1}" == "1" ]]; then
    echo "[predict] (5) eval (prediction -> native GT grid, order 0; Dice on GT grid)"
    python3 "$HERE/diagnostics/eval_corrector.py" \
        --map "$MAP_JSON" --pred-dir "$OUT_DIR_PRED" --out-csv "$EVAL_CSV"
else
    echo "[predict] (5) skip eval (RUN_EVAL=0). Map: $MAP_JSON"
fi
