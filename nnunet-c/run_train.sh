#!/usr/bin/env bash
# Finetune a corrector control (B=855 or C=845) from the Dataset835 weights.
#
# Pipeline (GPU box; needs nnunetv2 + $nnUNet_raw/$nnUNet_preprocessed/$nnUNet_results):
#   1. nnUNetv2_extract_fingerprint  -d <id>
#   2. nnUNetv2_plan_experiment      -d <id>                 # valid 5-ch nnUNetPlans
#   3. build_finetune_plan.py        --control X             # potholes 1 & 3 -> nnUNetPlansFinetune
#   4. nnUNetv2_preprocess           -d <id> -plans_name nnUNetPlansFinetune -c <cfg>
#   5. check_preprocessed.py         --control X             # POTHOLE-4 HARD GATE
#   6. adapt_checkpoint.py           (835 ckpt 1ch -> 5ch)   # first-conv surgery
#   7. nnUNetv2_train <id> <cfg> <fold> -p nnUNetPlansFinetune -pretrained_weights <adapted>
#
# Usage:
#   bash nnunet-c/run_train.sh B 0
#   MASK_INIT=small_random bash nnunet-c/run_train.sh C 0
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
CONTROL="${1:?usage: run_train.sh <B|C> <fold>}"
FOLD="${2:?usage: run_train.sh <B|C> <fold>}"
CONFIG="${CONFIG:-$HERE/configs/corrector.yaml}"
PLAN_NAME="${PLAN_NAME:-nnUNetPlansFinetune}"
MASK_INIT="${MASK_INIT:-zero}"
WORK_TMP="${WORK_TMP:-$HERE/staging/_finetune}"
# torch.compile (nnUNet default ON) can produce broken forward passes on some
# torch/CUDA combos -> all-background preds -> pseudo-dice stuck at 0. Default OFF;
# export nnUNet_compile=1 to re-enable once you've confirmed it's stable.
export nnUNet_compile="${nnUNet_compile:-f}"

echo "================================================================"
echo "[run_train] control=$CONTROL fold=$FOLD config=$CONFIG"
echo "================================================================"

eval "$(python3 "$HERE/scripts/corrector_env.py" --config "$CONFIG" --control "$CONTROL")"

if [[ "$EXTERNAL" == "1" ]]; then
    echo "[run_train] control $CONTROL is external (Dataset$CTRL_DATASET_ID); nothing to train."
    exit 0
fi
: "${nnUNet_preprocessed:?export nnUNet_preprocessed}"
: "${nnUNet_results:?export nnUNet_results}"
if [[ -z "$REF_CKPT" || ! -f "$REF_CKPT" ]]; then
    echo "[run_train] ERROR: Dataset835 checkpoint not found: '$REF_CKPT'" >&2
    echo "            confirm reference_plan/reference_fold in $CONFIG and \$nnUNet_results." >&2
    exit 1
fi

# SKIP_PREPROCESS=1: dataset/plan/preprocess already done -> jump straight to the
# gate + surgery + train (e.g. to re-run training with nnUNet_compile=f without
# wiping & regenerating the preprocessed data, which nnUNetv2_preprocess does).
if [[ "${SKIP_PREPROCESS:-0}" == "1" ]]; then
    echo "[run_train] SKIP_PREPROCESS=1 -> skipping fingerprint/plan/merge/preprocess"
else
    echo "[run_train] (1) extract_fingerprint d=$CTRL_DATASET_ID"
    nnUNetv2_extract_fingerprint -d "$CTRL_DATASET_ID" --verify_dataset_integrity

    echo "[run_train] (2) plan_experiment d=$CTRL_DATASET_ID"
    nnUNetv2_plan_experiment -d "$CTRL_DATASET_ID"

    echo "[run_train] (3) build_finetune_plan (potholes 1 & 3 + per-channel resampler) -> $PLAN_NAME"
    python3 "$HERE/scripts/build_finetune_plan.py" \
        --config "$CONFIG" --control "$CONTROL" --out-plan-name "$PLAN_NAME"

    echo "[run_train] (3b) install per-channel resampler into nnunetv2 (ch0 order3, ch1-N order0)"
    python3 - "$HERE/engine/corrector_resampling.py" <<'PY'
import sys, shutil, os
import nnunetv2.preprocessing.resampling as r
dst = os.path.join(os.path.dirname(r.__file__), "corrector_resampling.py")
shutil.copyfile(sys.argv[1], dst)
print(f"[run_train] installed resampler -> {dst}")
PY

    echo "[run_train] (4) preprocess with merged plan $PLAN_NAME"
    nnUNetv2_preprocess -d "$CTRL_DATASET_ID" -plans_name "$PLAN_NAME" -c "$CONFIGURATION"
fi

echo "[run_train] (5) POTHOLE-4 GATE: check_preprocessed"
python3 "$HERE/diagnostics/check_preprocessed.py" \
    --config "$CONFIG" --control "$CONTROL" --plan-name "$PLAN_NAME"

mkdir -p "$WORK_TMP"
ADAPTED="$WORK_TMP/ckpt_${REF_DATASET_ID}_to${N_CHANNELS}ch_${CONTROL}.pth"
echo "[run_train] (6) first-conv surgery 1ch->${N_CHANNELS}ch (mask_init=$MASK_INIT)"
python3 "$HERE/scripts/adapt_checkpoint.py" \
    --in "$REF_CKPT" --out "$ADAPTED" \
    --channels "$N_CHANNELS" --mask-init "$MASK_INIT" \
    --report-json "$WORK_TMP/adapt_report_${CONTROL}.json"

# Install the corrector finetune trainer into nnunetv2 so `-tr` can discover it.
echo "[run_train] (6b) install corrector trainer -> nnunetv2 ($CORRECTOR_TRAINER)"
python3 - "$HERE/engine/nnUNetTrainer_corrector.py" <<'PY'
import sys, shutil, os
import nnunetv2.training.nnUNetTrainer.nnUNetTrainer as m
dst = os.path.join(os.path.dirname(m.__file__), "nnUNetTrainer_corrector.py")
shutil.copyfile(sys.argv[1], dst)
print(f"[run_train] installed trainer -> {dst}")
PY

# The corrector trainer reads its schedule from these (corrector.yaml::finetune).
export CORRECTOR_EPOCHS CORRECTOR_LR

echo "[run_train] (7) nnUNetv2_train $CTRL_DATASET_ID $CONFIGURATION $FOLD -p $PLAN_NAME"
echo "          trainer=$CORRECTOR_TRAINER  epochs=$CORRECTOR_EPOCHS  initial_lr=$CORRECTOR_LR"
nnUNetv2_train "$CTRL_DATASET_ID" "$CONFIGURATION" "$FOLD" \
    -p "$PLAN_NAME" -tr "$CORRECTOR_TRAINER" -pretrained_weights "$ADAPTED"

echo "[run_train] done: Dataset${CTRL_DATASET_ID} fold $FOLD finetuned from $REF_CKPT"
