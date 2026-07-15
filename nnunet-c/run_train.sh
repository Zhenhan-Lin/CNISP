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
# CASCADE=1 -> native-cascade (Route A) layout: 1-ch CT data + a per-case CNISP
# seg_prev. The multi-step data prep (2 datasets + relocate_prevseg) lives in
# CNISP/RUNBOOK.md; this script then reuses that preprocessed data
# (SKIP_PREPROCESS=1) and only adds --cascade to the gate. Set CORRECTOR_TRAINER=
# nnUNetTrainer_OrbitalCascade (corrector.yaml or env) to train the overhaul.
CASCADE="${CASCADE:-1}"          # default: native-cascade + OrbitalCascade aug (B AND C); CASCADE=0 = legacy stacked
GATE_ARGS=""; [[ "$CASCADE" == "1" ]] && GATE_ARGS="--cascade"
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

# The per-channel resampler (ch0 order3, ch1-N order0) must be installed for BOTH
# preprocessing (writes ch1-4 as {0,1}) AND training/validation (nnUNet's
# end-of-training predict imports the plan's resampling_fn_data). It's a cheap,
# idempotent file copy into nnunetv2 site-packages, so install it UNCONDITIONALLY
# here -- otherwise a SKIP_PREPROCESS run on a box where it was never installed
# crashes at validation with an import error.
echo "[run_train] (0) install per-channel resampler into nnunetv2 (ch0 order3, ch1-N order0)"
python3 - "$HERE/engine/corrector_resampling.py" <<'PY'
import sys, shutil, os
import nnunetv2.preprocessing.resampling as r
dst = os.path.join(os.path.dirname(r.__file__), "corrector_resampling.py")
shutil.copyfile(sys.argv[1], dst)
print(f"[run_train] installed resampler -> {dst}")
PY

# SKIP_PREPROCESS defaults to 1 (SKIP): the corrector's preprocess is the slow,
# OOM-prone step, and it only needs to succeed ONCE (nnUNetv2_preprocess has no
# resume, so re-running redoes ALL cases). So by default we jump straight to the
# gate + surgery + train and REUSE the existing preprocessed data. The gate below
# (check_preprocessed) runs unconditionally and fails loudly if preprocess is
# missing/incomplete. To (re)build the preprocessed data, run SKIP_PREPROCESS=0.
if [[ "${SKIP_PREPROCESS:-1}" == "1" ]]; then
    echo "[run_train] SKIP_PREPROCESS=1 (default) -> skipping fingerprint/plan/merge/preprocess"
    echo "[run_train]   (reusing existing preprocessed data; set SKIP_PREPROCESS=0 to (re)build)"
elif [[ "$CASCADE" == "1" ]]; then
    echo "[run_train] ERROR: CASCADE=1 with SKIP_PREPROCESS=0 is not supported here." >&2
    echo "            Cascade prep builds TWO datasets (main + prior) + relocates the" >&2
    echo "            seg_prev -- run CNISP/RUNBOOK.md, then re-run this with" >&2
    echo "            SKIP_PREPROCESS=1 (default) to reuse that preprocessed data." >&2
    exit 2
else
    echo "[run_train] (1) extract_fingerprint d=$CTRL_DATASET_ID"
    nnUNetv2_extract_fingerprint -d "$CTRL_DATASET_ID" --verify_dataset_integrity

    echo "[run_train] (2) plan_experiment d=$CTRL_DATASET_ID"
    nnUNetv2_plan_experiment -d "$CTRL_DATASET_ID"

    echo "[run_train] (3) build_finetune_plan (potholes 1 & 3 + per-channel resampler) -> $PLAN_NAME"
    python3 "$HERE/scripts/build_finetune_plan.py" \
        --config "$CONFIG" --control "$CONTROL" --out-plan-name "$PLAN_NAME"

    echo "[run_train] (4) preprocess with merged plan $PLAN_NAME"
    # PREPROCESS_NP caps the preprocessing worker count. The corrector cases are
    # 5-channel volumes on the (large) GT-native grid, so nnUNet's default worker
    # count can exhaust RAM ("Some background worker is 6 feet under" = OOM).
    # Lower this if you hit that (2 is safe; raise if you have RAM headroom).
    echo "[run_train]     workers: -np ${PREPROCESS_NP:-<nnUNet default>}"
    nnUNetv2_preprocess -d "$CTRL_DATASET_ID" -plans_name "$PLAN_NAME" \
        -c "$CONFIGURATION" ${PREPROCESS_NP:+-np "$PREPROCESS_NP"}
fi

echo "[run_train] (5) POTHOLE-4 GATE: check_preprocessed ${GATE_ARGS:+($GATE_ARGS)}"
python3 "$HERE/diagnostics/check_preprocessed.py" \
    --config "$CONFIG" --control "$CONTROL" --plan-name "$PLAN_NAME" $GATE_ARGS

mkdir -p "$WORK_TMP"
ADAPTED="$WORK_TMP/ckpt_${REF_DATASET_ID}_to${N_CHANNELS}ch_${CONTROL}.pth"
echo "[run_train] (6) first-conv surgery 1ch->${N_CHANNELS}ch (mask_init=$MASK_INIT)"
python3 "$HERE/scripts/adapt_checkpoint.py" \
    --in "$REF_CKPT" --out "$ADAPTED" \
    --channels "$N_CHANNELS" --mask-init "$MASK_INIT" \
    --report-json "$WORK_TMP/adapt_report_${CONTROL}.json"

# Install the corrector runtime modules into nnunetv2 so `-tr` can discover the
# trainer AND its sibling imports resolve. The OrbitalCascade trainer imports
# `corrector_augment` + `corrector_stratified_loader` from its own package
# (nnunetv2.training.nnUNetTrainer.*), so all four must land in that dir. Copying
# is cheap + idempotent; missing files (e.g. on the stock-trainer path) are skipped.
echo "[run_train] (6b) install corrector runtime modules -> nnunetv2 ($CORRECTOR_TRAINER)"
python3 - "$HERE/engine" <<'PY'
import sys, shutil, os
import nnunetv2.training.nnUNetTrainer.nnUNetTrainer as m
pkg = os.path.dirname(m.__file__)
eng = sys.argv[1]
mods = ["nnUNetTrainer_corrector.py", "nnUNetTrainer_OrbitalCascade.py",
        "corrector_augment.py", "corrector_stratified_loader.py"]
for name in mods:
    src = os.path.join(eng, name)
    if os.path.isfile(src):
        shutil.copyfile(src, os.path.join(pkg, name))
        print(f"[run_train] installed {name} -> {pkg}")
    else:
        print(f"[run_train] (skip) {name} not present in {eng}")
PY

# The corrector trainer reads its schedule from these (corrector.yaml::finetune).
export CORRECTOR_EPOCHS CORRECTOR_LR

# RESUME=1: continue an interrupted run from its latest/best checkpoint
# (nnUNetv2_train --c). Use this after a crash/kill (e.g. system-RAM OOM) so you
# don't restart from scratch. --c ignores -pretrained_weights (it restores the
# in-progress checkpoint instead), so we drop it here.
if [[ "${RESUME:-0}" == "1" ]]; then
    echo "[run_train] (7) RESUME nnUNetv2_train --c $CTRL_DATASET_ID $CONFIGURATION $FOLD -p $PLAN_NAME"
    echo "          (continues from checkpoint_final/latest/best in the fold dir)"
    nnUNetv2_train "$CTRL_DATASET_ID" "$CONFIGURATION" "$FOLD" \
        -p "$PLAN_NAME" -tr "$CORRECTOR_TRAINER" --c
else
    echo "[run_train] (7) nnUNetv2_train $CTRL_DATASET_ID $CONFIGURATION $FOLD -p $PLAN_NAME"
    echo "          trainer=$CORRECTOR_TRAINER  epochs=$CORRECTOR_EPOCHS  initial_lr=$CORRECTOR_LR"
    nnUNetv2_train "$CTRL_DATASET_ID" "$CONFIGURATION" "$FOLD" \
        -p "$PLAN_NAME" -tr "$CORRECTOR_TRAINER" -pretrained_weights "$ADAPTED"
fi

echo "[run_train] done: Dataset${CTRL_DATASET_ID} fold $FOLD finetuned from $REF_CKPT"
