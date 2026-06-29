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
CHK="${CHK:-checkpoint_best.pth}"     # nnUNet-C predict checkpoint (best)
CNISP_CHK="${CNISP_CHK:-latest}"      # CNISP test checkpoint (latest; matches train prelabels)
RUN_CNISP="${RUN_CNISP:-auto}"        # auto: 1 for control C, 0 otherwise
GPUS="${GPUS:-0 1}"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
CNISP_DIR="$REPO_ROOT/orbital_shape_prior_st1"
export nnUNet_compile="${nnUNet_compile:-f}"
# Test now assembles the 5ch case on CNISP's iso-0.5 head grid (GRID=iso): control
# C's ch1..4 come from CNISP's iso-0.5 prelabels, so the CNISP run must EMIT them.
# EMIT_ISO=auto -> 1 for control C (cnisp) under iso grid, else 0. GRID=gt restores
# the legacy GT-native-grid assembly (and no iso emit needed).
GRID="${GRID:-iso}"
EMIT_ISO="${EMIT_ISO:-auto}"
ISO_PRELABEL_DIR="${ISO_PRELABEL_DIR:-$HERE/data/cnisp_pred_test_iso}"

echo "================================================================"
echo "[predict] control=$CONTROL fold=$FOLD"
echo "[predict] nnUNet-C ckpt=$CHK   CNISP ckpt=$CNISP_CHK"
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

# iso-0.5 prelabels are REQUIRED when the build consumes the iso grid for C.
if [[ "$EMIT_ISO" == "auto" ]]; then
    [[ "$PRELABEL_SOURCE" == "cnisp" && "$GRID" == "iso" ]] && EMIT_ISO=1 || EMIT_ISO=0
fi
ISO_ARGS=""
[[ "$EMIT_ISO" == "1" ]] && ISO_ARGS="--emit-iso-prelabel-dir $ISO_PRELABEL_DIR --emit-iso-mm 0.5"
echo "[predict] GRID=$GRID EMIT_ISO=$EMIT_ISO  (iso dir: $ISO_PRELABEL_DIR)"

# ── single-image debug mode ──────────────────────────────────────────
# SOURCE=<source_id> (or BUILD_CASEFILE=<path>) restricts the test build to ONE
# image (all its steps via --steps auto, or BUILD_STEPS), writing to ISOLATED
# test_input_single/ + predictions_single/ so the full build is never clobbered.
# Pair it with RUN_CNISP=0 (reuse the existing CNISP/native preds; a single image
# does not warrant re-running a full CNISP test sweep).
SOURCE="${SOURCE:-}"
BUILD_CASEFILE="${BUILD_CASEFILE:-}"
TEST_ROOT="$HERE/test_input"
PRED_ROOT="$HERE/predictions"
if [[ -n "$SOURCE" || -n "$BUILD_CASEFILE" ]]; then
    TEST_ROOT="$HERE/test_input_single"
    PRED_ROOT="$HERE/predictions_single"
    if [[ -z "$BUILD_CASEFILE" ]]; then
        BUILD_CASEFILE="$(mktemp "${TMPDIR:-/tmp}/corr_one_XXXXXX.txt")"
        printf '%s\n' "$SOURCE" > "$BUILD_CASEFILE"
    fi
    echo "[predict] SINGLE-IMAGE mode: casefile=$BUILD_CASEFILE"
    echo "          -> isolated outputs under $TEST_ROOT and $PRED_ROOT"
fi

# ── 1. CNISP test inference = CNISP's OWN thick nnunet_pred deployment run ─
# (03_infer.py: test_cases.txt + adaptive sweep from the test yaml; outputs to
#  runs/$EXPERIMENT/$RUN_TAG/native_space_step_XX/.) nnUNet-C only consumes it.
if [[ "$RUN_CNISP" == "1" ]]; then
    echo "[predict] (1) CNISP thick nnunet_pred test (03_infer) -> runs/$EXPERIMENT/$RUN_TAG"
    PYTHONPATH="$CNISP_DIR:$REPO_ROOT:${PYTHONPATH:-}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPUS%% *}}" \
    python3 "$CNISP_DIR/scripts/03_infer.py" \
        -p "$CNISP_DIR/configs/paths.yaml" \
        -t "$CNISP_DIR/configs/$CNISP_TRAIN_YAML" \
        -c "$CNISP_DIR/configs/$CNISP_TEST_YAML" \
        -m "$CNISP_MODEL_NAME" --checkpoint "$CNISP_CHK" \
        --test-label-source nnunet_pred --run-tag "$RUN_TAG" --experiment "$EXPERIMENT" \
        $ISO_ARGS
else
    echo "[predict] (1) skip CNISP test inference (RUN_CNISP=0; using existing runs/$EXPERIMENT/$RUN_TAG)"
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
echo "[predict] (3) build_corrector_testset (convert CNISP runs output -> 5ch) -> $TEST_ROOT"
python3 "$HERE/scripts/build_corrector_testset.py" \
    --config "$CONFIG" --control "$CONTROL" --steps "${BUILD_STEPS:-auto}" \
    --prelabel-grid "$GRID" --out "$TEST_ROOT" \
    ${BUILD_CASEFILE:+--casefile "$BUILD_CASEFILE"}

IMAGES_TS="$TEST_ROOT/$CTRL_DATASET_NAME/imagesTs"
OUT_DIR_PRED="${OUT_DIR_PRED:-$PRED_ROOT/$CTRL_DATASET_NAME/fold_${FOLD}}"
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
MAP_JSON="$TEST_ROOT/$CTRL_DATASET_NAME/test_cases_map.json"
EVAL_CSV="${EVAL_CSV:-$PRED_ROOT/$CTRL_DATASET_NAME/eval_${CONTROL}_fold${FOLD}.csv}"
if [[ "${RUN_EVAL:-1}" == "1" ]]; then
    echo "[predict] (5) eval (prediction -> native GT grid, order 0; Dice on GT grid)"
    python3 "$HERE/diagnostics/eval_corrector.py" \
        --map "$MAP_JSON" --pred-dir "$OUT_DIR_PRED" --out-csv "$EVAL_CSV"
else
    echo "[predict] (5) skip eval (RUN_EVAL=0). Map: $MAP_JSON"
fi
