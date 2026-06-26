#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════
# nnUNet-C "corrector" 4-stage orchestrator
# ════════════════════════════════════════════════════════════════════
# Strict sequential dependency chain (each stage gated on the previous one's
# products; pass --force to redo a present stage):
#
#   Stage 1  nnUNet(835) sparse predict on DEGRADED CT
#            for corrector_train + CNISP test caselists.
#            products: ${WORK_DIR}/input/${EXP}/sparse_step_XX/*_0000.nii.gz
#                      ${WORK_DIR}/prediction/${EXP}/sparse_step_XX[_native]/*.nii.gz   (B prelabels)
#                      ${aligned_dir}/labels_dataset835_${EXP}_step_XX/*.nii.gz          (Stage-3 nnunet_pred input)
#
#   Stage 2  Retrain CNISP with GT observations -> ${CNISP_MODEL_NAME}
#            (cross-folder: cd CNISP, PYTHONPATH=., NO conda)
#            product : ${CNISP_MODEL_BASEDIR}/${CNISP_MODEL_NAME}/best_checkpoint.pth
#
#   Stage 3  CNISP test-optimization -> prelabels (control C)
#            for corrector_train + test, IDENTICAL settings (only casefile differs).
#            product : runs/${EXP}/${RUN_TAG}/native_space_step_XX/*_cnisp_stepXX.nii.gz
#
#   Stage 4  Build 5ch Dataset + finetune corrector (B=855, C=845).
#            per control: build_dataset -> plan/preprocess -> merge plan (potholes 1&3)
#                         -> preprocess -> POTHOLE-4 GATE -> first-conv surgery -> train.
#
# Stage 2 is the most env-sensitive step: training happens in
# orbital_shape_prior_st1/ but this orchestrator lives in nnunet-c/. We cd into
# the CNISP project, set PYTHONPATH explicitly, and pass CUDA through. No conda
# activation (per project convention; export the right interpreter on PATH).
#
# Usage:
#   bash nnunet-c/run_full_pipeline.sh                 # all stages
#   bash nnunet-c/run_full_pipeline.sh 1 3             # only stages 1 and 3
#   FORCE=1 bash nnunet-c/run_full_pipeline.sh 4
# ════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── Config block (single source of truth) ────────────────────────────
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
CNISP_DIR="$REPO_ROOT/orbital_shape_prior_st1"
CONFIG="${CONFIG:-$HERE/configs/corrector.yaml}"

CONTROLS="${CONTROLS:-B C}"          # which 5ch controls to build/finetune (A is external)
FOLD="${FOLD:-0}"                    # finetune fold for Stage 4
PLAN_NAME="${PLAN_NAME:-nnUNetPlansFinetune}"
CHECKPOINT="${CHECKPOINT:-best}"     # CNISP checkpoint for Stage 3
FORCE="${FORCE:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Resolve identities/paths from corrector.yaml (control C carries CNISP fields).
eval "$(python3 "$HERE/scripts/corrector_env.py" --config "$CONFIG" --control C)"
EXP="$EXPERIMENT"
SYNTH_PKL="${SYNTH_PKL:-$WORK_DIR/corrector_synth_sweep.pkl}"

banner() {
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  $*"
    echo "════════════════════════════════════════════════════════════════"
}

force_flag() { [[ "$FORCE" == "1" ]] && echo "--force" || true; }

# Which stages to run (default: 1 2 3 4).
STAGES="${*:-1 2 3 4}"
run_stage() { [[ " $STAGES " == *" $1 "* ]]; }

# ════════════════════════════════════════════════════════════════════
# Stage 1 — nnUNet(835) sparse predict on degraded CT
# ════════════════════════════════════════════════════════════════════
stage1() {
    banner "Stage 1: nnUNet(835) sparse predict on degraded CT (${EXP})"
    echo "[stage1] inputs : corrector_train.txt + CNISP test_cases.txt"
    echo "[stage1] config : $NNUNET_CONFIG_YAML"

    # 0) derive corrector_train casenames + enforce no-leakage.
    python3 "$HERE/scripts/derive_train_casefile.py" --config "$CONFIG"
    local train_cf="$CASEFILES_DIR/$CORRECTOR_TRAIN_CASEFILE"
    local test_cf="$CASEFILES_DIR/test_cases.txt"

    local sweep_marker="$WORK_DIR/prediction/$EXP/sweep_manifest.json"
    if [[ "$FORCE" != "1" && -f "$sweep_marker" ]]; then
        echo "[stage1] sweep manifest present ($sweep_marker) -> skip (FORCE=1 to redo)"
        return 0
    fi

    # 1) stage CTs (merged caselist) -> work_dir/source_to_path.json
    echo "[stage1] (1) prepare_inputs (corrector_train + test)"
    python3 "$REPO_ROOT/nnunet/prepare_inputs.py" --config "$NNUNET_CONFIG_YAML" \
        --casefile "$train_cf" --casefile "$test_cf"

    # 2) synthesize the (source, step) sweep grid from the GT-obs degradation bank
    echo "[stage1] (2) synth_train_sweep -> $SYNTH_PKL"
    python3 "$REPO_ROOT/nnunet/synth_train_sweep.py" --config "$NNUNET_CONFIG_YAML" \
        --train-config "$CNISP_DIR/configs/$CNISP_TRAIN_YAML" --out "$SYNTH_PKL"

    # 3) degrade CT -> 4) nnUNet sparse predict -> 5) canonical-align patches
    echo "[stage1] (3) sparsify_inputs --mode $EXP"
    python3 "$REPO_ROOT/nnunet/sparsify_inputs.py" --config "$NNUNET_CONFIG_YAML" \
        --mode "$EXP" --modality ct --experiment "$EXP" \
        --sweep-pkl "$SYNTH_PKL" $(force_flag)
    echo "[stage1] (4) predict_sparse_iso"
    python3 "$REPO_ROOT/nnunet/predict_sparse_iso.py" --config "$NNUNET_CONFIG_YAML" \
        --experiment "$EXP" --skip-step-01 $(force_flag)
    echo "[stage1] (5) build_dataset835_sparse_patches (standard prefix for Stage 3)"
    python3 "$REPO_ROOT/nnunet/build_dataset835_sparse_patches.py" \
        --config "$NNUNET_CONFIG_YAML" --experiment "$EXP" --skip-step-01 $(force_flag)

    echo "[stage1] products under: $WORK_DIR/{input,prediction}/$EXP/ and aligned labels_dataset835_${EXP}_step_XX/"
}

# ════════════════════════════════════════════════════════════════════
# Stage 2 — retrain CNISP with GT observations (cross-folder)
# ════════════════════════════════════════════════════════════════════
stage2() {
    banner "Stage 2: retrain CNISP (GT obs) -> $CNISP_MODEL_NAME"
    local ckpt="$CNISP_MODEL_BASEDIR/$CNISP_MODEL_NAME/best_checkpoint.pth"
    if [[ "$FORCE" != "1" && -f "$ckpt" ]]; then
        echo "[stage2] checkpoint present ($ckpt) -> skip (FORCE=1 to retrain)"
        return 0
    fi
    echo "[stage2] train yaml : $CNISP_DIR/configs/$CNISP_TRAIN_YAML"
    echo "[stage2] cd $CNISP_DIR ; PYTHONPATH=.:$REPO_ROOT ; CUDA=$CUDA_VISIBLE_DEVICES (no conda)"
    (
        cd "$CNISP_DIR"
        PYTHONPATH=".:$REPO_ROOT:${PYTHONPATH:-}" \
        CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
        bash scripts/run_02_train.sh "configs/$CNISP_TRAIN_YAML"
    )
    [[ -f "$ckpt" ]] || { echo "[stage2] ERROR: expected $ckpt not produced" >&2; exit 1; }
    echo "[stage2] product: $ckpt"
}

# ════════════════════════════════════════════════════════════════════
# Stage 3 — CNISP test-optimization -> prelabels (control C)
# ════════════════════════════════════════════════════════════════════
stage3() {
    banner "Stage 3: CNISP prelabels (test-opt), corrector_train + test"
    local run_dir="$CNISP_OUTPUT_BASEDIR/$CNISP_MODEL_NAME/runs/$EXP/$RUN_TAG"
    local marker="$run_dir/native_sweep_manifest.json"
    if [[ "$FORCE" != "1" && -f "$marker" ]]; then
        echo "[stage3] prelabels present ($marker) -> skip (FORCE=1 to redo)"
        return 0
    fi
    CONFIG="$CONFIG" CHECKPOINT="$CHECKPOINT" bash "$HERE/scripts/gen_prelabels.sh" both
    echo "[stage3] products under: $run_dir/native_space_step_XX/"
}

# ════════════════════════════════════════════════════════════════════
# Stage 4 — build datasets + finetune correctors (B, C)
# ════════════════════════════════════════════════════════════════════
stage4() {
    banner "Stage 4: build 5ch datasets + finetune (controls: $CONTROLS)"
    : "${nnUNet_raw:?export nnUNet_raw}"
    : "${nnUNet_preprocessed:?export nnUNet_preprocessed}"
    : "${nnUNet_results:?export nnUNet_results}"
    for ctrl in $CONTROLS; do
        banner "Stage 4 [$ctrl]: build_dataset"
        CONFIG="$CONFIG" bash "$HERE/run_build_dataset.sh" "$ctrl" train
        banner "Stage 4 [$ctrl]: finetune (plan/preprocess/merge/GATE/surgery/train)"
        CONFIG="$CONFIG" PLAN_NAME="$PLAN_NAME" bash "$HERE/run_train.sh" "$ctrl" "$FOLD"
    done
}

# ── Dispatch (set -e preserved: a selected stage that fails aborts) ──
echo "[run_full_pipeline] config=$CONFIG experiment=$EXP controls='$CONTROLS' fold=$FOLD stages='$STAGES'"
if run_stage 1; then stage1; fi
if run_stage 2; then stage2; fi
if run_stage 3; then stage3; fi
if run_stage 4; then stage4; fi
banner "run_full_pipeline complete (stages: $STAGES)"
