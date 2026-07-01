#!/usr/bin/env bash
# One-shot: regenerate observed per-step metadata + RE-MAP CNISP masks (native +
# iso) from EXISTING latents/preds, applying the OS-flip / crop-frame fix.
#
# This does NOT re-infer: no latent optimization is run. It only (1) writes the
# per-step OBSERVED-patch alignment metadata the fixed native_mapping needs, and
# (2) re-decodes each saved latent (or reuses the cached pred) and re-runs the
# native/iso mapping, overwriting the previously misplaced masks.
#
# Two prelabel trees are covered (mirrors the two CNISP launchers):
#   TRAIN  nnunet-c/data/aligned_patch  --(032)-->  nnunet-c/data/cnisp_pred
#   TEST   <CNISP aligned_dir>          --(03_infer)--> runs/<exp>/<run_tag>/native_space*/
#                                                   + iso -> data/cnisp_pred_test_iso
#
# Prerequisites (must already exist from the earlier real runs):
#   * saved latents:  runs/<exp>/<run_tag>/step_XX/latents/*.npy  (TEST)
#                     nnunet-c/data/cnisp_pred/latent/*.npy        (TRAIN, 032)
#   * the sparse label patches the metadata is (re)derived from.
#
# Usage:
#   bash nnunet-c/run_remap_fix.sh                 # all stages (meta + train + test)
#   bash nnunet-c/run_remap_fix.sh meta            # only regenerate metadata
#   bash nnunet-c/run_remap_fix.sh train           # only remap the 032 TRAIN prelabels
#   bash nnunet-c/run_remap_fix.sh test            # only remap the CNISP TEST run
#   CONTROL=C EMIT_ISO=1 bash nnunet-c/run_remap_fix.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
CNISP_DIR="$REPO_ROOT/orbital_shape_prior_st1"
CONFIG="${CONFIG:-$HERE/configs/corrector.yaml}"
CONTROL="${CONTROL:-C}"
WHICH="${1:-all}"

# CNISP test-inference checkpoint: 'latest' to MATCH the prelabels the training
# masks were generated with (see run_corrector_predict.sh).
CNISP_CHK="${CNISP_CHK:-latest}"
# Emit iso-0.5 TEST prelabels for control-C (the corrector consumes them).
EMIT_ISO="${EMIT_ISO:-1}"
GPUS="${GPUS:-0 1}"

# Resolve all identities/paths from corrector.yaml (single source of truth).
eval "$(python3 "$HERE/scripts/corrector_env.py" --config "$CONFIG" --control "$CONTROL")"

ISO_DIR="$DATA_ROOT/cnisp_pred_test_iso"

echo "================================================================"
echo "[remap-fix] control=$CONTROL experiment=$EXPERIMENT stage=$WHICH"
echo "[remap-fix] CNISP model=$CNISP_MODEL_NAME  ckpt=$CNISP_CHK  run_tag=$RUN_TAG"
echo "[remap-fix] nnunet config=$NNUNET_CONFIG_YAML"
echo "================================================================"

# ── Stage 1: (re)generate OBSERVED per-step alignment metadata ────────
# Both aligners now write metadata_dataset835_<exp>_step_XX/ next to the
# label_dataset835_<exp>_step_XX/ patches. A plain re-run fills in the missing
# metadata JSONs WITHOUT rewriting the (deterministic) label patches.
do_meta() {
    echo "[remap-fix] (1a) TRAIN tree metadata -> $DATA_ROOT/aligned_patch/metadata_dataset835_${EXPERIMENT}_step_XX/"
    python3 "$HERE/scripts/align_corrector_data.py" --config "$CONFIG"

    echo "[remap-fix] (1b) TEST tree metadata (shared CNISP aligned_dir), split=test"
    python3 "$REPO_ROOT/nnunet/build_dataset835_sparse_patches.py" \
        --config "$NNUNET_CONFIG_YAML" \
        --experiment "$EXPERIMENT" --split test
}

# ── Stage 2: remap the 032 TRAIN prelabels from saved latents ─────────
# run_corrector_cnisp.sh in REMAP_FROM_LATENT mode: each (source,step,eye)
# reuses its saved latent, skips optimization, re-decodes + re-maps, and
# OVERWRITES data/cnisp_pred/*.nii.gz. Parallel across $GPUS.
do_train() {
    echo "[remap-fix] (2) remap TRAIN prelabels (032 --remap-from-latent) -> $DATA_CNISP_PRED"
    REMAP_FROM_LATENT=1 GPUS="$GPUS" bash "$HERE/run_corrector_cnisp.sh"
}

# ── Stage 3: remap the CNISP TEST run from cached preds/latents ───────
# 03_infer.py --resume-from-latent: reuse each (case,step) cached pred (or its
# saved latent) and re-run ONLY the native/iso mapping, overwriting
# runs/<exp>/<run_tag>/native_space*/ (+ iso prelabels when EMIT_ISO=1).
do_test() {
    local iso_args=""
    [[ "$EMIT_ISO" == "1" ]] && iso_args="--emit-iso-prelabel-dir $ISO_DIR --emit-iso-mm 0.5"
    echo "[remap-fix] (3) remap CNISP TEST (03_infer --resume-from-latent) -> runs/$EXPERIMENT/$RUN_TAG/native_space*/"
    [[ "$EMIT_ISO" == "1" ]] && echo "[remap-fix]     + iso prelabels -> $ISO_DIR"
    PYTHONPATH="$CNISP_DIR:$REPO_ROOT:${PYTHONPATH:-}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPUS%% *}}" \
    python3 "$CNISP_DIR/scripts/03_infer.py" \
        -p "$CNISP_PATHS_YAML" \
        -t "$CNISP_DIR/configs/$CNISP_TRAIN_YAML" \
        -c "$CNISP_DIR/configs/$CNISP_TEST_YAML" \
        -m "$CNISP_MODEL_NAME" --checkpoint "$CNISP_CHK" \
        --test-label-source nnunet_pred --run-tag "$RUN_TAG" --experiment "$EXPERIMENT" \
        --resume-from-latent $iso_args
}

case "$WHICH" in
    meta)  do_meta ;;
    train) do_train ;;
    test)  do_test ;;
    all)   do_meta; do_train; do_test ;;
    *) echo "usage: run_remap_fix.sh [all|meta|train|test]" >&2; exit 2 ;;
esac

echo "[remap-fix] done ($WHICH)."
cat <<EOF
[remap-fix] ---------------------------------------------------------------
The CNISP masks / iso prelabels are now geometrically corrected. Their grid,
affine, filenames and manifests are UNCHANGED, so nnUNet-C consumers pick them
up as-is -- but their CACHES must be invalidated (they key on output existence,
not prelabel content):

  TRAIN (control C): rebuild the 5ch dataset (build_corrector_dataset has no
                     cache, so a plain re-run picks up data/cnisp_pred), then
                     re-preprocess + retrain:
    python3 nnunet-c/scripts/build_corrector_dataset.py --control C
    bash nnunet-c/run_train.sh C <fold>

  TEST (control C):  the 5ch testset cache is stale after remap -> force a
                     rebuild (RUN_CNISP=0 reuses the masks we just fixed):
    RUN_CNISP=0 REBUILD_TESTSET=1 bash nnunet-c/run_corrector_predict.sh C <fold>

NOTE: corrector Dice only improves after RETRAINING on the corrected prelabels
(the model was trained on the old, misplaced ones -- inference-only swap would
be a train/test mismatch).
--------------------------------------------------------------------------
EOF
