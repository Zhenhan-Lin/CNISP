#!/usr/bin/env bash
# Stage 3: generate CNISP prelabels via test-optimization, with IDENTICAL
# settings for the corrector_train caselist and the CNISP test caselist.
#
# Reproducibility contract (plan section VI): both invocations share the same
# paths.yaml (-p), GT-obs train/architecture config (-t), test-opt config (-c),
# model checkpoint (-m + --checkpoint), --test-label-source nnunet_pred,
# --experiment, and --run-tag. ONLY --test-casefile differs. Do not edit one
# without the other.
#
# Outputs (control C prelabels):
#   ${output_basedir}/<model>/runs/<experiment>/<run_tag>/native_space_step_XX/
#       <gtstem>_cnisp_stepXX.nii.gz  + manifest.json
#
# Usage:
#   bash nnunet-c/scripts/gen_prelabels.sh            # both caselists
#   bash nnunet-c/scripts/gen_prelabels.sh train      # corrector_train only
#   bash nnunet-c/scripts/gen_prelabels.sh test       # CNISP test only
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
NNC_ROOT="$(cd "$HERE/.." && pwd)"
REPO_ROOT="$(cd "$NNC_ROOT/.." && pwd)"
CONFIG="${CONFIG:-$NNC_ROOT/configs/corrector.yaml}"
WHICH="${1:-both}"           # both | train | test
CHECKPOINT="${CHECKPOINT:-best}"

# Resolve identities/paths from corrector.yaml (uses control C identity here).
eval "$(python3 "$HERE/corrector_env.py" --config "$CONFIG" --control C)"

CNISP_DIR="$REPO_ROOT/orbital_shape_prior_st1"
PATHS_YAML="$CNISP_DIR/configs/paths.yaml"
TRAIN_YAML="$CNISP_DIR/configs/$CNISP_TRAIN_YAML"
TEST_YAML="$CNISP_DIR/configs/$CNISP_TEST_YAML"

# Ensure the corrector_train casenames file exists (derive from source_ids) and
# enforce the no-leakage asserts before any inference runs.
echo "[gen_prelabels] deriving corrector_train casefile under $CASEFILES_DIR"
python3 "$HERE/derive_train_casefile.py" --config "$CONFIG"

run_infer() {
    local casefile="$1"
    echo "================================================================"
    echo "[gen_prelabels] CNISP test-optimization"
    echo "  model      : $CNISP_MODEL_NAME (checkpoint=$CHECKPOINT)"
    echo "  experiment : $EXPERIMENT   run_tag: $RUN_TAG"
    echo "  test yaml  : $TEST_YAML"
    echo "  casefile   : $casefile"
    echo "================================================================"
    (
        cd "$CNISP_DIR"
        PYTHONPATH=".:$REPO_ROOT:${PYTHONPATH:-}" \
        python3 scripts/03_infer.py \
            -p "$PATHS_YAML" \
            -t "$TRAIN_YAML" \
            -c "$TEST_YAML" \
            -m "$CNISP_MODEL_NAME" \
            --checkpoint "$CHECKPOINT" \
            --test-label-source nnunet_pred \
            --run-tag "$RUN_TAG" \
            --experiment "$EXPERIMENT" \
            --test-casefile "$casefile"
    )
}

case "$WHICH" in
    train) run_infer "$CORRECTOR_TRAIN_CASEFILE" ;;
    test)  run_infer "test_cases.txt" ;;
    both)
        run_infer "$CORRECTOR_TRAIN_CASEFILE"
        run_infer "test_cases.txt"
        ;;
    *) echo "usage: gen_prelabels.sh [both|train|test]" >&2; exit 2 ;;
esac

echo "[gen_prelabels] done ($WHICH)"
