#!/bin/bash
# ============================================================
# Average-shape (zero-latent) Dice baseline on the test set.
#
# Decodes the CNISP shape prior's MEAN anatomy f(x, z=0) and scores its Dice on
# every test case in the SAME frame + Dice function as the fitted-latent test
# (scripts/033_average_shape_dice.py -> eval_case_at_resolution with a zero latent
# override, step_size=1). This is the FLOOR that per-case latent fitting beats.
#
# Mirrors run_03_test.sh's env/model resolution.
#   ./run_033_average_shape.sh [test_yaml] [test_label_source] [test_casefile]
# Examples:
#   ./run_033_average_shape.sh                                  # ceiling (atlas_gt)
#   ./run_033_average_shape.sh "" atlas_gt test_cases_v7small.txt   # subset
# ============================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$REPO_ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

_yaml_scalar() {  # $1=file $2=key
    grep -E "^[[:space:]]*$2:" "$1" | head -1 \
        | sed -E 's/^[^:]*:[[:space:]]*//; s/[[:space:]]*#.*$//' | tr -d '"'"'"
}

PATHS_YAML="$PROJECT_ROOT/configs/paths.yaml"
TRAIN_YAML="${CNISP_TRAIN_YAML:-$PROJECT_ROOT/configs/train_sty2.yaml}"
[[ -f "$TRAIN_YAML" || ! -f "$PROJECT_ROOT/configs/$TRAIN_YAML" ]] || \
    TRAIN_YAML="$PROJECT_ROOT/configs/$TRAIN_YAML"
TEST_YAML="${1:-$PROJECT_ROOT/configs/test_default.yaml}"
MODEL_NAME="${CNISP_MODEL_NAME:-$(_yaml_scalar "$TRAIN_YAML" model_name)}"
CHECKPOINT="${CNISP_CHECKPOINT:-best}"
TEST_LABEL_SOURCE="${2:-}"
TEST_CASEFILE="${3:-}"
OUT_CSV="${OUT_CSV:-$PROJECT_ROOT/average_shape_dice.csv}"

echo "============================================================"
echo "Average-shape (z=0) Dice baseline"
echo "  Paths config:   $PATHS_YAML"
echo "  Train config:   $TRAIN_YAML"
echo "  Test config:    $TEST_YAML"
echo "  Model:          $MODEL_NAME   checkpoint=$CHECKPOINT"
echo "  Out CSV:        $OUT_CSV"
echo "============================================================"

EXTRA=()
[[ -n "$TEST_LABEL_SOURCE" ]] && EXTRA+=(--test-label-source "$TEST_LABEL_SOURCE")
[[ -n "$TEST_CASEFILE"     ]] && EXTRA+=(--test-casefile "$TEST_CASEFILE")

python3 "$PROJECT_ROOT/scripts/033_average_shape_dice.py" \
    -p "$PATHS_YAML" \
    -t "$TRAIN_YAML" \
    -c "$TEST_YAML" \
    -m "$MODEL_NAME" \
    --checkpoint "$CHECKPOINT" \
    --out-csv "$OUT_CSV" \
    "${EXTRA[@]}"
