#!/usr/bin/env bash
# ============================================================
# nnUNet inference + CNISP retrain/infer + per-model
# visualization + nnUNet-vs-CNISP paired comparison.
#
# What each phase does:
#   cnisp-train     Train the orbital implicit shape prior.
#                   (orbital_shape_prior_st1/scripts/run_02_train.sh)
#                   Auto-skipped if best_checkpoint.pth already exists;
#                   pass --force-train to override.
#
#   nnunet-predict  Run nnUNetv2_predict on the staged native CT inputs
#                   under $work_dir/nnunet_input/.
#                   (nnunet/run_predict_native.sh)
#
#   cnisp-infer     CNISP test-time latent optimization + per-step
#                   native-space mapping (writes native_space_step_XX/
#                   + native_sweep_manifest.json that the compare phase
#                   consumes).
#                   (orbital_shape_prior_st1/scripts/run_03_test.sh)
#
#   cnisp-viz       CNISP-side result summary: recon_summary.png,
#                   cross_resolution_analysis/, native_sweep_summary.json.
#                   (orbital_shape_prior_st1/scripts/run_04_visualization.sh)
#
#   compare         nnUNet vs CNISP paired Dice tables
#                   (paired_per_source.csv, paired_summary.csv|.txt).
#                   build_cnisp_native_sweep.py is a no-op when
#                   cnisp-infer already wrote native_space_step_XX/.
#                   (nnunet/build_cnisp_native_sweep.py
#                    + nnunet/compare_native.py)
#
# Dependency order (the order phases run when none are specified):
#   cnisp-train -> nnunet-predict -> cnisp-infer -> cnisp-viz -> compare
#
# Usage:
#   bash run_pipeline.sh                                   # all phases
#   bash run_pipeline.sh cnisp-infer cnisp-viz             # subset
#   bash run_pipeline.sh --force-train                     # retrain even if checkpoint exists
#   bash run_pipeline.sh --test-config <path>              # override CNISP test yaml
#   bash run_pipeline.sh --config <path>                   # override nnunet/configs.yaml
#   bash run_pipeline.sh --gpu 0                           # forward to CUDA_VISIBLE_DEVICES
#   bash run_pipeline.sh -h
# ============================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# ── Defaults ─────────────────────────────────────────────────
CONFIG="$REPO_ROOT/nnunet/configs.yaml"
TEST_CONFIG=""                       # passed through to CNISP run_03/run_04
FORCE_TRAIN=0
GPU_OVERRIDE="1"                      # CUDA_VISIBLE_DEVICES override
PHASES_DEFAULT=(cnisp-train nnunet-predict cnisp-infer cnisp-viz compare)
PHASES=()

usage() {
    sed -n '2,/^# ====/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 0
}

# ── Arg parse ────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)         usage ;;
        --config)          CONFIG="$2"; shift 2 ;;
        --config=*)        CONFIG="${1#*=}"; shift ;;
        --test-config)     TEST_CONFIG="$2"; shift 2 ;;
        --test-config=*)   TEST_CONFIG="${1#*=}"; shift ;;
        --force-train)     FORCE_TRAIN=1; shift ;;
        --gpu)             GPU_OVERRIDE="$2"; shift 2 ;;
        --gpu=*)           GPU_OVERRIDE="${1#*=}"; shift ;;
        --)                shift; while [[ $# -gt 0 ]]; do PHASES+=("$1"); shift; done ;;
        -*)
            echo "[run_pipeline] unknown option: $1" >&2
            usage
            ;;
        *)                 PHASES+=("$1"); shift ;;
    esac
done

if [[ ${#PHASES[@]} -eq 0 ]]; then
    PHASES=("${PHASES_DEFAULT[@]}")
fi

if [[ ! -f "$CONFIG" ]]; then
    echo "[run_pipeline] config not found: $CONFIG" >&2
    exit 2
fi

# ── Validate phase names early (no PyYAML needed) ────────────
VALID_PHASES=(cnisp-train nnunet-predict cnisp-infer cnisp-viz compare)
for phase in "${PHASES[@]}"; do
    found=0
    for v in "${VALID_PHASES[@]}"; do [[ "$phase" == "$v" ]] && found=1; done
    if [[ $found -eq 0 ]]; then
        echo "[run_pipeline] unknown phase: '$phase'" >&2
        echo "  valid phases: ${VALID_PHASES[*]}" >&2
        exit 2
    fi
done

if [[ -n "$GPU_OVERRIDE" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU_OVERRIDE"
fi

# ── Resolve CNISP paths from yaml so we can do existence checks ──
read_yaml_field() {
    # $1 = yaml file, $2 = dotted field
    python3 - "$1" "$2" <<'PY'
import sys, yaml
path, field = sys.argv[1], sys.argv[2]
with open(path) as f:
    cfg = yaml.safe_load(f) or {}
cur = cfg
for k in field.split("."):
    if not isinstance(cur, dict):
        cur = None
        break
    cur = cur.get(k)
print("" if cur is None else cur)
PY
}

CNISP_PATHS_YAML_REL="$(read_yaml_field "$CONFIG" "cnisp_paths_yaml")"
if [[ -z "$CNISP_PATHS_YAML_REL" ]]; then
    echo "[run_pipeline] $CONFIG: missing 'cnisp_paths_yaml'" >&2
    exit 2
fi
# Resolve relative paths against the repo root
if [[ "$CNISP_PATHS_YAML_REL" = /* ]]; then
    CNISP_PATHS_YAML="$CNISP_PATHS_YAML_REL"
else
    CNISP_PATHS_YAML="$REPO_ROOT/$CNISP_PATHS_YAML_REL"
fi

CNISP_MODEL_NAME="$(read_yaml_field "$CONFIG" "cnisp_model_name")"
CNISP_MODEL_BASEDIR="$(read_yaml_field "$CNISP_PATHS_YAML" "model_basedir")"
CNISP_OUTPUT_BASEDIR="$(read_yaml_field "$CNISP_PATHS_YAML" "output_basedir")"
WORK_DIR="$(read_yaml_field "$CONFIG" "work_dir")"

echo "============================================================"
echo "CNISP <-> nnUNet pipeline"
echo "  repo_root:           $REPO_ROOT"
echo "  config:              $CONFIG"
echo "  cnisp_paths_yaml:    $CNISP_PATHS_YAML"
echo "  cnisp_model_name:    $CNISP_MODEL_NAME"
echo "  cnisp_model_dir:     $CNISP_MODEL_BASEDIR/$CNISP_MODEL_NAME"
echo "  cnisp_output_dir:    $CNISP_OUTPUT_BASEDIR/$CNISP_MODEL_NAME"
echo "  nnunet work_dir:     $WORK_DIR"
[[ -n "$TEST_CONFIG"   ]] && echo "  cnisp test yaml:     $TEST_CONFIG"
[[ -n "$GPU_OVERRIDE"  ]] && echo "  CUDA_VISIBLE_DEVICES=$GPU_OVERRIDE"
echo "  phases:              ${PHASES[*]}"
echo "============================================================"

# ── Phase implementations ────────────────────────────────────

phase_cnisp_train() {
    echo ""
    echo "[phase] cnisp-train -----------------------------------"
    local ckpt="$CNISP_MODEL_BASEDIR/$CNISP_MODEL_NAME/best_checkpoint.pth"
    if [[ -f "$ckpt" && $FORCE_TRAIN -eq 0 ]]; then
        echo "  best_checkpoint.pth already exists:"
        echo "    $ckpt"
        echo "  -> skipping training (pass --force-train to override)."
        return 0
    fi
    bash "$REPO_ROOT/orbital_shape_prior_st1/scripts/run_02_train.sh"
}

phase_nnunet_predict() {
    echo ""
    echo "[phase] nnunet-predict --------------------------------"
    CONFIG="$CONFIG" bash "$REPO_ROOT/nnunet/run_predict_native.sh"
}

phase_cnisp_infer() {
    echo ""
    echo "[phase] cnisp-infer -----------------------------------"
    if [[ -n "$TEST_CONFIG" ]]; then
        bash "$REPO_ROOT/orbital_shape_prior_st1/scripts/run_03_test.sh" "$TEST_CONFIG"
    else
        bash "$REPO_ROOT/orbital_shape_prior_st1/scripts/run_03_test.sh"
    fi
}

phase_cnisp_viz() {
    echo ""
    echo "[phase] cnisp-viz -------------------------------------"
    if [[ -n "$TEST_CONFIG" ]]; then
        bash "$REPO_ROOT/orbital_shape_prior_st1/scripts/run_04_visualization.sh" "$TEST_CONFIG"
    else
        bash "$REPO_ROOT/orbital_shape_prior_st1/scripts/run_04_visualization.sh"
    fi
}

phase_compare() {
    echo ""
    echo "[phase] compare ---------------------------------------"
    python3 "$REPO_ROOT/nnunet/build_cnisp_native_sweep.py" --config "$CONFIG"
    python3 "$REPO_ROOT/nnunet/compare_native.py"            --config "$CONFIG"
}

# ── Dispatch ─────────────────────────────────────────────────
START_TS="$(date +%s)"
for phase in "${PHASES[@]}"; do
    case "$phase" in
        cnisp-train)    phase_cnisp_train ;;
        nnunet-predict) phase_nnunet_predict ;;
        cnisp-infer)    phase_cnisp_infer ;;
        cnisp-viz)      phase_cnisp_viz ;;
        compare)        phase_compare ;;
    esac
done
END_TS="$(date +%s)"

echo ""
echo "============================================================"
printf "Pipeline complete in %ds. Phases run: %s\n" \
    "$((END_TS - START_TS))" "${PHASES[*]}"
echo ""
echo "Where to look for results:"
echo "  CNISP single-model summary:"
echo "    $CNISP_OUTPUT_BASEDIR/$CNISP_MODEL_NAME/recon_summary.png"
echo "    $CNISP_OUTPUT_BASEDIR/$CNISP_MODEL_NAME/cross_resolution_analysis/"
echo "    $CNISP_OUTPUT_BASEDIR/$CNISP_MODEL_NAME/native_sweep_summary.json"
echo "  nnUNet vs CNISP paired comparison:"
echo "    $WORK_DIR/paired_per_source.csv"
echo "    $WORK_DIR/paired_summary.csv"
echo "    $WORK_DIR/paired_summary.txt"
echo "============================================================"
