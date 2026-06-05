#!/usr/bin/env bash
# ============================================================
# Run every simulation experiment (thin, thick, optional real)
# end-to-end in ONE command, sharing the strategy-independent
# work and (optionally) running thin/thick in parallel on two GPUs.
#
# Why this wrapper exists
# -----------------------
# A naive "two full pipelines at once" is UNSAFE: thin and thick
# share several strategy-independent phases and one shared output:
#   * cnisp-train               -> one checkpoint
#   * nnunet-predict (native)   -> prediction/native/ (+ provenance)
#   * cnisp-prep-dataset835-gt  -> labels_dataset835/ (dense GT patches)
#   * compare's cross-experiment summary -> comparison/experiment_summary.*
#                                            + comparison/viz/experiments/
# Running those concurrently races (wipe+rebuild, half-written files).
#
# So this script:
#   1. runs the SHARED phases exactly once (sequential),
#   2. runs the PER-EXPERIMENT phases for thin & thick (sequential, or
#      parallel on two GPUs -- their paths are disjoint),
#   3. runs `compare` sequentially at the end (per-exp tables + the
#      shared cross-experiment overlay, which must not race),
#   4. optionally runs the opt-in real-paired line.
#
# Usage
# -----
#   bash run_all_experiments.sh                         # sequential, GPU 0
#   bash run_all_experiments.sh --gpu 1                 # sequential, GPU 1
#   bash run_all_experiments.sh --parallel              # thin->GPU0, thick->GPU1
#   bash run_all_experiments.sh --parallel \
#        --gpu-thin 0 --gpu-thick 1
#   bash run_all_experiments.sh --with-real             # also run real-pair line
#   bash run_all_experiments.sh --parallel --with-real --gpu-real 0
#
# Notes
# -----
# * The thick config is generated from nnunet/configs.yaml by flipping
#   sweep_degrade_mode -> thick, so the two configs never drift.
# * `--force` is intentionally NOT supported here: it would re-run the
#   already-finished thin/atlas_gt. Provenance skips what's fresh and
#   rebuilds the rest. To force a full clean rebuild, call run_pipeline.sh
#   directly with --force.
# * real EXP is hard-coded 'real' inside run_pipeline.sh; it ignores
#   sweep_degrade_mode, so it runs off the base (thin) config fine. It
#   needs a populated realpair manifest (configs.yaml::realpair_manifest).
# ============================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# ── Defaults ─────────────────────────────────────────────────
BASE_CONFIG="$REPO_ROOT/nnunet/configs.yaml"     # thin (sweep_degrade_mode: thin)
THICK_CONFIG="$REPO_ROOT/nnunet/configs_thick.yaml"
PARALLEL=0
WITH_REAL=0
GPU=0            # sequential GPU
GPU_THIN=0       # parallel: thin GPU
GPU_THICK=1      # parallel: thick GPU
GPU_REAL=""      # defaults to GPU (seq) or GPU_THIN (parallel)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --parallel)     PARALLEL=1; shift ;;
        --with-real)    WITH_REAL=1; shift ;;
        --gpu)          GPU="$2"; shift 2 ;;
        --gpu=*)        GPU="${1#*=}"; shift ;;
        --gpu-thin)     GPU_THIN="$2"; shift 2 ;;
        --gpu-thin=*)   GPU_THIN="${1#*=}"; shift ;;
        --gpu-thick)    GPU_THICK="$2"; shift 2 ;;
        --gpu-thick=*)  GPU_THICK="${1#*=}"; shift ;;
        --gpu-real)     GPU_REAL="$2"; shift 2 ;;
        --gpu-real=*)   GPU_REAL="${1#*=}"; shift ;;
        -h|--help)      sed -n '2,/^# ====/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)              echo "[run_all] unknown option: $1" >&2; exit 2 ;;
    esac
done

# ── Phase groups (keep in sync with run_pipeline.sh) ─────────
SHARED_PHASES=(cnisp-train nnunet-predict cnisp-prep-dataset835-gt)
PEREXP_PHASES=(
    cnisp-infer
    nnunet-predict-sweep
    cnisp-prep-dataset835-sparse
    cnisp-infer-nnunet-pred
    cnisp-native-remap
    cnisp-viz
)

run_pipe() { bash "$REPO_ROOT/run_pipeline.sh" "$@"; }

# ── Generate the thick config from the (thin) base ───────────
if ! grep -q '^sweep_degrade_mode:' "$BASE_CONFIG"; then
    echo "[run_all] ERROR: $BASE_CONFIG has no sweep_degrade_mode key" >&2
    exit 2
fi
sed 's/^sweep_degrade_mode:.*/sweep_degrade_mode: thick/' \
    "$BASE_CONFIG" > "$THICK_CONFIG"
echo "[run_all] thick config -> $THICK_CONFIG"

# ── 1) Shared phases, once ───────────────────────────────────
SHARED_GPU="$GPU"
[[ $PARALLEL -eq 1 ]] && SHARED_GPU="$GPU_THIN"
echo ""
echo "############################################################"
echo "# [run_all] shared phases (once, GPU=$SHARED_GPU)"
echo "############################################################"
run_pipe --gpu "$SHARED_GPU" --config "$BASE_CONFIG" "${SHARED_PHASES[@]}"

# ── 2) Per-experiment phases (thin + thick) ──────────────────
if [[ $PARALLEL -eq 1 ]]; then
    echo ""
    echo "############################################################"
    echo "# [run_all] thin (GPU=$GPU_THIN) || thick (GPU=$GPU_THICK)  [parallel]"
    echo "############################################################"
    LOG_THIN="$REPO_ROOT/.run_all_thin.log"
    LOG_THICK="$REPO_ROOT/.run_all_thick.log"
    ( run_pipe --gpu "$GPU_THIN"  --config "$BASE_CONFIG"  "${PEREXP_PHASES[@]}" ) \
        >"$LOG_THIN" 2>&1 &
    pid_thin=$!
    ( run_pipe --gpu "$GPU_THICK" --config "$THICK_CONFIG" "${PEREXP_PHASES[@]}" ) \
        >"$LOG_THICK" 2>&1 &
    pid_thick=$!
    echo "[run_all] thin  pid=$pid_thin  log=$LOG_THIN"
    echo "[run_all] thick pid=$pid_thick log=$LOG_THICK"
    rc=0
    wait "$pid_thin"  || { echo "[run_all] THIN failed (see $LOG_THIN)"  >&2; rc=1; }
    wait "$pid_thick" || { echo "[run_all] THICK failed (see $LOG_THICK)" >&2; rc=1; }
    echo "── thin tail ──";  tail -n 15 "$LOG_THIN"  || true
    echo "── thick tail ──"; tail -n 15 "$LOG_THICK" || true
    [[ $rc -ne 0 ]] && exit 1
else
    echo ""
    echo "############################################################"
    echo "# [run_all] thin then thick  [sequential, GPU=$GPU]"
    echo "############################################################"
    run_pipe --gpu "$GPU" --config "$BASE_CONFIG"  "${PEREXP_PHASES[@]}"
    run_pipe --gpu "$GPU" --config "$THICK_CONFIG" "${PEREXP_PHASES[@]}"
fi

# ── 3) compare, sequential (per-exp tables + shared cross-exp) ──
# Run last and never in parallel: each compare rewrites the shared
# comparison/experiment_summary.* + viz/experiments/ aggregate.
CMP_GPU="$GPU"; [[ $PARALLEL -eq 1 ]] && CMP_GPU="$GPU_THIN"
echo ""
echo "############################################################"
echo "# [run_all] compare thin + thick (sequential)"
echo "############################################################"
run_pipe --gpu "$CMP_GPU" --config "$BASE_CONFIG"  compare
run_pipe --gpu "$CMP_GPU" --config "$THICK_CONFIG" compare

# ── 4) Optional real-paired line ─────────────────────────────
if [[ $WITH_REAL -eq 1 ]]; then
    RGPU="${GPU_REAL:-$CMP_GPU}"
    echo ""
    echo "############################################################"
    echo "# [run_all] real-paired line (GPU=$RGPU)"
    echo "############################################################"
    run_pipe --gpu "$RGPU" --config "$BASE_CONFIG" \
        cnisp-prep-realpair cnisp-infer-realpair
fi

echo ""
echo "============================================================"
echo "[run_all] all experiments done."
echo "  cross-experiment overlay : <work_dir>/comparison/viz/experiments/"
echo "  cross-experiment table   : <work_dir>/comparison/experiment_summary.{csv,txt}"
[[ $WITH_REAL -eq 1 ]] && \
echo "  real-paired results      : <cnisp_out>/<model>/runs/real/real_pair/test_results.csv"
echo "============================================================"
