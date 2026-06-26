#!/usr/bin/env bash
# Parallel CNISP inference for the corrector across GPU0 / GPU1 / CPU.
#
# Concurrency: CNISP test-optimization is per-case independent, so we shard
# SOURCES (OD+OS kept together) across worker processes. Each device runs ONE
# worker process pinned to it:
#   - gpu0, gpu1 : CUDA_VISIBLE_DEVICES=<id>           (GPU concurrency)
#   - cpu        : CUDA_VISIBLE_DEVICES="" -> torch CPU (CPU concurrency)
# Weighted shards (GPUs get more sources than the slow CPU) balance load.
#
# Robustness (the point of this script):
#   * each worker's stdout+stderr -> nnunet-c/logs/{gpu0,gpu1,cpu}.log
#   * each worker's exit code recorded in nnunet-c/logs/cnisp_status.tsv
#   * a FINAL REVIEW lists every worker PASS/FAIL; any worker that crashed
#     (non-zero rc, OOM, killed) is reported with the EXACT re-run command
#   * re-runs are resumable: 032 --skip-existing skips sources already on disk
#   * RERUN_FAILED=1 re-launches ONLY the workers that failed last time
#
# Usage:
#   bash nnunet-c/run_corrector_cnisp.sh                      # gpu0+gpu1+cpu
#   DEVICES="0 1"   bash nnunet-c/run_corrector_cnisp.sh      # GPUs only
#   GPU_SHARDS=3 CPU_SHARDS=1 bash nnunet-c/run_corrector_cnisp.sh
#   RERUN_FAILED=1 bash nnunet-c/run_corrector_cnisp.sh       # retry crashed workers
set -uo pipefail   # NB: no -e; we manage per-worker rc ourselves

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
CNISP_DIR="$REPO_ROOT/orbital_shape_prior_st1"
INFER="${INFER_SCRIPT:-$CNISP_DIR/scripts/032_cnisp_infer_corrector.py}"
LOG_DIR="$HERE/logs"
STATUS="$LOG_DIR/cnisp_status.tsv"
REVIEW="$LOG_DIR/cnisp_review.txt"
mkdir -p "$LOG_DIR"

# ── config block ─────────────────────────────────────────────────────
DEVICES="${DEVICES:-0 1 cpu}"        # space-separated: GPU ids and/or "cpu"
GPU_SHARDS="${GPU_SHARDS:-2}"        # shard slots per GPU worker (weight)
CPU_SHARDS="${CPU_SHARDS:-1}"        # shard slots for the CPU worker (weight)
MODEL="${MODEL:-orbital_ad_v6_5_gt}"
TRAIN_YAML="${TRAIN_YAML:-configs/train_v6_5_gt.yaml}"
TEST_YAML="${TEST_YAML:-configs/test_corrector.yaml}"
CHECKPOINT="${CHECKPOINT:-best}"
LABEL_SOURCE="${LABEL_SOURCE:-nnunet_pred}"
EXPERIMENT="${EXPERIMENT:-thick}"
CASEFILE="${CASEFILE:-corrector_train_cases.txt}"
STEPS="${STEPS:-3,6,9}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"      # global cap on (source,step) samples (0=all)
GPU_THREADS="${GPU_THREADS:-4}"
CPU_THREADS="${CPU_THREADS:-8}"
RERUN_FAILED="${RERUN_FAILED:-0}"

# Parallel arrays describing workers. w_dev is the device TOKEN ("0"/"1"/"cpu")
# -- never empty, so the TSV round-trips cleanly (an empty CUDA field would be
# eaten by `read`, since tab is IFS-whitespace).
w_name=(); w_dev=(); w_ids=(); w_threads=()

cuda_of() { [[ "$1" == "cpu" ]] && echo "" || echo "$1"; }
name_of() { [[ "$1" == "cpu" ]] && echo "cpu" || echo "gpu$1"; }

build_workers_fresh() {
    read -r -a DEV_ARR <<< "$DEVICES"
    local total=0 d
    for d in "${DEV_ARR[@]}"; do
        [[ "$d" == "cpu" ]] && total=$(( total + CPU_SHARDS )) || total=$(( total + GPU_SHARDS ))
    done
    NUM_SHARDS="$total"
    local cursor=0 d w threads ids k
    for d in "${DEV_ARR[@]}"; do
        if [[ "$d" == "cpu" ]]; then w="$CPU_SHARDS"; threads="$CPU_THREADS";
        else w="$GPU_SHARDS"; threads="$GPU_THREADS"; fi
        ids=""
        for ((k=0; k<w; k++)); do ids+="${ids:+,}$(( cursor + k ))"; done
        cursor=$(( cursor + w ))
        w_name+=("$(name_of "$d")"); w_dev+=("$d"); w_ids+=("$ids"); w_threads+=("$threads")
    done
}

build_workers_from_failed() {
    [[ -f "$STATUS" ]] || { echo "[rerun] no status file $STATUS" >&2; exit 2; }
    # columns: name<TAB>dev<TAB>ids<TAB>num_shards<TAB>threads<TAB>pid<TAB>rc
    while IFS=$'\t' read -r name dev ids nsh threads pid rc; do
        [[ "$name" == "name" ]] && continue            # header
        [[ -z "${name:-}" ]] && continue
        if [[ "$rc" != "0" ]]; then
            w_name+=("$name"); w_dev+=("$dev"); w_ids+=("$ids"); w_threads+=("$threads")
            NUM_SHARDS="$nsh"
        fi
    done < "$STATUS"
    if [[ "${#w_name[@]}" -eq 0 ]]; then
        echo "[rerun] no failed workers in $STATUS -- nothing to do."; exit 0
    fi
    echo "[rerun] re-launching ${#w_name[@]} failed worker(s) from $STATUS"
}

worker_cmd() {  # $1=dev $2=ids -> the exact python command (for review/rerun)
    local cuda; cuda="$(cuda_of "$1")"
    echo "CUDA_VISIBLE_DEVICES=\"$cuda\" PYTHONPATH=\".:$REPO_ROOT\" python3 $INFER" \
         "-m $MODEL -t $TRAIN_YAML -c $TEST_YAML --checkpoint $CHECKPOINT" \
         "--test-label-source $LABEL_SOURCE --experiment $EXPERIMENT" \
         "--test-casefile $CASEFILE --steps $STEPS --max-samples $MAX_SAMPLES" \
         "--num-shards $NUM_SHARDS --shard-id $2 --skip-existing"
}

# ── build the worker set ─────────────────────────────────────────────
if [[ "$RERUN_FAILED" == "1" ]]; then build_workers_from_failed; else build_workers_fresh; fi

echo "================================================================"
echo "[run_corrector_cnisp] devices='$DEVICES' num_shards=$NUM_SHARDS"
echo "[run_corrector_cnisp] model=$MODEL steps=$STEPS casefile=$CASEFILE max_samples=$MAX_SAMPLES"
echo "[run_corrector_cnisp] logs -> $LOG_DIR/{gpu*,cpu}.log"
echo "================================================================"

# ── launch workers ───────────────────────────────────────────────────
pids=()
for i in "${!w_name[@]}"; do
    name="${w_name[$i]}"; dev="${w_dev[$i]}"; ids="${w_ids[$i]}"; th="${w_threads[$i]}"
    cuda="$(cuda_of "$dev")"
    log="$LOG_DIR/${name}.log"
    echo "  -> worker '$name' (dev=$dev CUDA='$cuda') shards=$ids -> $log"
    {
        echo "### worker=$name dev=$dev shard_ids=$ids num_shards=$NUM_SHARDS $(date)"
        echo "### cmd: $(worker_cmd "$dev" "$ids")"
    } > "$log"
    (
        cd "$CNISP_DIR"
        CUDA_VISIBLE_DEVICES="$cuda" \
        OMP_NUM_THREADS="$th" MKL_NUM_THREADS="$th" \
        PYTHONPATH=".:$REPO_ROOT:${PYTHONPATH:-}" \
        python3 "$INFER" \
            -m "$MODEL" -t "$TRAIN_YAML" -c "$TEST_YAML" \
            --checkpoint "$CHECKPOINT" \
            --test-label-source "$LABEL_SOURCE" \
            --experiment "$EXPERIMENT" \
            --test-casefile "$CASEFILE" \
            --steps "$STEPS" --max-samples "$MAX_SAMPLES" \
            --num-shards "$NUM_SHARDS" --shard-id "$ids" --skip-existing
    ) >> "$log" 2>&1 &
    pids+=("$!")
done
echo "[run_corrector_cnisp] launched ${#pids[@]} worker(s). Live: tail -f $LOG_DIR/*.log"

# ── wait + record per-worker exit codes ──────────────────────────────
rcs=()
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then rcs+=("0"); else rcs+=("$?"); fi
done

printf 'name\tdev\tids\tnum_shards\tthreads\tpid\trc\n' > "$STATUS"
for i in "${!w_name[@]}"; do
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${w_name[$i]}" "${w_dev[$i]}" "${w_ids[$i]}" "$NUM_SHARDS" \
        "${w_threads[$i]}" "${pids[$i]}" "${rcs[$i]}" >> "$STATUS"
done

# ── FINAL REVIEW (count failures in the PARENT shell, not the tee subshell) ─
n_fail=0
for i in "${!rcs[@]}"; do [[ "${rcs[$i]}" != "0" ]] && n_fail=$(( n_fail + 1 )); done
{
    echo "================ CNISP corrector run review ($(date)) ================"
    printf '%-8s %-5s %-10s %-5s %s\n' "WORKER" "DEV" "SHARDS" "RC" "STATUS"
    for i in "${!w_name[@]}"; do
        status="OK"; [[ "${rcs[$i]}" != "0" ]] && status="FAILED"
        printf '%-8s %-5s %-10s %-5s %s\n' \
            "${w_name[$i]}" "${w_dev[$i]}" "${w_ids[$i]}" "${rcs[$i]}" "$status"
    done
    masks=$(ls "$REPO_ROOT"/nnunet-c/data/cnisp_pred/*.nii.gz 2>/dev/null | wc -l | tr -d ' ')
    echo "produced masks in data/cnisp_pred: $masks"
    if [[ "$n_fail" -gt 0 ]]; then
        echo ""
        echo "FAILED workers -- re-run individually (resumable via --skip-existing):"
        for i in "${!w_name[@]}"; do
            [[ "${rcs[$i]}" == "0" ]] && continue
            echo "  # ${w_name[$i]} (see $LOG_DIR/${w_name[$i]}.log)"
            echo "  $(worker_cmd "${w_dev[$i]}" "${w_ids[$i]}")"
        done
        echo ""
        echo "Or retry ALL failed workers at once:"
        echo "  RERUN_FAILED=1 bash nnunet-c/run_corrector_cnisp.sh"
    fi
    echo "===================================================================="
} | tee "$REVIEW"

echo "[run_corrector_cnisp] status -> $STATUS ; review -> $REVIEW ; failures=$n_fail"
exit $(( n_fail > 0 ? 1 : 0 ))
