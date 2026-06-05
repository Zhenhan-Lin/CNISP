DATASET_ID="${DATASET_ID:-835}"
DATASET_NAME="${DATASET_NAME:-PHOTON_CT_QAfiltered}"
PIVOT_CSV="${PIVOT_CSV:-/fs5/p_masi/linz18/QA_record/table_collection/PHOTON_CP_QA_pivot_table.csv}"
REVIEW_CSV="${REVIEW_CSV:-/fs5/p_masi/linz18/EyeSegmentation/nnUNet_results/Dataset805_PHOTON_MRI_Q1_CT/nnUNetTrainer_resampling_results/review_checklist.csv}"
CUSTOM_PLAN_SRC="${CUSTOM_PLAN_SRC:-/fs5/p_masi/linz18/EyeSegmentation/nnUNet_preprocessed/Dataset835_PHOTON_CT_QAfiltered/nnUNetPlans_iso05.json}"
CUSTOM_PLAN_NAME="${CUSTOM_PLAN_NAME:-nnUNetPlans_iso05}"
CFG="${CFG:-3d_fullres}"
# TRAIN_FOLDS="${TRAIN_FOLDS:-0 1 2 3 4}"
TRAIN_FOLDS="${TRAIN_FOLDS:-1 2 3 4}"
# CONTINUE=1 -> resume from checkpoint (--c). Default 0 = train from
# scratch (use this to recover a collapsed fold; resuming a bad
# checkpoint just keeps it stuck).
CONTINUE="${CONTINUE:-0}"
GPU_ID="${GPU_ID:-0}"
SKIP_BUILD="${SKIP_BUILD:-0}"
SKIP_PREPROCESS="${SKIP_PREPROCESS:-0}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DS_DIR_NAME="$(printf "Dataset%03d_%s" "${DATASET_ID}" "${DATASET_NAME}")"

# # ── Step 1-2: Preprocess ──────────────────────────────────────────
# # Dataset835 already carries its own ${CUSTOM_PLAN_NAME}.json under
# # ${nnUNet_preprocessed}/${DS_DIR_NAME}/, so we no longer copy a plan
# # in from another dataset -- just plan/preprocess this dataset.
# if [[ "$SKIP_PREPROCESS" != "1" ]]; then
#     echo -e "\n--- Step 1: Default plan_and_preprocess ---"
#     nnUNetv2_plan_and_preprocess -d "${DATASET_ID}" --verify_dataset_integrity

#     echo -e "\n--- Step 2: Copy custom plan ---"
#     cp -v "${CUSTOM_PLAN_SRC}" "${nnUNet_preprocessed}/${DS_DIR_NAME}/${CUSTOM_PLAN_NAME}.json"

#     echo -e "\n--- Step 3: Preprocess with ${CUSTOM_PLAN_NAME} ---"
#     nnUNetv2_preprocess -d "${DATASET_ID}" -plans_name "${CUSTOM_PLAN_NAME}" -c "${CFG}"
# fi

# ── Step 4: Train ─────────────────────────────────────────────────
echo -e "\n--- Step 4: Train ---"
CONT_FLAG=""
[[ "$CONTINUE" == "1" ]] && CONT_FLAG="--c"
for F in ${TRAIN_FOLDS}; do
    echo "[FOLD ${F}] nnUNetv2_train ${DATASET_ID} ${CFG} ${F} -p ${CUSTOM_PLAN_NAME} ${CONT_FLAG}"
    nnUNetv2_train "${DATASET_ID}" "${CFG}" "${F}" -p "${CUSTOM_PLAN_NAME}" ${CONT_FLAG}
done

echo -e "\n=== Done. Results: ${nnUNet_results}/${DS_DIR_NAME} ==="