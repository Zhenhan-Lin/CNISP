DATASET_ID="${DATASET_ID:-835}"
DATASET_NAME="${DATASET_NAME:-PHOTON_CT_QAfiltered}"
PIVOT_CSV="${PIVOT_CSV:-/home-local/linz18/eye_segmentation/table_collection/PHOTON_CP_QA/PHOTON_CP_QA_pivot_table.csv}"
REVIEW_CSV="${REVIEW_CSV:-/home-local/linz18/eye_segmentation/QA_nnUNet_CT/check_list/nnUNetTrainer_resampling_results/review_checklist.csv}"
CUSTOM_PLAN_SRC="${CUSTOM_PLAN_SRC:-/fs5/p_masi/linz18/EyeSegmentation/nnUNet_preprocessed/Dataset805_PHOTON_MRI_Q1_CT/nnUNetPlans_iso05.json}"
CUSTOM_PLAN_NAME="${CUSTOM_PLAN_NAME:-nnUNetPlans_iso05}"
CFG="${CFG:-3d_fullres}"
TRAIN_FOLDS="${TRAIN_FOLDS:-0 1 2 3 4}"
GPU_ID="${GPU_ID:-0}"
SKIP_BUILD="${SKIP_BUILD:-0}"
SKIP_PREPROCESS="${SKIP_PREPROCESS:-0}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DS_DIR_NAME="$(printf "Dataset%03d_%s" "${DATASET_ID}" "${DATASET_NAME}")"

# # ── Step 1-3: Preprocess ──────────────────────────────────────────
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
for F in ${TRAIN_FOLDS}; do
    echo "[FOLD ${F}] nnUNetv2_train ${DATASET_ID} ${CFG} ${F} -p ${CUSTOM_PLAN_NAME}"
    nnUNetv2_train "${DATASET_ID}" "${CFG}" "${F}" -p "${CUSTOM_PLAN_NAME}" --c
done

echo -e "\n=== Done. Results: ${nnUNet_results}/${DS_DIR_NAME} ==="