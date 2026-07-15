# Cascade Corrector (Route A) — build & train runbook

Native-cascade layout for arm C (Dataset845). The CNISP prior is no longer stacked
into the image (ch1–4); it becomes a per-case **`seg_prev`** that nnUNet loads at
runtime and `MoveSegAsOneHotToDataTransform` folds into the data tensor **after**
intensity augmentation — exactly mirroring nnUNet's own cascade. This fixes the
structural defect (intensity/spatial aug corrupting the binary prior) and unlocks
the Part-1 training overhaul (prior-only aug, stratified batching, stratified
checkpoint selection).

**Why this shape (resolved from `inspect_cascade_route.py` on the box):**
- `determine_num_input_channels` adds `len(foreground_labels)` purely from the
  config's `previous_stage` → **1 CT + 4 one-hot prior = 5 input channels, free**.
- `load_case` reads `seg_prev` as `folder_with_segs_from_previous_stage/{id}.b2nd`
  (a plain blosc2 seg, identical to a preprocessed `_seg.b2nd`).
- At predict time, `nnUNetv2_predict -prev_stage_predictions <folder>` reads plain
  `{caseid}.nii.gz` masks (see "Predict" at the bottom — delivered separately).

Dataset ids used below (defaults; override with the flags shown):
- main:  `Dataset845_PHOTON_CT_CORR_C_cnisp`     (1-ch CT + GT)
- prior: `Dataset846_PHOTON_CT_CORR_C_cnisp_prior` (1-ch CT + CNISP prior AS label)

Run every step from the repo root with `$nnUNet_raw/$nnUNet_preprocessed/$nnUNet_results`
exported. `$CFG` below = `3d_fullres`, `$PLAN` = `nnUNetPlansFinetune`.

---

## A. One-time data prep

### 1. Build both raw datasets (main + parallel prior), one pass
```bash
python nnunet-c/scripts/build_corrector_dataset.py \
    --control C --layout cascade \
    --steps 3,6,9 --max-samples <N> [--workers 8]
```
- Writes `Dataset845` (imagesTr `{id}_0000.nii.gz` = CT, labelsTr `{id}.nii.gz` = GT)
  and `Dataset846_..._prior` (same CT + the CNISP prior integer map `{1,2,3,4}` as
  the label). Both on the identical iso ref-grid, so their preprocessed grids match.
- `--max-samples` / `--steps` behave exactly as the stacked builder. The stratified
  loader needs cases for **each** of steps {3,6,9}, so keep all three in `--steps`.
- Prior id/name override: `--prior-dataset-id 846 --prior-dataset-name <name>`.

### 2. Fingerprint + plan BOTH datasets
```bash
nnUNetv2_extract_fingerprint -d 845 --verify_dataset_integrity
nnUNetv2_plan_experiment     -d 845
nnUNetv2_extract_fingerprint -d 846 --verify_dataset_integrity
nnUNetv2_plan_experiment     -d 846
```

### 3. Build the cascade finetune plan for the MAIN dataset
```bash
python nnunet-c/scripts/build_finetune_plan.py \
    --control C --cascade --out-plan-name "$PLAN"
```
- Merges 835 ch0 stats + spacing + arch (as before) AND sets
  `configurations.$CFG.previous_stage = cnisp_prior`, materialises the `cnisp_prior`
  mirror config, and drops the per-channel data resampler (1-ch CT → default order-3;
  the prior rides the seg resampler, order 0). Check the printed `overrides` list.

### 4. Give the PRIOR dataset the SAME plan (identical geometry)
```bash
python3 - <<'PY'
import json, os
pp = os.environ["nnUNet_preprocessed"]
main  = f"{pp}/Dataset845_PHOTON_CT_CORR_C_cnisp/nnUNetPlansFinetune.json"
prior = f"{pp}/Dataset846_PHOTON_CT_CORR_C_cnisp_prior/nnUNetPlansFinetune.json"
d = json.load(open(main))
d["dataset_name"] = "Dataset846_PHOTON_CT_CORR_C_cnisp_prior"
d["configurations"]["3d_fullres"].pop("previous_stage", None)  # prior is not a cascade
json.dump(d, open(prior, "w"), indent=2)
print("wrote", prior)
PY
```
Same `spacing` (from the 835 merge) + identical CT ⇒ identical nonzero-crop + resample
⇒ the prior's `_seg.b2nd` lands on the main data's voxel grid.

### 5. Preprocess BOTH with the plan
```bash
nnUNetv2_preprocess -d 845 -plans_name "$PLAN" -c "$CFG" [-np 2]
nnUNetv2_preprocess -d 846 -plans_name "$PLAN" -c "$CFG" [-np 2]
```
(Lower `-np` if you hit RAM/OOM. 846 is 1-channel, so it's cheap.)

### 6. Relocate the prior segs into the cascade `seg_prev` slot
```bash
python nnunet-c/scripts/relocate_prevseg.py --control C --plan-name "$PLAN"
# add --move to free the prior dataset's disk; --overwrite to re-run
```
- Reads the exact dest folder from a live `nnUNetTrainer` (ground truth), then copies
  `Dataset846/.../{id}_seg.b2nd → <dest>/{id}.b2nd`. Aborts if any main case lacks a
  prior. Prints `dest now has seg_prev for N/N main cases`.

### 7. Gate (POTHOLE-4, cascade mode)
```bash
python nnunet-c/diagnostics/check_preprocessed.py \
    --control C --plan-name "$PLAN" --cascade
```
- Expect **PASS**: data is 1-ch CT (z-scored to 835 stats), label ⊆ {0..4}, and the
  per-case `seg_prev` exists, is integer {0..4}, and matches the data grid.

---

## B. Train (reuses A; no re-preprocess)
```bash
CASCADE=1 SKIP_PREPROCESS=1 \
CORRECTOR_TRAINER=nnUNetTrainer_OrbitalCascade \
bash nnunet-c/run_train.sh C 0
```
`run_train.sh` then: installs the 4 runtime modules into nnunetv2
(`nnUNetTrainer_corrector`, `nnUNetTrainer_OrbitalCascade`, `corrector_augment`,
`corrector_stratified_loader`), runs the cascade gate, adapts the 835 checkpoint
1→5ch, and trains with `-tr nnUNetTrainer_OrbitalCascade`.

The trainer (auto, because the plan is a cascade config):
- `is_cascaded=True` → nnUNet loads `seg_prev`, one-hots it (4 ch), `num_input_channels=5`;
- `batch_size=4`, `oversample_foreground_percent=0.75`, **stratified** loader
  (1 case per step {3,6,9} + 1 bg) — toggle with `CORRECTOR_STRATIFIED=0`;
- inserts **prior-only** aug (centroid jitter + channel dropout) after the stock
  cascade morph block — `CORRECTOR_JITTER_VOXELS="4,2,2"`, `CORRECTOR_DROP_ALL=0.1`,
  `CORRECTOR_DROP_EACH=0.25`;
- snapshots `checkpoint_epoch_XXXX.pth` every `save_every` epochs for selection.

**Sanity to eyeball in the train log:** `num_input_channels: 5` and no
"seg_prev not found" errors.

---

## C. Model selection (optional, after training)
```bash
python nnunet-c/diagnostics/select_checkpoint.py \
    --map <fixed-val test_cases_map.json> --images-ts <val imagesTs> \
    --dataset-id 845 --plan-name "$PLAN" --configuration "$CFG" \
    --trainer nnUNetTrainer_OrbitalCascade --fold 0 \
    --checkpoints "$nnUNet_results/Dataset845_*/*/fold_0/checkpoint_epoch*.pth" \
    --work-dir nnunet-c/predictions/_select_C --criterion stratified_mean
```
Whole-volume, step-stratified selection over the periodic snapshots (reuses
`eval_corrector.py`). *(Val images-ts here is the cascade predict input — see below.)*

---

## D. Predict — DELIVERED SEPARATELY (next)

Under cascade the predict input is a **1-ch CT** plus a folder of **`{caseid}.nii.gz`
CNISP prior masks** passed to `nnUNetv2_predict -prev_stage_predictions <dir>`
(nnUNet preprocesses + one-hots the prior itself). This needs:
- `build_corrector_testset.py --layout cascade` (write 1-ch CT + a prevseg dir), and
- `run_corrector_predict.sh` cascade branch (1-ch build + `-prev_stage_predictions`
  + install the 4 modules).

These two edits are the remaining piece and come next; training (A+B) does not depend
on them.
