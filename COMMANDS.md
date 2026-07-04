# CNISP / nnUNet-C command cheat-sheet

Concise "one need → one/few commands" reference. Run from the repo root on the
GPU/data host. All heavy steps need these exported first:

```bash
export nnUNet_raw=/fs5/p_masi/linz18/EyeSegmentation/nnUNet_raw
export nnUNet_preprocessed=/fs5/p_masi/linz18/EyeSegmentation/nnUNet_preprocessed
export nnUNet_results=/fs5/p_masi/linz18/EyeSegmentation/nnUNet_results
```

Controls: **A** = stock 835 on degraded CT (external, no train), **B** = nnUNet
prelabel corrector (Dataset855), **C** = CNISP prelabel corrector (Dataset845).

---

## nnUNet-C: training data

Generate the corrector training data (stop at 320 (case,step) samples; folder-aware):
```bash
bash nnunet-c/run_corrector_data.sh          # degrade to 320 samples + 835 predict on degraded
# build stage only / predict stage only:
bash nnunet-c/run_corrector_data.sh build
bash nnunet-c/run_corrector_data.sh predict
```

Change the target count (else edit `corrector.yaml::corrector_data.target_samples`):
```bash
python nnunet-c/scripts/build_corrector_data.py --target-samples 320
```

Control C only — align nnUNet preds → CNISP patches, then run CNISP to get ch1-4:
```bash
python nnunet-c/scripts/align_corrector_data.py     # writes corrector_train_cases.txt
bash nnunet-c/run_corrector_cnisp.sh                # CNISP infer -> data/cnisp_pred (GPU0+GPU1)
```

Build the 5-channel nnUNet raw dataset:
```bash
python nnunet-c/scripts/build_corrector_dataset.py --control C
python nnunet-c/scripts/build_corrector_dataset.py --control B --require-cnisp   # match C's case set
```

---

## nnUNet-C: finetune (200 epochs / lr 0.005 from `corrector.yaml::finetune`)

All 5 folds for one control (fold 0 does fingerprint/plan/preprocess; rest reuse it):
```bash
bash nnunet-c/run_train.sh C 0
for f in 1 2 3 4; do SKIP_PREPROCESS=1 bash nnunet-c/run_train.sh C $f; done

bash nnunet-c/run_train.sh B 0
for f in 1 2 3 4; do SKIP_PREPROCESS=1 bash nnunet-c/run_train.sh B $f; done
```

Override the schedule for one run (defaults come from config):
```bash
CORRECTOR_EPOCHS=200 CORRECTOR_LR=0.008 bash nnunet-c/run_train.sh C 0
```

---

## nnUNet-C: test / predict

Full test set, one fold (resumes: only predicts new cases; `FORCE=1` re-does all):
```bash
bash nnunet-c/run_corrector_predict.sh B 0
bash nnunet-c/run_corrector_predict.sh C 0
```
Outputs: `nnunet-c/predictions/PHOTON_CT_CORR_{B,C}_*/eval_{B,C}_fold{F}.csv`.

**Single case — diagnostics** (`nnunet-c/debugger/`, read-only, one `(sid,step)`):
```bash
# prelabel resolution walk (where do ch1..4 go empty?).
# OMIT --step to sweep ALL step_sizes present for the source; add --step N for one.
python nnunet-c/debugger/debug_test_prelabel.py --control C --grid iso \
    --sid atlas_orbit0001_ubMask_al2_fill
python nnunet-c/debugger/debug_test_prelabel.py --control C --grid iso \
    --sid atlas_orbit0001_ubMask_al2_fill --step 3
# decide the true label scheme of the CNISP iso prelabel vs GT
python nnunet-c/debugger/debug_iso_scheme.py --control C --grid iso \
    --sid atlas_orbit0001_ubMask_al2_fill --step 3
# 5ch IO / preprocessed tensor / first-conv per-channel weight norms (control B)
python nnunet-c/debugger/debug_corrector_io.py \
    --dataset-id 855 --dataset-name PHOTON_CT_CORR_B_stacked \
    --plans nnUNetPlansFinetune --config 3d_fullres \
    --trainer nnUNetTrainer_corrector --fold 0 --chk checkpoint_best.pth \
    --train-case corr_chk_14455_step03 \
    --test-images nnunet-c/test_input/PHOTON_CT_CORR_B_stacked/imagesTs \
    --test-case-id corr_chk_14455_step03 --run-predict-preproc

# same, control C (845) on the first ATLAS test case (atlas isn't a train case,
# so --train-case is omitted). Find the exact id:
#   ls nnunet-c/test_input/PHOTON_CT_CORR_C_cnisp/imagesTs | grep '^corr_atlas' \
#     | sed 's/_0000\.nii\.gz$//' | sort | head -1
python nnunet-c/debugger/debug_corrector_io.py \
    --dataset-id 845 --dataset-name PHOTON_CT_CORR_C_cnisp \
    --plans nnUNetPlansFinetune --config 3d_fullres \
    --trainer nnUNetTrainer_corrector --fold 0 --chk checkpoint_best.pth \
    --test-images nnunet-c/test_input/PHOTON_CT_CORR_C_cnisp/imagesTs \
    --test-case-id corr_atlas_orbit0001_ubMask_al2_fill_step03 \
    --run-predict-preproc
# (if the 845 model was trained with the stock trainer, use --trainer nnUNetTrainer)
```

**Single source — per-step Dice** (the actual Dice numbers, not the debug walk):
```bash
# A) re-score EXISTING masks (does NOT predict / does NOT load a checkpoint):
#    Dice reflects whatever checkpoint produced the masks in --pred-dir.
python nnunet-c/diagnostics/eval_corrector.py \
    --map nnunet-c/test_input/PHOTON_CT_CORR_C_cnisp/test_cases_map.json \
    --pred-dir nnunet-c/predictions/PHOTON_CT_CORR_C_cnisp/fold_0 \
    --source-id atlas_orbit0001_ubMask_al2_fill
# prints per-step ON/Recti/Globe/Fat/mean Dice + a "by step" summary, AND saves:
#   <pred-dir>/eval_C__atlas_orbit0001_ubMask_al2_fill.csv          (per-case)
#   <pred-dir>/eval_C__atlas_orbit0001_ubMask_al2_fill_by_step.csv  (by-step)
# override the location with --out-csv <path>.

# B) FRESH predict with the current best checkpoint, then eval (THIS one loads
#    the ckpt; A does not). SOURCE=<sid> = single-image mode: isolated *_single
#    dirs, predictions kept, Dice printed + CSV saved. Reuses the cached 5ch test
#    inputs (never rebuilds them unless REBUILD_TESTSET=1).
RUN_CNISP=0 SOURCE=atlas_orbit0001_ubMask_al2_fill BUILD_STEPS=auto \
  bash nnunet-c/run_corrector_predict.sh C 0
```
(`BUILD_STEPS=auto` = all discovered steps. `CHK` defaults to `checkpoint_best.pth`
-- use `CHK=checkpoint_final.pth` for final. Masks + Dice CSVs land under
`nnunet-c/predictions_single/<name>/`.)

Force a full rebuild/re-predict (e.g. prelabels changed):
```bash
FORCE=1 REBUILD_TESTSET=1 bash nnunet-c/run_corrector_predict.sh C 0
```

---

## Comparison + figures (nnUNet vs CNISP-v6.5-gt vs nnUNet-C, thick)

Full compare phase — paired CSVs, per-method + head-to-head figures, combined
overlay, and fig5/6/7 evaluation, all under `comparison/`:
```bash
bash run_pipeline.sh --config nnunet/configs_v6_5_gt.yaml compare
```

Just the paired CSV/summary for one CNISP run (no figures):
```bash
python simulation/comparison/compare_native.py \
    --config nnunet/configs_v6_5_gt.yaml --cnisp-run-tag corrector_gt --experiment thick
```

Just the 4-line combined figure (all methods on one plot; delta = nnUNet-C − nnUNet):
```bash
python simulation/comparison/combined_summary.py \
    --config nnunet/configs_v6_5_gt.yaml --comparison-dir comparison --experiment thick
```

The 5-pipeline evaluation figures (volume stability / agreement / surface quality).
Real data: pass `--mask-index index.json` (or set `eval_mask_index` in the config);
omit it for the synthetic illustrative layout.
```bash
# optional: build the shared metrics table once from a MASK_INDEX
python simulation/evaluation/build_metrics.py --mask-index index.json \
    --out-csv comparison/viz/evaluation__thick/metrics_long.csv
# then each figure (pass --metrics-csv, or --mask-index, or neither=synthetic)
python simulation/evaluation/volume_stability_summary.py  --out comparison/viz/evaluation__thick --mode thick
python simulation/evaluation/volume_agreement_summary.py  --out comparison/viz/evaluation__thick --mode thick
python simulation/evaluation/surface_quality_summary.py   --out comparison/viz/evaluation__thick --mode thick
```

Key comparison outputs (repo-level `comparison/`):
- `paired_per_source__<run_tag>__<exp>.csv`, `paired_summary__*.{csv,txt}`
- `viz/paired__<run_tag>__<exp>/paired_dice_vs_eff_res.png`
- `viz/combined__<exp>/combined_dice_vs_eff_res.png`
- `viz/evaluation__<exp>/{volume_stability_by_resolution,volume_agreement_bland_altman,surface_quality_metrics}.png`

---

## Troubleshooting

`RuntimeError: More than one dataset name found for dataset id NNN` — a duplicate
`DatasetNNN_*` folder (or a stray name like `..._stacked]`) exists across
`nnUNet_raw/preprocessed/results`. Keep the intended one, remove/rename the other:
```bash
ls -d "$nnUNet_raw"/DatasetNNN_* "$nnUNet_preprocessed"/DatasetNNN_* "$nnUNet_results"/DatasetNNN_*
mv "$nnUNet_results/DatasetNNN_<bad_name>" "$nnUNet_results/DatasetNNN_<good_name>"
```

**Predict result never changes / doesn't reflect the latest weights** — the nnUNet
results dir is named `<trainer>__<plan>__<config>`, so **you MUST predict with the
same `finetune.trainer` you trained with** (i.e. the same config). Mismatch =
predict silently reads an OLD checkpoint from the *other* trainer's dir:
- `corrector.yaml`      → `finetune.trainer: nnUNetTrainer_corrector` → weights in
  `Dataset845_*/nnUNetTrainer_corrector__nnUNetPlansFinetune__3d_fullres/`
- `corrector_rollback.yaml` → `finetune.trainer: nnUNetTrainer` (stock) → weights in
  `Dataset845_*/nnUNetTrainer__nnUNetPlansFinetune__3d_fullres/`

So a rollback-trained model must be predicted with the rollback config (and force a
fresh 5ch input with `REBUILD_TESTSET=1`, since the single-image input is cached):
```bash
RUN_CNISP=0 SOURCE=<sid> BUILD_STEPS=auto REBUILD_TESTSET=1 \
  CONFIG=nnunet-c/configs/corrector_rollback.yaml \
  bash nnunet-c/run_corrector_predict.sh C 0
```
Verify stage-4 prints `checkpoint=.../nnUNetTrainer__.../fold_0/checkpoint_best.pth`
(the right trainer dir) with **no** `[warn] checkpoint file not found`.
