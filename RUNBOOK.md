# nnUNet-C Corrector — Master Runbook

Authoritative, detailed run instructions for the **current** corrector pipeline
(supersedes the older `nnunet-c/CASCADE_RUNBOOK.md`). Covers arms A/B/C, the
native-cascade (Route A) path for arm C, single-case testing, cleanup/reset, and
what to check at each stage. The FOV-truncation experiment has its own file:
**`RUNBOOK_FOV.md`**.

- **Arm A** = plain 835 nnUNet on the degraded CT (`Dataset835_PHOTON_CT_QAfiltered`,
  external — nothing to build/train).
- **Arm B** = stacked nnUNet-prior corrector (`Dataset855_PHOTON_CT_CORR_B_stacked`,
  5-ch image).
- **Arm C** = CNISP-prior corrector (`Dataset845_PHOTON_CT_CORR_C_cnisp`). The
  **current** C uses the **native cascade (Route A)**: image = 1-ch CT, the CNISP
  prior rides a per-case `seg_prev`, folded in by `MoveSegAsOneHot` after intensity
  aug. Its parallel prior dataset is `Dataset846_PHOTON_CT_CORR_C_cnisp_prior`.

All commands run from the repo root (`CNISP/`). `$CFG=3d_fullres`, `$PLAN=nnUNetPlansFinetune`.

---

## 0. Prerequisites (GPU/data box)

```bash
# nnUNet env (must be exported in every shell)
export nnUNet_raw=/fs5/p_masi/linz18/EyeSegmentation/nnUNet_raw
export nnUNet_preprocessed=/fs5/p_masi/linz18/EyeSegmentation/nnUNet_preprocessed
export nnUNet_results=/fs5/p_masi/linz18/EyeSegmentation/nnUNet_results
# torch.compile off by default (broken forward passes on some CUDA combos)
export nnUNet_compile=f
```
- Python venv with `nnunetv2` (2.6.3), `batchgeneratorsv2`, `blosc2`, torch, numpy,
  scipy, nibabel. The corrector modules are installed INTO nnunetv2 site-packages by
  `run_train.sh` / `run_corrector_predict.sh` (idempotent copies) — you never edit
  site-packages by hand.
- Config: `nnunet-c/configs/corrector.yaml` (identities/paths/schedule). Inspect the
  resolved env for any control with:
  ```bash
  eval "$(python3 nnunet-c/scripts/corrector_env.py --control C)"; env | grep -E 'CTRL_|CORRECTOR_|REF_|DATA_'
  ```

### Dataset identities (from corrector.yaml)
| Arm | id | name | image | prior |
|---|---|---|---|---|
| A | 835 | PHOTON_CT_QAfiltered | 1-ch CT | — (external) |
| B | 855 | PHOTON_CT_CORR_B_stacked | 5-ch | nnUNet pred (ch1–4) |
| C (cascade) | 845 | PHOTON_CT_CORR_C_cnisp | 1-ch CT | CNISP `seg_prev` (Dataset846) |
| C prior | 846 | PHOTON_CT_CORR_C_cnisp_prior | 1-ch CT | (label = CNISP prior) |
| ref | 835 | PHOTON_CT_QAfiltered | plan `nnUNetPlans`, json `nnUNetPlans_iso05`, fold 2 |

### Env-var quick reference
`run_train.sh <B|C> <fold>`:
`CASCADE`(default 1) · `SKIP_PREPROCESS`(default 1) · `PREPROCESS_NP` · `MASK_INIT`(zero) ·
`RESUME`(0/1) · `PLAN_NAME` · `CONFIG` · `CORRECTOR_TRAINER` · `CORRECTOR_EPOCHS/LR`
(from config `finetune:`) · cascade trainer knobs `CORRECTOR_PRIOR_AUG`(default 1 =
new prior-channel aug ON; `0` = stock cascade morph only) · `CORRECTOR_STRATIFIED`(1) ·
`CORRECTOR_STRATA`("3,6,9") · `CORRECTOR_JITTER_VOXELS`("4,2,2") ·
`CORRECTOR_DROP_ALL`(0.1) · `CORRECTOR_DROP_EACH`(0.25).

`run_corrector_predict.sh <B|C> <fold>`:
`CASCADE`(default 1) · `CHK`(checkpoint_best.pth) · `CNISP_CHK`(latest) · `GPUS` · `GRID`(iso) ·
`RUN_CNISP`(auto) · `EMIT_ISO`(auto) · `ISO_MM` · `BUILD_STEPS`(auto) · `REBUILD_TESTSET` ·
`FORCE` · `RUN_EVAL`(1) · `SOURCE`/`BUILD_CASEFILE` (single-case) · `RESUME_FROM_LATENT`.

---

## 1. Arm C — native cascade (the current corrected pipeline)

### Stage 0 — CNISP TRAIN prelabels (the prior source)
Generates `data/cnisp_pred_train_iso/` (iso-0.5 CNISP decode), the ch1–4 / prior:
```bash
EMIT_ISO=1 STEPS=3,6,9 bash nnunet-c/run_corrector_cnisp.sh    # gpu0+gpu1; DEVICES="0 1 cpu" to add CPU
#   RERUN_FAILED=1 bash nnunet-c/run_corrector_cnisp.sh        # retry only crashed shards
#   SKIP_EXISTING=0 ...                                        # recompute all (regenerate latents)
```
Check: `nnunet-c/data/cnisp_pred_train_iso/native_space_step_XX/*_cnisp_iso_stepXX.nii.gz` +
`manifest_by_source/<source_id>.json`; worker logs in `nnunet-c/logs/`.

### Stage 1 — degraded CTs + manifest
```bash
python nnunet-c/scripts/build_corrector_data.py --steps 3,6,9 --target-samples 200
#   --force to re-degrade existing images
```
Writes `nnunet-c/data/images/{case}_step{XX}_0000.nii.gz` +
`nnunet-c/data/corrector_data_manifest.json`.
Check: `ls nnunet-c/data/images | wc -l` and the manifest `cases[*].steps[*].kept`.

### Stage 2 — build the two cascade datasets (845 main + 846 prior)
```bash
python nnunet-c/scripts/build_corrector_dataset.py \
    --control C --layout cascade --steps 3,6,9 --max-samples 200 [--workers 8]
```
Writes `$nnUNet_raw/Dataset845_PHOTON_CT_CORR_C_cnisp/{imagesTr,labelsTr}` (1-ch CT + GT)
and `$nnUNet_raw/Dataset846_PHOTON_CT_CORR_C_cnisp_prior/{imagesTr,labelsTr}` (same CT +
CNISP prior label), each with `dataset.json` (1 image channel).
Check: `python -c "import json;print(json.load(open('$nnUNet_raw/Dataset845_PHOTON_CT_CORR_C_cnisp/dataset.json'))['channel_names'])"` → `{'0': 'CT'}`.

### Stage 3 — cascade plan + preprocess BOTH
```bash
nnUNetv2_extract_fingerprint -d 845 --verify_dataset_integrity && nnUNetv2_plan_experiment -d 845
nnUNetv2_extract_fingerprint -d 846 --verify_dataset_integrity && nnUNetv2_plan_experiment -d 846

python nnunet-c/scripts/build_finetune_plan.py --control C --cascade --out-plan-name "$PLAN"
# give 846 the SAME plan (identical geometry):
python3 - <<'PY'
import json, os
pp = os.environ["nnUNet_preprocessed"]
d = json.load(open(f"{pp}/Dataset845_PHOTON_CT_CORR_C_cnisp/nnUNetPlansFinetune.json"))
d["dataset_name"] = "Dataset846_PHOTON_CT_CORR_C_cnisp_prior"
d["configurations"]["3d_fullres"].pop("previous_stage", None)   # prior is not a cascade
json.dump(d, open(f"{pp}/Dataset846_PHOTON_CT_CORR_C_cnisp_prior/nnUNetPlansFinetune.json", "w"), indent=2)
print("wrote 846 plan")
PY
nnUNetv2_preprocess -d 845 -plans_name "$PLAN" -c "$CFG" -np 2
nnUNetv2_preprocess -d 846 -plans_name "$PLAN" -c "$CFG" -np 2
```
Check: `plan_after.json` has `configurations.3d_fullres.previous_stage = cnisp_prior`
and NO `resampling_fn_data`; the printed `overrides` list from build_finetune_plan.

### Stage 4 — relocate the prior into the seg_prev slot + gate
```bash
python nnunet-c/scripts/relocate_prevseg.py --control C --plan-name "$PLAN"   # add --move to free 846 disk
python nnunet-c/diagnostics/check_preprocessed.py --control C --plan-name "$PLAN" --cascade
```
Check: relocate prints `dest now has seg_prev for N/N main cases`; the gate prints
**PASS** (1-ch data z-scored to 835 stats, seg_prev integer {0..4} on the data grid,
labels ⊆ {0..4}).

### Stage 5 — train (installs modules, adapts 835 1→5ch, trains)
```bash
CASCADE=1 SKIP_PREPROCESS=1 \
CORRECTOR_TRAINER=nnUNetTrainer_OrbitalCascade \
bash nnunet-c/run_train.sh C 0
#   RESUME=1 ...  to continue an interrupted run (nnUNetv2_train --c)
```
Check the train log: `num_input_channels: 5`, no "seg_prev not found", and the
`[StepStratified]` loader message. Periodic snapshots land as
`$nnUNet_results/Dataset845_*/nnUNetTrainer_OrbitalCascade__${PLAN}__${CFG}/fold_0/checkpoint_epoch_XXXX.pth`.

### Stage 6 — predict + eval
```bash
CASCADE=1 CORRECTOR_TRAINER=nnUNetTrainer_OrbitalCascade CHK=checkpoint_best.pth \
bash nnunet-c/run_corrector_predict.sh C 0
```
Builds a 1-ch CT testset + `prevsegTs/` and runs
`nnUNetv2_predict … -prev_stage_predictions <prevsegTs>` then `eval_corrector.py`.
Outputs: `nnunet-c/predictions/PHOTON_CT_CORR_C_cnisp/eval_C_fold0.csv` (+ `_by_step.csv`).

### Stage 7 — refinement metrics + stratified checkpoint selection
```bash
MAP=nnunet-c/test_input/PHOTON_CT_CORR_C_cnisp/test_cases_map.json
PRED=nnunet-c/predictions/PHOTON_CT_CORR_C_cnisp/fold_0
# ASSD/HD95/NSD + volume + signed bias (reuses simulation.evaluation.surface_metrics):
python nnunet-c/diagnostics/eval_corrector.py --map $MAP --pred-dir $PRED --full-metrics

# whole-volume, step-stratified checkpoint selection over the periodic snapshots:
python nnunet-c/diagnostics/select_checkpoint.py \
    --map $MAP --images-ts nnunet-c/test_input/PHOTON_CT_CORR_C_cnisp/imagesTs \
    --dataset-id 845 --plan-name "$PLAN" --configuration "$CFG" \
    --trainer nnUNetTrainer_OrbitalCascade --fold 0 \
    --checkpoints "$nnUNet_results/Dataset845_*/*/fold_0/checkpoint_epoch*.pth" \
    --work-dir nnunet-c/predictions/_select_C --criterion stratified_mean

# fair B-vs-C on the shared (source,step) set:
python nnunet-c/diagnostics/eval_corrector.py --map $MAP --pred-dir $PRED \
    --intersect-with nnunet-c/test_input/PHOTON_CT_CORR_B_stacked/test_cases_map.json
```

---

## 2. Arm B — same cascade + aug as C (default; prior = nnUNet pred)
Arm B trains with the **identical** native-cascade + OrbitalCascade aug as C (the
default now) — only the prior source differs (nnUNet pred vs CNISP). Repeat §1
Stages 1–7 with `--control B` (datasets `855` main + `856` prior; the prior label =
the 835 nnUNet pred under `data/nnunet_pred/`, produced by the nnUNet sweep):
```bash
python nnunet-c/scripts/build_corrector_dataset.py --control B --layout cascade \
    --steps 3,6,9 --max-samples 200 --require-cnisp
# then §1 Stages 3-4 with --control B: build_finetune_plan --control B --cascade;
# copy the plan to 856; preprocess 855 + 856; relocate_prevseg --control B;
# check_preprocessed --control B --cascade
bash nnunet-c/run_train.sh B 0                 # CASCADE=1 + OrbitalCascade are the defaults
bash nnunet-c/run_corrector_predict.sh B 0
```
Legacy stacked B (old 5-ch image, no prior-aug): build WITHOUT `--layout cascade` and run
`CASCADE=0 CORRECTOR_TRAINER=nnUNetTrainer_corrector bash nnunet-c/run_train.sh B 0`
(its `check_preprocessed` then verifies ch1–4 are binary instead of seg_prev).

---

## 3. Single-case test (fast iteration)
`SOURCE=<source_id>` (or `BUILD_CASEFILE=<file>`) restricts predict to ONE image into
ISOLATED `test_input_single/` + `predictions_single/` (never clobbers the full run).
Pair with `RUN_CNISP=0` to reuse existing CNISP preds:
```bash
# cascade single-case:
CASCADE=1 CORRECTOR_TRAINER=nnUNetTrainer_OrbitalCascade \
RUN_CNISP=0 SOURCE=atlas_orbit0001_ubMask_al2_fill BUILD_STEPS=auto \
bash nnunet-c/run_corrector_predict.sh C 0

# a config variant (e.g. a rollback config):
RUN_CNISP=0 SOURCE=atlas_orbit0001_ubMask_al2_fill BUILD_STEPS=auto \
CONFIG=nnunet-c/configs/corrector_rollback_buggy_train.yaml \
bash nnunet-c/run_corrector_predict.sh C 0
```
Check: `nnunet-c/predictions_single/PHOTON_CT_CORR_C_cnisp/fold_0/*.nii.gz` + the printed
per-step Dice.

---

## 4. Cleanup / reset (force regeneration)
nnUNet preprocessing has **no resume** — to rebuild a stage you delete its outputs first.

```bash
RAW=$nnUNet_raw/Dataset845_PHOTON_CT_CORR_C_cnisp
PP=$nnUNet_preprocessed/Dataset845_PHOTON_CT_CORR_C_cnisp
PRIORPP=$nnUNet_preprocessed/Dataset846_PHOTON_CT_CORR_C_cnisp_prior
RES=$nnUNet_results/Dataset845_PHOTON_CT_CORR_C_cnisp

# re-build the raw dataset (Stage 2):        rm -rf "$RAW/imagesTr" "$RAW/labelsTr"
# re-preprocess 845 (Stage 3):               rm -rf "$PP/nnUNetPlansFinetune_3d_fullres" "$PP"/predicted_next_stage "$PP"/*.json
# re-preprocess the prior 846:               rm -rf "$PRIORPP/nnUNetPlansFinetune_3d_fullres"
# re-relocate seg_prev (Stage 4):            rm -rf "$PP"/predicted_next_stage   # then rerun relocate_prevseg
# retrain from scratch (Stage 5):            rm -rf "$RES/nnUNetTrainer_OrbitalCascade__nnUNetPlansFinetune__3d_fullres/fold_0"   # or RESUME=1 to continue
# rebuild the test inputs (Stage 6):         rm -rf nnunet-c/test_input/PHOTON_CT_CORR_C_cnisp        # or REBUILD_TESTSET=1
# re-predict (Stage 6):                      rm -rf nnunet-c/predictions/PHOTON_CT_CORR_C_cnisp       # or FORCE=1
# regenerate CNISP prelabels (Stage 0):      rm -rf nnunet-c/data/cnisp_pred_train_iso               # or SKIP_EXISTING=0
# regenerate degraded CTs (Stage 1):         rm -rf nnunet-c/data/images                             # or --force
```
Notes:
- The corrector modules in nnunetv2 site-packages are re-copied on every `run_*` — no
  need to delete them; to force a clean copy just re-run the wrapper.
- `predicted_next_stage/` MUST be repopulated (`relocate_prevseg.py`) after any 845
  re-preprocess, or training fails with "seg_prev not found".
- After deleting `imagesTr/labelsTr`, re-run Stage 2 THEN Stages 3–4 (grids change).

---

## 5. What to check (per stage)
| Stage | File / signal | Expect |
|---|---|---|
| 0 | `data/cnisp_pred_train_iso/native_space_step_XX/…` | one mask per (source, step) + manifest |
| 2 | `Dataset845…/dataset.json` `channel_names` | `{"0":"CT"}` (cascade) / 5 keys (stacked) |
| 3 | `Dataset845…/plan_after.json` | `configurations.3d_fullres.previous_stage=cnisp_prior`, no `resampling_fn_data` |
| 3 | `…/nnUNetPlansFinetune_3d_fullres/{id}.b2nd`, `{id}_seg.b2nd`, `{id}.pkl` | 600+ files (200 cases × 3 steps) |
| 4 | `check_preprocessed --cascade` | PASS: 1-ch, seg_prev {0..4} same grid, labels ⊆ {0..4} |
| 5 | train log | `num_input_channels: 5`; `checkpoint_epoch_XXXX.pth` snapshots appear |
| 6 | `predictions/…/eval_C_fold0.csv` + `_by_step.csv` | per-case + per-step Dice |
| 7 | `--full-metrics` summary | ASSD/HD95/NSD/volBias per structure |

---

## 6. Config variants
`--config nnunet-c/configs/<name>.yaml` (and `CONFIG=…` for the wrappers) switches the
whole identity/schedule set — used for rollback / buggy-mapping ablations
(`corrector_rollback*.yaml`, `corrector_*_buggy*.yaml`). The dataset ids/names,
`data_root`, `steps`, and `finetune:` schedule all come from the chosen config, so a
variant runs the identical pipeline against its own datasets/outputs.

## 7. FOV-truncation experiment → `RUNBOOK_FOV.md`.
