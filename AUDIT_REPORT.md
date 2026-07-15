# CNISP Repository Audit Report

**Date**: 2026-07-15
**Scope**: Full codebase audit — cascade nnUNet training strategy, FOV experiment implementation, visualization pipeline

---

## Table of Contents

1. [Repository Overview](#1-repository-overview)
2. [Cascade nnUNet Training Strategy (Detailed)](#2-cascade-nnunet-training-strategy)
3. [FOV Experiment Implementation & Training Strategy](#3-fov-experiment-implementation--training-strategy)
4. [Visualization Pipeline & Metrics](#4-visualization-pipeline--metrics)
5. [Key Observations & Notes](#5-key-observations--notes)

---

## 1. Repository Overview

CNISP is a multi-module orbital CT segmentation research codebase comparing three approaches:


| Module         | Path                       | Role                                                                             |
| -------------- | -------------------------- | -------------------------------------------------------------------------------- |
| **CNISP**      | `orbital_shape_prior_st1/` | Neural implicit shape prior (AutoDecoder + test-time latent optimization)        |
| **nnUNet**     | `nnunet/`                  | Dataset835 baseline + comparison pipeline on sparse/degraded inputs              |
| **nnUNet-C**   | `nnunet-c/`                | Corrector experiment: controls A/B/C testing whether CNISP priors improve nnUNet |
| **Simulation** | `simulation/`              | Shared degradation, metrics computation, and comparison/evaluation plotting      |




### Experimental Arms

| Arm | Dataset ID | Description | Input | Prior Source |
|-----|-----------|-------------|-------|-------------|
| A | 835 | Plain nnUNet on degraded CT | 1-ch CT (external, no training) | — |
| B (cascade) | 855 + 856 | nnUNet-prior corrector via native cascade | 1-ch CT + `seg_prev` from nnUNet pred | Dataset835 nnUNet prediction |
| C (cascade) | 845 + 846 | CNISP-prior corrector via native cascade | 1-ch CT + `seg_prev` from CNISP | CNISP iso-0.5 mm decode |

> **Change (15a5fb6, 52fb2a5)**: Arm B now uses the **same native-cascade architecture + OrbitalCascade augmentation** as Arm C. The only difference between B and C is the **prior source** (nnUNet pred vs CNISP). The legacy stacked 5-ch Arm B is still available by setting `CASCADE=0 CORRECTOR_TRAINER=nnUNetTrainer_corrector`.




### 4 Foreground Structures

`ON` (optic nerve), `Recti` (recti muscles), `Globe`, `Fat` (orbital fat) — nnUNet label scheme `{1:ON, 2:Recti, 3:Globe, 4:Fat}`.

---



## 2. Cascade nnUNet Training Strategy



### 2.1 Architecture: Native Cascade (Shared by Arm B and C)

Both Arm B and Arm C now use **nnUNet's native cascade machinery** with the **same trainer** (`nnUNetTrainer_OrbitalCascade`), the **same augmentation strategy**, and the **same stratified batching**. The only difference is the **prior source**: Arm B uses the Dataset835 nnUNet prediction, Arm C uses the CNISP iso-0.5 mm decode.

The key design (using Arm C as the example; Arm B is structurally identical):

- **Dataset 845** (main): 1-channel CT image + GT segmentation label
- **Dataset 846** (prior): same 1-channel CT + CNISP prior prediction as the label
- *(Arm B equivalently: Dataset 855 main + Dataset 856 prior with nnUNet pred as the label)*
- After preprocessing both datasets with the **same finetune plan** (`nnUNetPlansFinetune`), Dataset 846's `{id}_seg.b2nd` files are relocated into Dataset 845's `predicted_next_stage/` folder as `seg_prev` files
- nnUNet's `is_cascaded=True` then automatically:
  - Loads the per-case CNISP `seg_prev` `.b2nd`
  - Sets `num_input_channels = 1 CT + 4 one-hot prior = 5`
  - Emits `MoveSegAsOneHotToDataTransform` + stock cascade morphological augmentation
  - One-hots + vstacks the prior in whole-volume validation/predict



### 2.2 Trainer Class Hierarchy

```
nnUNetTrainer  (stock)
  └── nnUNetTrainer_corrector  (short-finetune schedule)
        └── nnUNetTrainer_OrbitalCascade  (Part 1 training overhaul)
```



#### `nnUNetTrainer_corrector` (`nnunet-c/engine/nnUNetTrainer_corrector.py`)

Only overrides `initialize()` to set the finetune schedule:

- **Epochs**: `CORRECTOR_EPOCHS` env var, default **200** (config sets **300**)
- **Initial LR**: `CORRECTOR_LR` env var, default **0.005** (config sets **0.01**)
- Uses PolyLR scheduler (inherited from stock nnUNet)



#### `nnUNetTrainer_OrbitalCascade` (`nnunet-c/engine/nnUNetTrainer_OrbitalCascade.py`)

Adds three Part-1 training changes on top of `nnUNetTrainer_corrector`:

**Change 1: Custom Prior Augmentation** (`get_training_transforms`)

- Asserts `is_cascaded=True`
- Calls stock `nnUNetTrainer.get_training_transforms` which includes:
  - Standard spatial/intensity augmentation
  - `MoveSegAsOneHotToDataTransform` (appends 4 one-hot prior channels from `seg_prev`)
  - Stock cascade morphological augmentation (`ApplyRandomBinaryOperatorTransform` + `RemoveRandomConnectedComponentFromOneHotEncodingTransform`)
- **Gated by `CORRECTOR_PRIOR_AUG`** (env var, default `"1"` = ON): when enabled, **inserts** two custom transforms BEFORE deep-supervision downsampling:
  1. `PriorCentroidJitterTransform`
  2. `PriorChannelDropoutTransform`
- When `CORRECTOR_PRIOR_AUG=0`, only the stock cascade morphological augmentation is used (for ablation experiments)

**Change 2: Step-Stratified Dataloader** (`get_dataloaders`)

- Replaces the TRAIN loader with `StepStratifiednnUNetDataLoader`
- Batch composition: 1 arbitrary (background) case + 1 case per step stratum {3,6,9}
- `batch_size = 1 + len(STRATA) = 4`
- `oversample_foreground_percent = len(STRATA) / batch_size = 0.75`
- VAL loader stays stock (whole-volume eval)
- Controlled by `CORRECTOR_STRATIFIED` env (default "1", on)

**Change 3: Periodic Checkpoint Snapshots** (`on_epoch_end`)

- Every `save_every` epochs (default 50), copies `checkpoint_latest.pth` → `checkpoint_epoch_XXXX.pth`
- Enables post-hoc checkpoint sweep by `select_checkpoint.py`



### 2.3 Custom Augmentations (Detail)



#### `PriorCentroidJitterTransform` (`corrector_augment.py`)

- Replaces accidental step-correlated centroid drift with **controlled, step-INDEPENDENT jitter**
- Same integer voxel shift δ applied to ALL 4 prior channels (preserves inter-structure spatial relationships)
- Order 0 shift with zero fill → one-hot channels stay binary
- Default `max_shift_voxels = (4, 2, 2)` (z, y, x in patch axis order)
- Configurable via `CORRECTOR_JITTER_VOXELS` env var



#### `PriorChannelDropoutTransform` (`corrector_augment.py`)

- Prevents corrector over-reliance on the shape prior
- Two modes checked in order per patch:
  1. **Full-prior dropout** (prob `p_all = 0.1`): zero ALL ch1–4 → forces image-only segmentation
  2. **Per-structure dropout** (prob `p_each = 0.25` per channel): zero individual prior channels → simulates partial CNISP failure
- ch0 (CT) and GT are never modified
- Training-time only; validation pipeline uses clean priors



### 2.4 Step-Stratified Dataloader (`corrector_stratified_loader.py`)

- Subclass of `nnUNetDataLoader`, only overrides `get_indices()`
- Parses step from preprocessed identifier `corr_{sid}_step{XX}` via regex `_step(\d+)$`
- Groups training cases by step size into `_by_step` dict
- Per batch: `[arbitrary_bg, step=strata[0], step=strata[1], ...]`
- nnUNet's foreground oversampling forces positions 1..N to foreground crops, position 0 is random-location (background)
- Default strata: `(3, 6, 9)` for thickness sweep
- Validates: `batch_size == 1 + len(strata)`, no empty strata



### 2.5 Per-Channel Resampling (`corrector_resampling.py`)

- Drop-in replacement for nnUNet's `resampling_fn_data`
- Channel 0 (CT): order 3 (cubic spline, proper intensity resampling)
- Channels 1..N (binary masks): order 0 (nearest → stays {0,1})
- Used for stacked (Arm B) datasets; cascade (Arm C) does not need it since the prior enters via `seg_prev` (resampled by the cascade pipeline's own integer resample)



### 2.6 Finetune Plan Merge (`build_finetune_plan.py`)

- Merges Dataset835's ch0 intensity stats + target spacing + architecture into the 845/855 plan
- Writes as `nnUNetPlansFinetune`
- For cascade: sets `previous_stage_name = cnisp_prior` so `is_cascaded=True` is active
- Optionally removes `resampling_fn_data` (cascade doesn't need the per-channel resampler)



### 2.7 Checkpoint Surgery (`finetune.py`)

- Adapts Dataset835 pretrained weights (1 input channel) to work with 5 input channels
- Channel 0 ← pretrained CT weights (preserves learned CT features)
- Channels 1–4 ← zeros (mask channels start as no-op → initially equivalent to single-channel nnUNet)
- Option for `small_random` init (x0.01) if zero-init causes dead gradients



### 2.8 Training Hyperparameters Summary (from `corrector.yaml`)


| Parameter | Value | Source |
|-----------|-------|--------|
| Reference dataset | 835 (PHOTON_CT_QAfiltered) | `reference_dataset_id` |
| Reference plan JSON | `nnUNetPlans_iso05` | `reference_plan_json` |
| Reference fold | 2 | `reference_fold` |
| Configuration | `3d_fullres` | `configuration` |
| Finetune trainer | `nnUNetTrainer_OrbitalCascade` (**both B and C**) | `finetune.trainer` in `corrector.yaml` |
| CASCADE default | **1** (native cascade for both B and C) | `run_train.sh` / `run_corrector_predict.sh` |
| Finetune epochs | 300 | `finetune.epochs` |
| Finetune initial LR | 0.01 | `finetune.initial_lr` |
| Degradation experiment | `thick` | `experiment` |
| Degradation steps | [3, 6, 9, 12] | `corrector_data.steps` |
| Thick threshold | 10.0 mm | `corrector_data.thick_threshold_mm` |
| Target samples | 320 | `corrector_data.target_samples` |
| Batch size (cascade) | 4 (1 bg + 3 strata) | Trainer override |
| Oversample foreground | 0.75 | Trainer override |
| Prior-channel aug | **ON** (default); `CORRECTOR_PRIOR_AUG=0` to disable | `CORRECTOR_PRIOR_AUG` env var |
| Jitter voxels | (4, 2, 2) | `CORRECTOR_JITTER_VOXELS` default |
| Full-prior dropout prob | 0.1 | `CORRECTOR_DROP_ALL` default |
| Per-channel dropout prob | 0.25 | `CORRECTOR_DROP_EACH` default |
| Checkpoint snapshot freq | every 50 epochs | `save_every` default |
| Predict grid | iso (0.5 mm) | `predict.grid` |




### 2.9 Training Pipeline (End-to-End Stage Sequence)


| Stage | Action | Key Script |
|-------|--------|------------|
| 0 | Generate CNISP TRAIN prelabels: (a) canonical-align each training case, (b) run Dataset835 nnUNet on the degraded CT → coarse 4-class seg, (c) CNISP latent optimization fits AutoDecoder to the nnUNet coarse seg, (d) dense decode at iso-0.5 mm → completed shape prior, (e) native inversion + remap to `{1,2,3,4}` | `run_corrector_cnisp.sh` → `032_cnisp_infer_corrector.py` |
| 1 | Build degraded CTs + manifest | `build_corrector_data.py` |
| 2     | Build cascade datasets (845 main + 846 prior)       | `build_corrector_dataset.py --layout cascade`               |
| 3     | Build finetune plan + preprocess both datasets      | `build_finetune_plan.py --cascade` + `nnUNetv2_preprocess`  |
| 4     | Relocate prior `seg_prev` + gate check              | `relocate_prevseg.py` + `check_preprocessed.py --cascade`   |
| 5     | Train (installs modules, adapts checkpoint, trains) | `run_train.sh C 0` with `CASCADE=1`                         |
| 6     | Predict + eval                                      | `run_corrector_predict.sh C 0` with `CASCADE=1`             |
| 7     | Refinement metrics + checkpoint selection           | `eval_corrector.py --full-metrics` + `select_checkpoint.py` |


---



## 3. FOV Experiment Implementation & Training Strategy



### 3.1 Experiment Design

The FOV-truncation experiment (Part 2, "isolate FOV") studies whether the CNISP-conditioned corrector can recover anatomy in an **imaged-but-empty** region:

- The native CT is FOV-truncated along z (a contiguous fraction blanked with air)
- **NO slice thickening** — the only degradation is the missing field of view
- The corrector must defer to the **completed CNISP prior** where ch0 has no evidence
- The CNISP prior is **re-fit to the truncated observation** (not zeroed)



### 3.2 Pseudo-Step Encoding

The truncation level is encoded as a **pseudo-step** `PP = round(keep_fraction * 100)`:


| keep_fraction | Pseudo-step | Meaning                            |
| ------------- | ----------- | ---------------------------------- |
| 0.50          | `_step50`   | Keep 50% of z-slices (most severe) |
| 0.65          | `_step65`   | Keep 65% of z-slices               |
| 0.80          | `_step80`   | Keep 80% of z-slices (mildest)     |


This encoding lets the **entire** cascade pipeline + stratified loader be reused unchanged — now stratifying by **FOV severity** instead of by **slice thickness**. Set `CORRECTOR_STRATA="50,65,80"`.

### 3.3 FOV Data Builder (`build_fov_truncated_data.py`)

- Reads an existing `corrector_data_manifest.json` (from `build_corrector_data.py`)
- For each case and each keep_fraction: calls `_truncate_one_ct()` from `nnunet/sparsify_inputs.py`
- Truncation options for `--side`:
  - `end` (default): superior cut-off
  - `start`: inferior cut-off
  - `both`: centred limited FOV
  - `random`: per-case random side selection
- Pad value: each CT's own minimum (air) by default

**Outputs**:

- `<out>/images/{case}_step{PP}_0000.nii.gz` — truncated CT
- `<out>/corrector_data_manifest.json` — same schema as thickness experiment
- `<out>/fov_truncation_manifest.json` — per-(case, PP) sidecar with:
  - `trunc_axis`: axis of truncation
  - `visible_range`: [lo, hi] slice indices of the visible region
  - `source_shape`: original CT shape
  - `keep_fraction`, `side`



### 3.4 FOV Training Strategy

The FOV experiment reuses the **entire** cascade training pipeline unchanged (Stages 2–5 from the thickness experiment), with the following config overrides:


| Parameter                  | Thickness Value          | FOV Value              |
| -------------------------- | ------------------------ | ---------------------- |
| Dataset ID                 | 845 / 846                | 847 / 848              |
| Dataset name               | `PHOTON_CT_CORR_C_cnisp` | `PHOTON_CT_CORR_C_fov` |
| `corrector_data.data_root` | `nnunet-c/data`          | `nnunet-c/data_fov`    |
| `corrector_data.steps`     | [3, 6, 9]                | [50, 65, 80]           |
| `CORRECTOR_STRATA`         | "3,6,9"                  | "50,65,80"             |
| Config file                | `corrector.yaml`         | `corrector_fov.yaml`   |


The training procedure:

1. Build truncated CTs with `build_fov_truncated_data.py`
2. Generate the CNISP prior on the truncated CTs — this is a **multi-step chain**, not a single operation:
   1. **Canonical alignment**: align each truncated CT into CNISP's canonical coordinate frame
   2. **nnUNet coarse segmentation (Dataset835, stage-1)**: run the pretrained Dataset835 nnUNet on each truncated CT to produce a coarse 4-class orbital segmentation, then canonical-align the result into per-step input patches (`labels_dataset835_step_XX/<casename>.nii.gz`)
   3. **CNISP latent optimization (032)**: the `032_cnisp_infer_corrector.py` script's `_build_label_obs_loader` loads the nnUNet coarse seg patch as the `label_obs_override` — CNISP's test-time latent optimization fits its AutoDecoder latent code to this **nnUNet-derived observation** (not the raw CT directly), producing an optimized latent per (case, step, eye)
   4. **Dense decode + native inversion**: the optimized latent is decoded at iso-0.5 mm spacing into a **completed** shape prior (the prior fills in anatomy the truncated CT cannot see), then inverted back to native space and remapped to nnUNet label scheme `{1,2,3,4}`
   - This is the same "CNISP deployment flow" used by the thickness experiment, driven by `run_corrector_cnisp.sh` with `LABEL_SOURCE=nnunet_pred` (the default); the key insight is that CNISP never sees the raw CT — it fits to the nnUNet coarse seg
3. Build cascade datasets (847 + 848) with `build_corrector_dataset.py --layout cascade`
4. Finetune plan + preprocess both
5. Relocate prior seg_prev + gate check
6. Train with `CASCADE=1 CORRECTOR_STRATA=50,65,80 CORRECTOR_TRAINER=nnUNetTrainer_OrbitalCascade`



### 3.5 FOV-Specific Evaluation

The `eval_corrector.py` script is extended with a `--region` argument for FOV experiments:


| Region          | Meaning                                                         |
| --------------- | --------------------------------------------------------------- |
| `all` (default) | Whole volume, same as thickness eval                            |
| `visible`       | Only the imaged portion (where ch0 has signal) — tests fidelity |
| `truncated`     | Only the blanked portion (where ch0 is air) — tests recovery    |


Implementation: the region mask is applied to both prediction and GT label arrays **before** every per-structure metric (Dice, surface metrics, volume), so Dice and surface metrics at the FOV cut include the cut face.

Requires `--trunc-manifest <fov_truncation_manifest.json>` to look up visible/truncated slice ranges per (source_id, step).

### 3.6 FOV Implementation Status


| Component                                    | Status                                                           |
| -------------------------------------------- | ---------------------------------------------------------------- |
| `build_fov_truncated_data.py`                | **Implemented** in repo                                          |
| `_truncate_one_ct()` in `sparsify_inputs.py` | **Implemented** in repo                                          |
| `eval_corrector.py --region` support         | **Implemented** in repo                                          |
| `CORRECTOR_STRATA` support in trainer        | **Implemented** in repo                                          |
| `corrector_fov.yaml` config                  | **NOT in repo** (planned; copy-and-modify from `corrector.yaml`) |
| `nnunet-c/data_fov/` output directory        | **NOT in repo** (generated at runtime)                           |


---



## 4. Visualization Pipeline & Metrics



### 4.1 Architecture Overview

The visualization system is split across three modules with a clear layered architecture:

```
┌──────────────────────────────────────────────────────────────────┐
│  DRIVERS (thin CLIs that wire data → aggregation → plots)       │
│  simulation/comparison/{method,paired,combined,experiment}_*    │
│  simulation/evaluation/{surface,volume_stability,volume_agree}* │
│  orbital_shape_prior_st1/engine/visualize.py                    │
├──────────────────────────────────────────────────────────────────┤
│  AGGREGATION                                                     │
│  nnunet/lib/viz.py  (aggregate_by_bucket, aggregate_paired, ...) │
│  simulation/evaluation/aggregate.py                              │
├──────────────────────────────────────────────────────────────────┤
│  RENDERING (matplotlib figures)                                  │
│  nnunet/lib/viz.py  (draw_*, plot_*, write_*)                   │
│  simulation/evaluation/plots.py  (stability_, volume_, surface_) │
├──────────────────────────────────────────────────────────────────┤
│  METRICS COMPUTATION                                             │
│  nnunet/lib/metrics.py  (native-space Dice, eff_res, label IO)   │
│  simulation/evaluation/metrics.py  (Dice, ASSD, HD95, NSD, vol) │
│  nnunet-c/diagnostics/eval_corrector.py  (per-case corrector)    │
└──────────────────────────────────────────────────────────────────┘
```



### 4.2 Metrics Computed



#### 4.2.1 Dice Coefficient

- **Binary per-structure Dice**: `2 * |P ∩ G| / (|P| + |G|)`, both-empty → 1.0
- Computed for each of the 4 foreground structures + 4-class mean
- **Implementations**:
  - `nnunet/lib/metrics.py::dice_for_source` — native-space, per-source, scheme-aware
  - `simulation/evaluation/metrics.py::compute_dice` — binary mask Dice
  - `nnunet-c/diagnostics/eval_corrector.py::_dice` — corrector eval Dice



#### 4.2.2 Surface Metrics (`simulation/evaluation/metrics.py::surface_metrics`)

- **ASSD** (Average Symmetric Surface Distance): mean of all symmetric surface distances (mm)
- **HD95** (95th-percentile Hausdorff Distance): 95th percentile of symmetric surface distances (mm)
- **NSD / Surface Dice** (Normalized Surface Dice): fraction of surface voxels within tolerance τ (default 1.0 mm)
- Uses Euclidean distance transform (`scipy.ndimage.distance_transform_edt`)
- Efficient bounding-box crop (union of pred + GT) before EDT computation
- Handles empty masks gracefully (returns NaN)



#### 4.2.3 Volume Metrics

- **Volume (mm³)**: `count_nonzero(mask) * voxel_volume`, per structure
- **Signed volume error (%)**: `100 * (V_pred - V_gt) / V_gt`
- **Volume CoV across resolutions (%)**: coefficient of variation of predicted volumes across different step sizes per (case, structure)
- **Per-scan volume range**: range of volumes across resolutions as percentage of mean
- **Lin's CCC** (Concordance Correlation Coefficient): for Bland-Altman analysis



#### 4.2.4 Effective Resolution

- `eff_res_mm = step_size * through_plane_spacing` (mm)
- Bucketed by configurable edges (default: `[1.0, 2.0, 3.0, 4.0, 5.0, 6.5, 8.5, 11.0, 13.0]` mm)
- All methods share the same eff_res bucketing for fair comparison



### 4.3 Visualization Outputs (Detailed)



#### 4.3.1 Per-Method Summary (`method_summary.py` → `nnunet/lib/viz.py`)

For EACH method (nnUNet-sparse, CNISP-atlasGT, CNISP-nnUNetPred, nnUNet-C, etc.):


| Figure                                    | Description                                     | Axes                                   |
| ----------------------------------------- | ----------------------------------------------- | -------------------------------------- |
| `{method}_recon_summary.png`              | Combined 3-subplot figure                       | —                                      |
| `{method}_overall_dice_vs_eff_res.png`    | Overall mean Dice vs eff_res                    | x=eff_res(mm), y=mean Dice             |
| `{method}_per_class_dice_vs_eff_res.png`  | Per-class (ON/Globe/Fat/Recti) Dice curves      | x=eff_res(mm), y=Dice, 4 colored lines |
| `{method}_per_case_dice_distribution.png` | Per-case Dice boxplot+scatter by eff_res bucket | x=bucket, y=per-case Dice              |


CSV/TXT outputs:

- `{method}_per_source.csv` — long format (source_id, gt_source, method, step_size, eff_res_mm, structure, dice)
- `{method}_summary_by_eff_res.csv` — aggregated (eff_res_bucket, structure, mean_dice, std_dice, n)
- `{method}_summary_by_eff_res.txt` — human-readable wide table



#### 4.3.2 Paired (Head-to-Head) Comparison (`paired_summary.py`)

Overlays nnUNet-sparse + CNISP + optional nnUNet-C on the SAME figure:


| Figure                                 | Description                                                |
| -------------------------------------- | ---------------------------------------------------------- |
| `paired_overall_dice_vs_eff_res.png`   | Multi-method overall Dice overlay                          |
| `paired_per_class_dice_vs_eff_res.png` | 2×2 grid: per-class with all methods overlaid              |
| `paired_delta_dice_vs_eff_res.png`     | Bar chart: (CNISP − nnUNet) Dice delta per bucket          |
| `paired_dice_vs_eff_res.png`           | Combined 3-row figure (all above)                          |
| `paired_summary_by_eff_res.csv`        | Bucket-by-structure table with both methods' stats + delta |




#### 4.3.3 Combined All-Methods Summary (`combined_summary.py`)

Single figure with ALL methods on one plot:


| Figure                                   | Description                            |
| ---------------------------------------- | -------------------------------------- |
| `combined_overall_dice_vs_eff_res.png`   | All methods overlaid                   |
| `combined_per_class_dice_vs_eff_res.png` | 2×2 per-class with all methods         |
| `combined_delta_dice_vs_eff_res.png`     | Delta panel (nnUNet-C − nnUNet-sparse) |
| `combined_dice_vs_eff_res.png`           | Combined 3-row master figure           |




#### 4.3.4 Cross-Experiment Summary (`experiment_summary.py`)

Overlays thin / thick / real on one figure per method:


| Figure                                       | Description                                                            |
| -------------------------------------------- | ---------------------------------------------------------------------- |
| `{method}_dice_vs_eff_res_by_experiment.png` | Per-method, thin/thick as curves, real as point                        |
| `overview_dice_vs_eff_res.png`               | Small-multiples: one subplot per method, all experiments overlaid      |
| `experiment_summary.csv` / `.txt`            | Cross-experiment table (experiment × method × structure → mean±std, n) |


Experiment styling:

- `thin`: blue, circle markers, solid line
- `thick`: red, square markers, dashed line
- `real`: green, diamond markers, point only (no connecting line)



#### 4.3.5 nnUNet-Only Native Summary (`build_nnunet_native_summary.py`)


| Figure                 | Description                                |
| ---------------------- | ------------------------------------------ |
| Native Dice vs eff_res | Overall + per-class (2-panel side by side) |
| Native Dice vs step    | Overall + per-class (2-panel side by side) |


CSV outputs: per-source, by-step, by-eff_res aggregations.

#### 4.3.6 Surface Quality Figure (`surface_quality_summary.py` → `plots.py::surface_figure`)


| Panel     | Metric              | Description                          |
| --------- | ------------------- | ------------------------------------ |
| Boxplot 1 | ASSD (mm) ↓         | Per-method boxplot across test cases |
| Boxplot 2 | HD95 (mm) ↓         | Per-method boxplot across test cases |
| Boxplot 3 | Surface Dice @1mm ↑ | Per-method boxplot across test cases |


5 methods compared: nnUNet, Cascade UNet, CNISP, Proposed, Oracle.

#### 4.3.7 Volume Stability Figure (`volume_stability_summary.py` → `plots.py::stability_figure`)


| Panel           | Metric                              | Description                                                                           |
| --------------- | ----------------------------------- | ------------------------------------------------------------------------------------- |
| (a) Bar chart   | Volume CoV across resolutions (%) ↓ | Per-structure bars for each of 5 methods, with 10% radiomics stability threshold line |
| (b) Violin plot | Per-scan volume range (ON) ↓        | Optic nerve volume wander across resolutions                                          |




#### 4.3.8 Volume Agreement Figure (`volume_agreement_summary.py` → `plots.py::volume_agreement_figure`)


| Panel | Content                        | Description                                                        |
| ----- | ------------------------------ | ------------------------------------------------------------------ |
| (a)   | Bland-Altman — nnUNet          | V_pred − V_GT vs mean, colored by thickness, with bias + LoA + CCC |
| (b)   | Bland-Altman — Proposed        | Same, for the proposed pipeline                                    |
| (c)   | Signed volume error (%) violin | Per-method violin plots                                            |




#### 4.3.9 CNISP-Specific Visualization (`visualize.py`)


| Artifact                      | Description                                                              |
| ----------------------------- | ------------------------------------------------------------------------ |
| `recon_layout.txt`            | File-tree summary of the reconstruction directory                        |
| `cross_res_dice_mean.png`     | Pairwise iso-space Dice heatmap (step_A × step_B), mean over structures  |
| `cross_res_dice_{struct}.png` | Per-structure heatmaps (ON, Globe, Fat, Recti)                           |
| `pairwise_dice_matrix.csv`    | Matrix CSV of pairwise Dice values                                       |
| `per_sample/{source_id}/`     | Per-source heatmap bundles (mean + per-structure) for outlier inspection |
| `native_sweep_summary.json`   | Per-step audit of native-space outputs vs manifest                       |




#### 4.3.10 Corrector Evaluation (`eval_corrector.py`)


| Output                            | Description                                                                                         |
| --------------------------------- | --------------------------------------------------------------------------------------------------- |
| `eval_{control}.csv`              | Per-case long CSV: case_id, source_id, step, dice_{ON,Recti,Globe,Fat}, dice_mean                   |
| `eval_{control}_by_step.csv`      | By-step aggregate: step, dice_{struct}, dice_mean, n                                                |
| Optional `--full-metrics` columns | assd_{struct}, hd95_{struct}, nsd_{struct}, vol_pred_{struct}, vol_gt_{struct}, signed_pct_{struct} |




### 4.4 Visualization Color Scheme



#### Per-Structure Colors (consistent across all figures)


| Structure        | Color              |
| ---------------- | ------------------ |
| ON (optic nerve) | `#d62728` (red)    |
| Globe            | `#1f77b4` (blue)   |
| Fat              | `#2ca02c` (green)  |
| Recti            | `#9467bd` (purple) |




#### Per-Method Colors (consistent across all paired/combined figures)


| Method           | Color              |
| ---------------- | ------------------ |
| nnUNet-sparse    | `#d62728` (red)    |
| CNISP-atlasGT    | `#1f77b4` (blue)   |
| CNISP-nnUNetPred | `#2ca02c` (green)  |
| nnUNet-C (C)     | `#ff7f0e` (orange) |
| nnUNet-C (B)     | `#9467bd` (purple) |




#### Evaluation Figure Method Colors


| Method       | Color              |
| ------------ | ------------------ |
| nnUNet       | `#d62728` (red)    |
| Cascade UNet | `#9467bd` (purple) |
| CNISP        | `#1f77b4` (blue)   |
| Proposed     | `#2ca02c` (green)  |
| Oracle       | `#7f7f7f` (gray)   |




### 4.5 Data Flow for Visualization

```
CNISP test-time optimization
  → sweep_results.pkl (canonical Dice, per-eye)
  → native_space_step_XX/*.nii.gz (whitelisted sources)

nnUNet predict on sparsified CTs
  → prediction/{exp}/sparse_step_XX_native/*.nii.gz
  → sweep_manifest.json

compare_native.py
  → paired_per_source__{run_tag}__{exp}.csv  (long: source×method×step×structure→dice)
  → paired_summary__{run_tag}__{exp}.csv/.txt

method_summary.py    → per-method PNG bundle + CSV/TXT
paired_summary.py    → head-to-head PNG bundle + CSV
combined_summary.py  → all-methods combined PNG
experiment_summary.py → cross-experiment PNG + CSV/TXT

build_mask_index.py  → MASK_INDEX JSON (5-arm pred/gt path registry)
build_metrics.py     → metrics_long.csv (surface + volume + Dice per row)
surface_quality_summary.py   → surface_quality_metrics.png
volume_stability_summary.py  → volume_stability_by_resolution.png
volume_agreement_summary.py  → volume_agreement_bland_altman.png
```

---



## 5. Key Observations & Notes



### 5.1 What is Fully Implemented and Tested

- The **entire cascade pipeline** for **both Arm B and Arm C**: dataset build, plan merge, preprocess, relocate, train, predict, eval — using the same trainer, augmentation, and stratified batching
- The legacy stacked pipeline for Arm B (5-channel image) — preserved as a fallback via `CASCADE=0`
- All custom training components: stratified loader, prior augmentations (with `CORRECTOR_PRIOR_AUG` gate), checkpoint snapshotting
- The **complete comparison/visualization pipeline**: compare_native, method/paired/combined/experiment summaries
- The **evaluation pipeline**: Dice, ASSD, HD95, NSD, volume, signed bias, volume CoV, Bland-Altman
- The **FOV data builder** and **region-restricted eval** (`--region visible|truncated`)



### 5.2 What is Planned but Not Yet in Repo


| Item                              | Status                                                      |
| --------------------------------- | ----------------------------------------------------------- |
| `corrector_fov.yaml`              | Design documented in `RUNBOOK_FOV.md`; file not yet created |
| `nnunet-c/data_fov/`              | Runtime-generated; no committed data                        |
| FOV experiment datasets (847/848) | Not yet built (requires GPU box execution)                  |




### 5.3 Design Choices Worth Noting

1. **Pseudo-step encoding for FOV**: Clever reuse of the thickness pipeline by encoding `keep_fraction` as a pseudo-step. Avoids any code duplication.
2. **Prior enters via `seg_prev`**: Both Arm B and Arm C ride nnUNet's native cascade `seg_prev` slot rather than being stacked as image channels. This means:
   - nnUNet handles one-hotting + concatenation + its own morphological aug automatically
   - The custom augmentations (jitter + dropout) are inserted cleanly into the transform chain
   - The corrector naturally supports the cascade predict path (`-prev_stage_predictions`)
3. **Eval is "pinned"**: Predictions are ALWAYS resampled onto the native GT grid (order 0); GT is NEVER resampled. This ensures A/B/C comparisons are on identical voxel grids.
4. **CNISP Dice in comparisons uses canonical-space** (from `sweep_results.pkl`, averaged per-eye), not native-space reconstructed masks. This decouples the eff_res aggregate from the `save_mask_source_ids` whitelist. Trade-off: CNISP curve is canonical-space Dice, not native-space merged-mask Dice.
5. **Source filtering**: Visualization summaries default to atlas-only (exclude `chk_*` prefixed sources), but per-source CSVs retain all sources. This keeps the human-labelled atlas cohort clean for figures while preserving the full evaluation on record.
6. **Synthetic placeholder data**: All evaluation figure drivers (`surface_quality_summary`, `volume_stability_summary`, `volume_agreement_summary`) support rendering with synthetic placeholder data when no real metrics are available, useful for layout prototyping.
7. **B-vs-C fair comparison by shared architecture**: Arm B and C sharing the same trainer/aug/batching means any Dice gap between them is attributable **solely** to the prior source quality (nnUNet pred vs CNISP), not to architectural or training-regime confounds.

---

## 6. Addendum: Arm B Cascade Unification (commits 52fb2a5, 15a5fb6)

### 6.1 Summary of Changes

Two commits unify Arm B's training path with Arm C's:

| Commit | Message | Key Changes |
|--------|---------|-------------|
| `52fb2a5` | "Arm B gets the same cascade + aug as Arm C by default" | Defaults changed: `CASCADE=1` in `run_train.sh`/`run_corrector_predict.sh`; `corrector.yaml::finetune.trainer` → `nnUNetTrainer_OrbitalCascade`; `build_corrector_dataset.py` and `build_corrector_testset.py` lift the "cascade is C-only" restriction |
| `15a5fb6` | "make the shared prior-channel aug an explicit optional flag" | Adds `CORRECTOR_PRIOR_AUG` env-var gate (default ON) to `nnUNetTrainer_OrbitalCascade.initialize()`; jitter+dropout transforms only inserted when `self._prior_aug` is True |

### 6.2 What Changed (File-by-File)

#### `nnunet-c/engine/nnUNetTrainer_OrbitalCascade.py`

- **New field `self._prior_aug`**: reads `CORRECTOR_PRIOR_AUG` env var (default `"1"`). When `"0"`, the custom jitter+dropout transforms are NOT inserted into the training pipeline — only the stock cascade morphological augmentation remains.
- **`get_training_transforms`**: the insertion of `PriorCentroidJitterTransform` + `PriorChannelDropoutTransform` is now wrapped in `if self._prior_aug:` guard.
- No other logic changed — the stratified loader, checkpoint snapshotting, and all hyperparameter defaults remain identical.

#### `nnunet-c/configs/corrector.yaml`

- `finetune.trainer` changed from `nnUNetTrainer_corrector` → **`nnUNetTrainer_OrbitalCascade`** (applies to both B and C).

#### `nnunet-c/run_train.sh`

- `CASCADE` default changed from `"0"` → **`"1"`** — native cascade is now the default for both arms.

#### `nnunet-c/run_corrector_predict.sh`

- `CASCADE` default changed from `"0"` → **`"1"`** — cascade predict is now the default for both arms.

#### `nnunet-c/scripts/build_corrector_dataset.py`

- Removed the `--layout cascade is for control C only` restriction.
- When `--layout cascade` is used with control B, the prior label reads from `prelabel_dir/{stem}.nii.gz` (the nnUNet pred, already in `{1,2,3,4}`) instead of the CNISP iso-direct decode.
- The `use_iso` check is now only enforced for control C (CNISP requires the iso-direct decode; nnUNet pred is already on the native grid in the correct label scheme).

#### `nnunet-c/scripts/build_corrector_testset.py`

- Same restriction lift: `--layout cascade` is no longer C-only.

#### `RUNBOOK.md`

- Arm B section rewritten: Arm B now follows the same §1 Stages 1–7 as C (with `--control B`), using datasets 855 (main) + 856 (prior).
- Legacy stacked B documented as a fallback: `CASCADE=0 CORRECTOR_TRAINER=nnUNetTrainer_corrector`.
- `CORRECTOR_PRIOR_AUG` added to the env-var quick reference.

### 6.3 Arm B vs C: What is Now Shared vs Different

| Aspect | Arm B | Arm C | Shared? |
|--------|-------|-------|---------|
| **Dataset layout** | 855 (main) + 856 (prior) | 845 (main) + 846 (prior) | Same structure, different IDs |
| **Image channel** | 1-ch CT | 1-ch CT | Identical |
| **Prior source** | Dataset835 nnUNet prediction | CNISP iso-0.5 mm decode | **Different** |
| **Prior entry mechanism** | `seg_prev` via native cascade | `seg_prev` via native cascade | Identical |
| **Trainer class** | `nnUNetTrainer_OrbitalCascade` | `nnUNetTrainer_OrbitalCascade` | Identical |
| **Finetune schedule** | 300 epochs, LR 0.01, PolyLR | 300 epochs, LR 0.01, PolyLR | Identical |
| **Augmentation** | Stock cascade morph + jitter + dropout | Stock cascade morph + jitter + dropout | Identical |
| **Stratified batching** | Batch 4, strata {3,6,9} | Batch 4, strata {3,6,9} | Identical |
| **Checkpoint surgery** | 835 1→5ch, ch1–4 zero-init | 835 1→5ch, ch1–4 zero-init | Identical |
| **Predict path** | `nnUNetv2_predict -prev_stage_predictions` | `nnUNetv2_predict -prev_stage_predictions` | Identical |

### 6.4 Legacy Stacked Arm B (Fallback)

The old 5-channel stacked Arm B is still accessible:

```bash
# Build WITHOUT --layout cascade (produces a single 5-ch dataset, no parallel prior)
python nnunet-c/scripts/build_corrector_dataset.py --control B --steps 3,6,9

# Train with the basic finetune trainer, no cascade
CASCADE=0 CORRECTOR_TRAINER=nnUNetTrainer_corrector bash nnunet-c/run_train.sh B 0

# Predict without cascade
CASCADE=0 bash nnunet-c/run_corrector_predict.sh B 0
```

In legacy stacked mode:
- Uses `corrector_resampling.py` (ch0 order 3, ch1–4 order 0) instead of cascade's `seg_prev` mechanism
- No jitter/dropout augmentation (only the finetune schedule override)
- `check_preprocessed` verifies ch1–4 are binary (instead of checking `seg_prev`)

### 6.5 Ablation Support via `CORRECTOR_PRIOR_AUG`

The `CORRECTOR_PRIOR_AUG` flag enables a clean ablation experiment:

| Setting | Effect | Use Case |
|---------|--------|----------|
| `CORRECTOR_PRIOR_AUG=1` (default) | Full prior-channel aug: centroid jitter + channel dropout | Production runs for both B and C |
| `CORRECTOR_PRIOR_AUG=0` | Stock cascade morphological aug only (no jitter, no dropout) | Ablation: measure the contribution of the custom prior aug |

All other training components (stratified batching, checkpoint snapshotting, cascade mechanism) remain active regardless of this flag.

---

*Report generated by automated codebase audit. Last updated: 2026-07-16 (addendum for commits 52fb2a5, 15a5fb6).*