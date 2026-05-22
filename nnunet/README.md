# nnUNet vs CNISP comparison

This folder collects everything that touches the new nnUNet (`Dataset835_PHOTON_CT_QAfiltered`, trained with `nnUNetPlans_iso05` via [run_nnUNet_iso05.sh](../run_nnUNet_iso05.sh)) to compare it against CNISP ([orbital_shape_prior_st1](../orbital_shape_prior_st1)) on the same 62-eye / 31-source test set.

It does **not** re-run CNISP — it reuses CNISP's existing per-step canonical-patch predictions (`output_basedir/<model>/step_XX/`) and `sweep_results.pkl`. New CNISP runs already emit `native_space_step_XX/` and `native_sweep_manifest.json` themselves (see `orbital_shape_prior_st1/engine/infer.py`); the per-step backfill script here only fires for inferences that finished before that change and is otherwise a no-op.

## Phases

- **Phase 1 (this folder)** — native-space, per-source full-head Dice (OD + OS merged on the CNISP side). GT only exists in native space, so this is the only meaningful direct Dice comparison.
- **Phase 1.5 (this folder)** — SMORE the 31 source CTs into `/fs5/p_masi/linz18/data/smore_resolved_images/`. Stand-alone; prep for Phase 2.
- **Phase 2 (deferred)** — re-predict on the SMORE'd inputs, NN-resample GT to the SMORE grid, and emit an iso-grid paired table.

The CNISP side is GT-conditioned (sparse-slice latent optimization, see [test_default.yaml](../orbital_shape_prior_st1/configs/test_default.yaml)); nnUNet is image-conditioned. The comparison is informative but not symmetric — surfaced in the report header.

## Layout

```
nnunet/
├── configs.yaml                    # shared paths / inference knobs
├── resolve_gt.py                   # 31-source <-> CT/GT/scheme resolver
├── compare_native.py               # paired full-head per-step Dice tables
├── helpers/
│   └── smore.py                    # SMORE compat-check, local + container backends
│
├── data_prep/                      # stage inputs for nnUNetv2_predict
│   ├── prepare_inputs.py           #   Phase 1:  symlink CTs into nnunet_input/
│   ├── sparsify_inputs.py          #   Phase 1b: sparsified CTs (1:1 with CNISP sweep)
│   └── prepare_smore_inputs.py     #   Phase 1c: symlink SMORE'd CTs
│
├── engine/                         # post-inference artifact + visualization builders
│   ├── upsample_sparse_preds.py    #   Phase 1b: NN-upsample preds to native grid
│   ├── build_cnisp_native_sweep.py #   compare-side backfill: CNISP step_XX -> native_space_step_XX
│   │                               #     (no-op when orbital_shape_prior_st1/engine/infer.py already wrote them)
│   ├── build_smore_test_images.py  #   Phase 1.5: SMORE the 31 source CTs
│   └── build_method_summary.py     #   per-method by-eff_res CSV + TXT + PNG (CNISP and nnUNet)
│
├── run_predict_native.sh           # Phase 1:  nnUNetv2_predict on original CT (step=1 baseline)
├── run_predict_sparse_sweep.sh     # Phase 1b: nnUNetv2_predict per step
├── run_predict_smore.sh            # Phase 1c: nnUNetv2_predict on SMORE'd CTs
├── run_compare.sh                  # end-to-end Phase 1 driver
└── README.md
```

## Phase 1: native-space comparison

```bash
# one shot
bash nnunet/run_compare.sh

# or step-by-step:
python nnunet/data_prep/prepare_inputs.py        --config nnunet/configs.yaml
bash   nnunet/run_predict_native.sh              # honours $CONFIG
python nnunet/engine/build_cnisp_native_sweep.py --config nnunet/configs.yaml   # no-op if infer.py already produced native_space_step_XX/
python nnunet/compare_native.py                  --config nnunet/configs.yaml
```

Inputs the scripts expect:

- `configs.yaml::cnisp_paths_yaml` -> `orbital_shape_prior_st1/configs/paths.yaml` (`aligned_dir`, `casefiles_dir`, `output_basedir`).
- CNISP must have already run inference: `output_basedir/<model_name>/sweep_results.pkl` and `step_XX/pred/*.nii.gz` exist.
  - Recent inferences also have `native_space_step_XX/` + `native_sweep_manifest.json`; `compare_native.py` will consume them directly.
  - Legacy inferences only have `step_XX/`; in that case `engine/build_cnisp_native_sweep.py` reconstructs `native_space_step_XX/` from `sweep_results.pkl`.
- CT image discovery is driven by `atlas_image_dir` and `pivot_csv`; missing CTs fail loudly with a per-source list.

Outputs (under `configs.yaml::work_dir`):

- `nnunet_input/<source_id>_0000.nii.gz` — symlink, channel-0 named for nnUNetv2.
- `source_to_path.json` — for downstream traceability.
- `nnunet_pred_native/<source_id>.nii.gz` — fold-0 prediction, native CT spacing (step=1 dense baseline).
- `paired_per_source.csv` — long: `(source_id, gt_source, method, step_size, eff_res_mm, structure, dice)`. Each method now contributes one row per (source, step, structure).
- `paired_summary.csv` — long aggregate: `(bucket, structure, mean_dice, std_dice, n_sources)` where `bucket` is `nnUNet (lo, hi]` or `CNISP (lo, hi]` per eff-res bucket from `summary_bucket_edges_mm`. nnUNet and CNISP appear side-by-side per bucket.
- `paired_summary.txt` — pretty-printed table with the asymmetry / OOD / pseudo-GT caveats.

Outputs (CNISP side, under `cnisp_paths.output_basedir/<model_name>/`):

- `native_space_step_XX/<original_stem>_cnisp_stepXX.nii.gz` — full-head OD+OS merge per CNISP sweep step.
- `native_space_step_XX/manifest.json` — `(source_id -> nifti path)` map.
- `native_sweep_manifest.json` — summary across all steps.

The `chk_*` rows in `paired_per_source.csv` carry `gt_source=chk_pseudo`; filter to `gt_source=='atlas'` for the manual-GT-only view.

### Per-method by-eff_res summary bundle

`compare_native.py` finishes by running `engine/build_method_summary.py` once for each method. Same script, same `paired_per_source.csv` -> nnUNet and CNISP get a matched bundle:

- `${work_dir}/viz/nnUNet/nnUNet_per_source.csv | _summary_by_eff_res.csv | .txt | _recon_summary.png`
- `${cnisp_output_basedir}/<model>/viz/CNISP_per_source.csv | _summary_by_eff_res.csv | .txt | _recon_summary.png`

Each `_recon_summary.png` is three stacked subplots:

1. Overall mean Dice vs effective resolution (errorbar over sources in each bucket).
2. Per-class Dice vs effective resolution (ON / Globe / Fat / Recti).
3. Per-case mean-Dice boxplot per eff_res bucket.

CNISP's old `recon_summary.png` from `cnisp-viz` plotted a separate "observed-only" line; that line is intentionally **omitted** here because `paired_per_source.csv` only carries dense Dice. Keeping both methods on the same single-curve layout makes side-by-side comparison honest.

## Phase 1b: native-space sparse-CT sweep

Mirrors CNISP's per-step inference on the nnUNet side: feed nnUNet a sparsified copy of each source CT (drop every Nth axial slice along the through-plane axis), then NN-upsample the prediction back to the original native grid for Dice. The `(source_id, step_size)` set is read directly from `${cnisp_output_basedir}/<model>/sweep_results.pkl`, so the two methods cannot drift out of sync across runs.

> **What "sparsified" means here:** the CT is **subsampled along the through-plane axis** — every Nth slice is kept verbatim and the rest are dropped, then the NIfTI affine's through-plane column is multiplied by `step_size` so the header reports the new (coarser) spacing. **No interpolation, no super-resolution.** This is distinct from Phase 1c, which feeds nnUNet SMORE-super-resolved CTs (every voxel is a neural-network output). CNISP's sweep, in contrast, sparsifies the *GT label* (not the CT image), so on a given `(source, step)` row the two methods consume strictly aligned "information content" (same kept-slice indices, same eff_res) from two different modalities.

```bash
# orchestrated by run_pipeline.sh's `nnunet-predict-sweep` phase, or:
python nnunet/data_prep/sparsify_inputs.py     --config nnunet/configs.yaml
bash   nnunet/run_predict_sparse_sweep.sh
python nnunet/engine/upsample_sparse_preds.py  --config nnunet/configs.yaml
```

Pipeline-level prereqs: `nnunet-predict` (provides the step_01 dense baseline that the upsample step symlinks) and `cnisp-infer` (provides `sweep_results.pkl`).

Outputs (under `${work_dir}`):

- `nnunet_input_step_XX/<sid>_0000.nii.gz` — sparsified CT; the affine's through-plane column is scaled by step_size, the other two columns and the origin are untouched.
- `nnunet_input_sparse_manifest.json` — `{step_axis_per_source, by_step: {XX: {sid: {input, eff_res_mm, step_axis}}}}`.
- `nnunet_pred_native_step_XX/<sid>.nii.gz` — nnUNet output at the sparse CT's spacing.
- `nnunet_pred_native_step_XX_upsampled/<sid>.nii.gz` — same mask NN-upsampled back to the dense native CT grid. step_01 is a symlink to `nnunet_pred_native/<sid>.nii.gz`.
- `nnunet_pred_native_sweep_manifest.json` — `{steps: {XX: {sid: path}}}`, consumed by `compare_native.py`.

Caveats:

- The nnUNet plan was trained at iso 0.5 mm, so large `step_size` rows (z-spacing up to ~11 mm) are intentionally out-of-distribution. That OOD curve is the whole point of this phase.
- `data_prep/sparsify_inputs.py` asserts `spacing[argmax_axis] * step ≈ CNISP eff_res` within `sparse_eff_res_tolerance` (default 5 %). If a CT was acquired sagittally / coronally and its largest-spacing voxel axis is not the through-plane axis, the script refuses to write rather than sparsify the wrong direction.

## Phase 1c: nnUNet on SMORE'd CTs

Independent of Phase 1 and Phase 1b. Saves the prediction mask only; downstream analysis is TBD.

```bash
# orchestrated by run_pipeline.sh's `nnunet-predict-smore` phase, or:
python nnunet/data_prep/prepare_smore_inputs.py --config nnunet/configs.yaml
bash   nnunet/run_predict_smore.sh
```

Prereq: run [Phase 1.5](#phase-15-smore-prep) first; this phase only consumes the existing `${smore_out_root}/<sid>${smore_suffix}.nii.gz` files. Sources without a SMORE output are skipped with a warning; the phase exits 2 only when zero sources are stageable, so the pipeline visibly fails instead of silently producing an empty output dir.

Outputs (under `${work_dir}`):

- `nnunet_input_smore/<sid>_0000.nii.gz` — symlink to the canonical SMORE'd CT.
- `nnunet_pred_smore/<sid>.nii.gz` — nnUNet prediction on the SMORE grid.

## Phase 1.5: SMORE prep

Independent of Phase 1; runs in parallel:

```bash
# local backend (host run-smore in PATH)
python nnunet/engine/build_smore_test_images.py \
    --config nnunet/configs.yaml \
    --smore-gpu-ids 0,1 --smore-per-gpu-concurrency 1

# container backend
python nnunet/engine/build_smore_test_images.py \
    --config nnunet/configs.yaml \
    --smore-backend container \
    --smore-sif /path/to/smore.sif \
    --smore-gpu-ids 0
```

Reuses helpers from [nnunet/nnunetv2_build_datasets2.py](nnunetv2_build_datasets2.py): compatibility check, local + container backends, per-GPU concurrency, and the multi-machine `mkdir`-based claim/lock.

Output layout under `configs.yaml::smore_out_root` (default `/fs5/p_masi/linz18/data/smore_resolved_images/`):

```
<source_id>_smore.nii.gz                     # SR output (canonical path);
                                             #   symlink to original CT for
                                             #   SMORE-incompatible cases.
_artifacts/<source_id>/                      # only for SMORE'd cases
    run_smore.log                            # SMORE stdout/stderr (per-case)
    weights/best_weights.pt
build_smore_test_images.<host>.<pid>.tsv
```

`<source_id>_smore.nii.gz` at the flat root is the canonical path downstream code reads. SMORE's own per-subject subdir + the staging `_src/` are torn down by the worker after each successful run; per-case `run_smore.log` and training weights are kept under `_artifacts/<source_id>/` (so `ls` of the flat root shows just the SR files).

If a CT is already iso (rare for clinical CT), the compatibility check fails; by default we pass the original through under the same `_smore` filename (no `_artifacts/` entry created) so downstream code stays uniform. Override with `--smore-on-incompatible skip`.

## Config knobs you'll most often touch

In `nnunet/configs.yaml`:

| key | meaning |
| --- | --- |
| `dataset_id`, `plan`, `configuration`, `trainer`, `folds`, `gpu_id` | nnUNetv2 predict identity, must match training |
| `cnisp_paths_yaml`, `cnisp_model_name` | which CNISP run to compare against |
| `atlas_image_dir` | where atlas CT images live (sibling to atlas_label_dir from CNISP's paths.yaml) |
| `pivot_csv`, `pivot_image_path_columns` | PHOTON pivot table for `chk_*` CT discovery |
| `work_dir` | Phase 1 / 1b / 1c working dir |
| `smore_out_root`, `smore_suffix` | Phase 1.5 SMORE shared output (flat layout: `<smore_out_root>/<sid><smore_suffix>.nii.gz`) |
| `sparse_eff_res_tolerance` | Phase 1b axis-detection tolerance (relative); default 0.05 |
| `summary_bucket_edges_mm` | eff-res buckets, inherited from `test_default.yaml` so reports line up with CNISP's `test_results.csv` `eff_res_bucket` column |

CLI flags override the corresponding yaml fields when set.
