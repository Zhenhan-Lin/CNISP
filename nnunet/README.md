# nnUNet vs CNISP comparison

This folder collects everything that touches the new nnUNet (`Dataset835_PHOTON_CT_QAfiltered`, trained with `nnUNetPlans_iso05` via [run_nnUNet_iso05.sh](../run_nnUNet_iso05.sh)) to compare it against CNISP ([orbital_shape_prior_st1](../orbital_shape_prior_st1)) on the same 62-eye / 31-source test set.

It does **not** re-run CNISP — it reuses CNISP's per-step predictions under `output_basedir/<model>/runs/<run_tag>/`. New CNISP runs already emit `native_space_step_XX/` and `native_sweep_manifest.json` themselves (see `orbital_shape_prior_st1/engine/infer.py`); `nnunet/engine/build_cnisp_native_sweep.py` only fires for inferences that finished before that change and is otherwise a no-op.

## Option C: two CNISP runs, one nnUNet sweep

For every pipeline invocation CNISP produces **two runs** off the same model weights and the same 62-eye test set:

| run_tag       | method label       | latent-opt input            | dense Dice target                                          | story                |
| ------------- | ------------------ | --------------------------- | ---------------------------------------------------------- | -------------------- |
| `atlas_gt`    | `CNISP-atlasGT`    | sparsified canonical GT      | canonical GT                                                | ceiling curve        |
| `nnunet_pred` | `CNISP-nnUNetPred` | Dataset835 sparse-CT pred, canonical-aligned per step | atlas manual GT (atlas cases) + Dataset835 dense pred canonical-aligned (chk_* cases) | deployment curve     |

`nnUNet-sparse` (image-conditioned) is shared across both stories. Its chk_* Dice target follows the CNISP run it's being compared against (legacy `chk_pseudo` GT for `atlas_gt`; Dataset835 dense pred for `nnunet_pred`) so the head-to-head Dice is always against the same target.

The CNISP side is always GT-conditioned (latent optimization against sparse observations); nnUNet is image-conditioned. The comparison stays asymmetric — surfaced in each `paired_summary__<run_tag>.txt` header.

## Phases

- **Phase 1 (this folder)** — native-space, per-source full-head Dice (OD + OS merged on the CNISP side). GT only exists in native space, so this is the only meaningful direct Dice comparison.
- **Phase 1.5 (this folder)** — SMORE the 31 source CTs into `/fs5/p_masi/linz18/data/smore_resolved_images/`. Stand-alone; prep for Phase 2.
- **Phase 2 (deferred)** — re-predict on the SMORE'd inputs, NN-resample GT to the SMORE grid, and emit an iso-grid paired table.

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
├── engine/                                  # post-inference artifact + visualization builders
│   ├── upsample_sparse_preds.py             #   Phase 1b: NN-upsample preds to native grid
│   ├── build_cnisp_native_sweep.py          #   compare-side backfill (run_tag aware):
│   │                                        #     CNISP step_XX -> runs/<run_tag>/native_space_step_XX
│   │                                        #     (no-op when infer.py already wrote them)
│   ├── build_smore_test_images.py           #   Phase 1.5: SMORE the 31 source CTs
│   ├── build_dataset835_canonical_patches.py  # Option C: chk_* DENSE Dice-target patches
│   │                                          #   (deployment-curve GT for chk_*)
│   ├── build_dataset835_sparse_patches.py   #   Option C: per-step Dataset835 SPARSE patches
│   │                                        #     (deployment-curve latent-opt INPUT)
│   └── build_method_summary.py              #   per-method by-eff_res CSV + TXT + PNG
│
├── run_predict_native.sh           # Phase 1:  nnUNetv2_predict on original CT (step=1 baseline)
├── run_predict_sparse_sweep.sh     # Phase 1b: nnUNetv2_predict per step
├── run_predict_smore.sh            # Phase 1c: nnUNetv2_predict on SMORE'd CTs
├── run_compare.sh                  # end-to-end Phase 1 driver
└── README.md
```

## Phase 1: native-space comparison

```bash
# one shot (drives every phase, both runs):
bash run_pipeline.sh

# step-by-step (single run_tag at a time):
python nnunet/data_prep/prepare_inputs.py        --config nnunet/configs.yaml
bash   nnunet/run_predict_native.sh              # honours $CONFIG
python nnunet/engine/build_cnisp_native_sweep.py --config nnunet/configs.yaml --run-tag atlas_gt
python nnunet/compare_native.py                  --config nnunet/configs.yaml --cnisp-run-tag atlas_gt
# Repeat the last two lines with --run-tag nnunet_pred / --cnisp-run-tag nnunet_pred
# for the deployment-curve comparison (requires Option C prep below).
```

Inputs the scripts expect:

- `configs.yaml::cnisp_paths_yaml` -> `orbital_shape_prior_st1/configs/paths.yaml` (`aligned_dir`, `casefiles_dir`, `output_basedir`).
- CNISP must have already inferred the requested run: `output_basedir/<model_name>/runs/<run_tag>/sweep_results.pkl` and `runs/<run_tag>/step_XX/pred/*.nii.gz` exist.
  - Recent inferences also have `runs/<run_tag>/native_space_step_XX/` + `runs/<run_tag>/native_sweep_manifest.json`; `compare_native.py` consumes them directly.
  - Legacy inferences only have `step_XX/`; `engine/build_cnisp_native_sweep.py --run-tag <T>` reconstructs `runs/<T>/native_space_step_XX/` from `sweep_results.pkl`.
- CT image discovery is driven by `atlas_image_dir` and `pivot_csv`; missing CTs fail loudly with a per-source list.

Outputs (under `configs.yaml::work_dir`, one set per `cnisp_runs_to_compare` entry):

- `nnunet_input/<source_id>_0000.nii.gz` — symlink, channel-0 named for nnUNetv2.
- `source_to_path.json` — for downstream traceability.
- `prediction/native/<source_id>.nii.gz` — fold-0 prediction, native CT spacing (step=1 dense baseline).
- `comparison/paired_per_source__<run_tag>.csv` — long: `(source_id, gt_source, method, step_size, eff_res_mm, structure, dice)`. `method` is `nnUNet-sparse` plus the CNISP method label for this run (e.g. `CNISP-atlasGT` or `CNISP-nnUNetPred`).
- `comparison/paired_summary__<run_tag>.csv` — long aggregate: `(bucket, structure, mean_dice, std_dice, n_sources)` where `bucket` is `nnUNet-sparse (lo, hi]` or `<cnisp_method_label> (lo, hi]` per eff-res bucket. The two methods appear side-by-side per bucket.
- `comparison/paired_summary__<run_tag>.txt` — pretty-printed table with the asymmetry / OOD / chk_* GT-mode caveats.

Outputs (CNISP side, under `cnisp_paths.output_basedir/<model_name>/runs/<run_tag>/`):

- `native_space_step_XX/<original_stem>_cnisp_stepXX.nii.gz` — full-head OD+OS merge per CNISP sweep step.
- `native_space_step_XX/manifest.json` — `(source_id -> nifti path)` map (also records `run_tag` + `test_label_source`).
- `native_sweep_manifest.json` — summary across all steps (used by `compare_native.py` to decide which chk_* GT to use).

For `comparison/paired_per_source__atlas_gt.csv`, `chk_*` rows carry `gt_source=chk_pseudo` (legacy QA-kept pseudo-GT). For `comparison/paired_per_source__nnunet_pred.csv`, `chk_*` rows carry `gt_source=chk_pseudo_dataset835` (Dataset835's dense pred). Filter on `gt_source=='atlas'` for the manual-GT-only view in either file.

### Per-method by-eff_res summary bundle

For each run_tag, `compare` calls `engine/build_method_summary.py` once per method. Same script, same `comparison/paired_per_source__<run_tag>.csv` -> the nnUNet and CNISP halves of that comparison get a matched bundle:

- `${work_dir}/comparison/viz/nnUNet-sparse__<run_tag>/nnUNet-sparse_*` (per-source CSV / summary CSV / TXT / PNG)
- `${cnisp_output_basedir}/<model>/viz/<run_tag>/<cnisp_method_label>_*` (same four artifacts)

Each `_recon_summary.png` is three stacked subplots:

1. Overall mean Dice vs effective resolution (errorbar over sources in each bucket).
2. Per-class Dice vs effective resolution (ON / Globe / Fat / Recti).
3. Per-case mean-Dice boxplot per eff_res bucket.

CNISP's old `recon_summary.png` from `cnisp-viz` plotted a separate "observed-only" line; that line is intentionally **omitted** here because `comparison/paired_per_source__<run_tag>.csv` only carries dense Dice. Keeping both methods on the same single-curve layout makes side-by-side comparison honest.

## Option C deployment-curve prep

These two phases produce the canonical-aligned Dataset835 artifacts CNISP needs to render the deployment curve (`run_tag=nnunet_pred`). Both are stamped into the `cnisp-prep-dataset835-*` phases of `run_pipeline.sh`, but you can also drive them by hand:

```bash
# chk_* DENSE Dice target + sidecar metadata
python nnunet/engine/build_dataset835_canonical_patches.py --config nnunet/configs.yaml
# per-step SPARSE latent-opt input (step_01 falls back to the dense pred above)
python nnunet/engine/build_dataset835_sparse_patches.py    --config nnunet/configs.yaml
```

Outputs (under `${cnisp_paths.aligned_dir}`):

- `labels_dataset835/<casename>.nii.gz` — canonical-aligned Dataset835 dense pred (chk_* dense Dice target in deployment mode; atlas patches written too for symmetric step-01 input).
- `metadata_dataset835/<casename>.json` — sidecar so `native_mapping.invert_alignment_single_eye` can place chk_* CNISP predictions back into the source's native head volume.
- `labels_dataset835_step_{XX}/<casename>.nii.gz` — Dataset835 sparse-CT pred canonical-aligned per step. CNISP's latent-opt input in deployment mode.

When nnUNet at high sparsity drops a globe entirely the canonical-align step refuses to write that eye / step; `engine/infer.py` logs and skips the missing rows so the deployment curve surfaces nnUNet's failure rather than papering over it.

## Phase 1b: native-space sparse-CT sweep

Mirrors CNISP's per-step inference on the nnUNet side: feed nnUNet a sparsified copy of each source CT (drop every Nth axial slice along the through-plane axis), then NN-upsample the prediction back to the original native grid for Dice. The `(source_id, step_size)` set is read directly from `${cnisp_output_basedir}/<model>/runs/atlas_gt/sweep_results.pkl` (override with `--cnisp-sweep-source <run_tag>` on `sparsify_inputs.py` if you'd rather track a different CNISP run's sweep). Both Option C stories then re-use the same nnUNet sparse-CT predictions, so the two methods cannot drift out of sync across runs.

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
- `prediction/sparse_step_XX/<sid>.nii.gz` — nnUNet output at the sparse CT's spacing.
- `prediction/sparse_step_XX_upsampled/<sid>.nii.gz` — same mask NN-upsampled back to the dense native CT grid. step_01 is a symlink to `prediction/native/<sid>.nii.gz`.
- `prediction/sweep_manifest.json` — `{steps: {XX: {sid: path}}}`, consumed by `compare_native.py`.

Caveats:

- The nnUNet plan was trained at iso 0.5 mm, so large `step_size` rows (z-spacing up to ~11 mm) are intentionally out-of-distribution. That OOD curve is the whole point of this phase.
- `data_prep/sparsify_inputs.py` picks the sparsification axis from the raw CT's affine (not from `argmax(zooms)`), so non-axial acquisitions are degraded along the same physical direction CNISP did. Two checks then gate every write:
  1. **Axis selection + obliqueness check** (per source): for each source, the script reads CNISP's per-row `step_axis` field from `sweep_results.pkl` to know which RAS direction CNISP sparsified that source along (a single int for legacy `slice_step_axis: <int>` runs, a per-source value for `slice_step_axis: auto` runs). It then picks the raw CT voxel axis whose physical direction best aligns with that RAS direction via `argmax(|affine[ras_axis, :3]|)`. For an axial CT under the default RAS axis 2 (S-I) this is the thick S-I axis; for a sagittal CT (under either `auto` mode or a legacy uniform S-I config) it's the thin in-plane axis pointing S-I (not the thick L-R one). If CNISP's sweep predates the per-row `step_axis` write-out, the fallback is the `cnisp_slice_step_axis` config knob (default 2). If the best alignment is below `sparse_axis_alignment_min` (default 0.95) the voxel grid is too oblique to RAS for any single-axis sparsification to be physically meaningful, and the source is dropped.
  2. **Magnitude check** (per source × step): the script compares `zooms[step_axis] * step` against CNISP's `effective_resolution_mm`. Differences within `sparse_eff_res_tolerance` (default 5 %) are silently OK; differences within `sparse_eff_res_max_drift` (default 30 %) are warned about but still written (these typically come from CNISP's canonical patch living on a different grid than the raw CT — e.g. `chk_*` QA-kept old-nnUNet preds on a ~1.25 mm iso grid); differences above the drift cap are dropped.
- The resulting `nnunet_input_sparse_manifest.json` lists only the surviving `(source, step)` pairs. Downstream phases (`run_predict_sparse_sweep.sh`, `upsample_sparse_preds.py`, `build_dataset835_sparse_patches.py`, `compare_native.py`) iterate over the manifest, so dropped sources/steps naturally yield fewer rows in the final paired CSVs without breaking anything else.

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
- `prediction/smore/<sid>.nii.gz` — nnUNet prediction on the SMORE grid.

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
| `cnisp_runs_to_compare` | list of `{run_tag, method_label}` entries; `compare` renders one paired bundle per entry. Defaults to atlas_gt + nnunet_pred. |
| `deployment_gt_dirname_for_chk` | chk_* GT directory under `${work_dir}` when a run uses `test_label_source=nnunet_pred` (slashes OK for subdirs). Default `prediction/native`. |
| `atlas_image_dir` | where atlas CT images live (sibling to atlas_label_dir from CNISP's paths.yaml) |
| `pivot_csv`, `pivot_image_path_columns` | PHOTON pivot table for `chk_*` CT discovery |
| `work_dir` | Phase 1 / 1b / 1c working dir |
| `smore_out_root`, `smore_suffix` | Phase 1.5 SMORE shared output (flat layout: `<smore_out_root>/<sid><smore_suffix>.nii.gz`) |
| `cnisp_slice_step_axis` | Phase 1b fallback canonical RAS direction (default 2 = S-I); used only when `sweep_results.pkl` rows don't carry a per-row `step_axis` field. New CNISP runs (per-case or uniform) always emit this field, so the knob mainly matters for legacy sweeps. |
| `sparse_axis_alignment_min` | Phase 1b minimum projection (default 0.95) of the chosen voxel axis onto the canonical RAS direction; sources with oblique grids below this are dropped |
| `sparse_eff_res_tolerance` | Phase 1b soft eff-res tolerance (relative); default 0.05 — drifts below this are silent |
| `sparse_eff_res_max_drift` | Phase 1b hard eff-res cap (relative); default 0.30 — drifts above this drop the (source, step) |
| `summary_bucket_edges_mm` | eff-res buckets, inherited from `test_default.yaml` so reports line up with CNISP's `test_results.csv` `eff_res_bucket` column |

In `orbital_shape_prior_st1/configs/paths.yaml` (Option C staging dirs, default to safe names — only override if you renamed the subtrees):

| key | meaning |
| --- | --- |
| `labels_dataset835_dirname` | chk_* DENSE Dice-target patches (default `labels_dataset835`) |
| `metadata_dataset835_dirname` | sidecar metadata for the above (default `metadata_dataset835`) |
| `labels_dataset835_step_prefix` | per-step SPARSE latent-opt input prefix (default `labels_dataset835_step_`; subdirs become `labels_dataset835_step_03`, ...) |

CLI flags override the corresponding yaml fields when set.
