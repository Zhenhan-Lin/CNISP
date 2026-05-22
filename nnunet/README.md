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
├── prepare_inputs.py               # symlink CTs into nnUNet_input/
├── run_predict_native.sh           # nnUNetv2_predict on original CT
├── build_cnisp_native_sweep.py     # backfill: CNISP step_XX -> native_space_step_XX
│                                   #   (no-op when infer.py already wrote them)
├── compare_native.py               # paired full-head Dice tables
├── build_smore_test_images.py      # Phase 1.5: SMORE the 31 source CTs
├── nnunetv2_build_datasets2.py     # (training-side) -- unchanged
├── run_compare.sh                  # end-to-end Phase 1 driver
└── README.md
```

## Phase 1: native-space comparison

```bash
# one shot
bash nnunet/run_compare.sh

# or step-by-step:
python nnunet/prepare_inputs.py             --config nnunet/configs.yaml
bash   nnunet/run_predict_native.sh         # honours $CONFIG
python nnunet/build_cnisp_native_sweep.py   --config nnunet/configs.yaml   # no-op if infer.py already produced native_space_step_XX/
python nnunet/compare_native.py             --config nnunet/configs.yaml
```

Inputs the scripts expect:

- `configs.yaml::cnisp_paths_yaml` -> `orbital_shape_prior_st1/configs/paths.yaml` (`aligned_dir`, `casefiles_dir`, `output_basedir`).
- CNISP must have already run inference: `output_basedir/<model_name>/sweep_results.pkl` and `step_XX/pred/*.nii.gz` exist.
  - Recent inferences also have `native_space_step_XX/` + `native_sweep_manifest.json`; `compare_native.py` will consume them directly.
  - Legacy inferences only have `step_XX/`; in that case `build_cnisp_native_sweep.py` reconstructs `native_space_step_XX/` from `sweep_results.pkl`.
- CT image discovery is driven by `atlas_image_dir` and `pivot_csv`; missing CTs fail loudly with a per-source list.

Outputs (under `configs.yaml::work_dir`):

- `nnunet_input/<source_id>_0000.nii.gz` — symlink, channel-0 named for nnUNetv2.
- `source_to_path.json` — for downstream traceability.
- `nnunet_pred_native/<source_id>.nii.gz` — fold-0 prediction, native CT spacing.
- `paired_per_source.csv` — long: `(source_id, gt_source, method, step_size, eff_res_mm, structure, dice)`.
- `paired_summary.csv` — long aggregate: `(bucket, structure, mean_dice, std_dice, n_sources)` where `bucket` is `nnUNet` or `CNISP (lo, hi]` per eff-res bucket from `summary_bucket_edges_mm`.
- `paired_summary.txt` — pretty-printed table with the asymmetry / pseudo-GT caveats.

Outputs (CNISP side, under `cnisp_paths.output_basedir/<model_name>/`):

- `native_space_step_XX/<original_stem>_cnisp_stepXX.nii.gz` — full-head OD+OS merge per CNISP sweep step.
- `native_space_step_XX/manifest.json` — `(source_id -> nifti path)` map.
- `native_sweep_manifest.json` — summary across all steps.

The `chk_*` rows in `paired_per_source.csv` carry `gt_source=chk_pseudo`; filter to `gt_source=='atlas'` for the manual-GT-only view.

## Phase 1.5: SMORE prep

Independent of Phase 1; runs in parallel:

```bash
# local backend (host run-smore in PATH)
python nnunet/build_smore_test_images.py \
    --config nnunet/configs.yaml \
    --smore-gpu-ids 0,1 --smore-per-gpu-concurrency 1

# container backend
python nnunet/build_smore_test_images.py \
    --config nnunet/configs.yaml \
    --smore-backend container \
    --smore-sif /path/to/smore.sif \
    --smore-gpu-ids 0
```

Reuses helpers from [nnunet/nnunetv2_build_datasets2.py](nnunetv2_build_datasets2.py): compatibility check, local + container backends, per-GPU concurrency, and the multi-machine `mkdir`-based claim/lock.

Output layout under `configs.yaml::smore_out_root` (default `/fs5/p_masi/linz18/data/smore_resolved_images/`):

```
<source_id>/
    _src/<source_id>.nii.gz                  # symlink to original CT
    run_smore.log                            # SMORE stdout/stderr (per-case)
    <source_id>/                             # SMORE writes here
        <source_id>_smore.nii.gz             # SR output
        weights/best_weights.pt
    <source_id>_smore.nii.gz -> <source_id>/<source_id>_smore.nii.gz
build_smore_test_images.<host>.<pid>.tsv
```

The top-level `<source_id>_smore.nii.gz` is the canonical path downstream code should use; the extra `<source_id>/` subdir is just SMORE's own naming convention (subject-id derived from the input basename, with `out_root` per-case so each case's `run_smore.log` is isolated).

If a CT is already iso (rare for clinical CT), the compatibility check fails; by default we pass the original through under the same `_smore` filename so downstream code stays uniform. Override with `--smore-on-incompatible skip`.

## Config knobs you'll most often touch

In `nnunet/configs.yaml`:

| key | meaning |
| --- | --- |
| `dataset_id`, `plan`, `configuration`, `trainer`, `folds`, `gpu_id` | nnUNetv2 predict identity, must match training |
| `cnisp_paths_yaml`, `cnisp_model_name` | which CNISP run to compare against |
| `atlas_image_dir` | where atlas CT images live (sibling to atlas_label_dir from CNISP's paths.yaml) |
| `pivot_csv`, `pivot_image_path_columns` | PHOTON pivot table for `chk_*` CT discovery |
| `work_dir` | Phase 1 working dir |
| `smore_out_root` | Phase 1.5 SMORE shared output (default `/fs5/p_masi/linz18/data/smore_resolved_images/`) |
| `summary_bucket_edges_mm` | eff-res buckets, inherited from `test_default.yaml` so reports line up with `test_summary.csv` |

CLI flags override the corresponding yaml fields when set.
