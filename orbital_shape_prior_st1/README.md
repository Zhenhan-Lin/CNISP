# Orbital Implicit Shape Prior — Stage 1

## Goal
Validate whether a neural implicit shape prior (AutoDecoder) can learn
orbital anatomy from mildly anisotropic CT segmentations and reconstruct
3D multi-structure shapes from sparse slices.

## Key design decisions (departures from Amiranashvili et al.)

| Decision | Rationale |
|---|---|
| **Multi-class (5-ch softmax)** | 4 orbital structures + background; follows Jansen et al. |
| **Canonical alignment via globe centroid** | Lightweight; avoids ANTs template registration |
| **OS→OD flip** | Doubles training data; left/right share the same prior |
| **Per-structure sampling weight** | ON is ~1% of volume; uniform sampling under-represents it |
| **No intensity input (Stage 1)** | Isolate shape prior ceiling; intensity integration is Stage 2 |

## Directory layout

```
orbital_shape_prior/
├── configs/
│   ├── paths.yaml              # data paths (user fills in)
│   ├── train_default.yaml      # training hyperparameters
│   └── eval_default.yaml       # test-time optimization params
├── data_prep/
│   ├── __init__.py
│   ├── canonical_align.py      # Step 1: globe localization + crop + flip
│   ├── alignment_qc.py         # Step 1 QC: centroid std, overlay viz
│   ├── build_caselist.py       # generate train/test casename files
│   └── sparsify.py             # synthetic anisotropy degradation
├── models/
│   ├── __init__.py
│   ├── multiclass_ad.py        # 5-class AutoDecoder + OccupancyPredictor
│   └── losses.py               # CE + multi-class Dice + latent L2
├── engine/
│   ├── __init__.py
│   ├── dataset.py              # OrbitalImplicitDataset (extends Amiranashvili)
│   ├── train.py                # training loop
│   ├── infer.py                # test-time latent optimization + reconstruction
│   ├── native_mapping.py       # project canonical patches back to native CT space
│   ├── visualize.py            # result-summary + cross-resolution heatmaps
│   └── io_utils.py             # NIfTI I/O, checkpoint management
├── diagnostics/
│   ├── __init__.py
│   ├── multiview_qc.py         # train-time multi-offset diagnostic
│   └── resolution_sweep.py     # adaptive per-case sparsity sweep used by infer
├── scripts/
│   ├── 01_prepare_data.py      # end-to-end data preparation
│   ├── 02_train.py             # launch training
│   ├── 03_infer.py             # launch inference (also writes per-step native space)
│   └── 04_visualization.py     # generate result summary + cross-resolution heatmaps
└── README.md
```

## Option C: ceiling vs deployment curves

The same model weights drive **two test-time inference runs** off the same 62-eye test set; they differ only in the latent-opt input and the dense Dice target.

| run_tag       | `test_label_source` | latent-opt input                                                                                | dense Dice target                                                                                                                       | output dir                              |
| ------------- | ------------------- | ----------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------- |
| `atlas_gt`    | `atlas_gt`          | `sparsen_volume(canonical GT)` from `${aligned_dir}/labels/`                                    | same canonical GT — **ceiling curve**, isolates the shape prior                                                                          | `output_basedir/<model>/runs/atlas_gt/` |
| `nnunet_pred` | `nnunet_pred`       | per-step Dataset835 sparse-CT pred canonical-aligned, from `${aligned_dir}/labels_dataset835_step_XX/` (built by `nnunet/engine/build_dataset835_sparse_patches.py`) | atlas cases: atlas manual GT (`${aligned_dir}/labels/atlas_*.nii.gz`). chk_* cases: Dataset835 dense pred canonical-aligned (`${aligned_dir}/labels_dataset835/chk_*.nii.gz`) — **deployment curve** | `output_basedir/<model>/runs/nnunet_pred/` |

The deployment curve includes nnUNet's prediction noise (centroid jitter, dropped globes at high sparsity, scheme mismatches) in the latent-opt input, while the ceiling curve receives a perfectly clean sparse observation. `nnunet/compare_native.py` reads both runs and labels the rows `CNISP-atlasGT` vs `CNISP-nnUNetPred` so they can be plotted side-by-side against `nnUNet-sparse`.

`test_label_source` and `run_tag` are independent knobs in `configs/test_default.yaml`, but in practice they should match (a `run_tag=atlas_gt` directory with `test_label_source=nnunet_pred` will silently land deployment outputs in the ceiling slot). The pipeline drives them as a pair via `scripts/run_03_test.sh "$YAML" "$test_label_source" "$run_tag"`.

## Execution order

```bash
# 1. Prepare aligned patches (once)
python scripts/01_prepare_data.py -p configs/paths.yaml

# 2. Train shape prior
python scripts/02_train.py -p configs/paths.yaml -c configs/train_default.yaml

# 3a. Ceiling curve (run_tag=atlas_gt, the default)
python scripts/03_infer.py -p configs/paths.yaml \
    -t configs/train_default.yaml -c configs/test_default.yaml \
    -m <model_name>
# 3b. Deployment curve (run_tag=nnunet_pred) -- requires the chk_* dense
#     and per-step sparse Dataset835 canonical patches under aligned_dir/
#     (produced by ../nnunet/engine/build_dataset835_canonical_patches.py
#     and build_dataset835_sparse_patches.py; see ../nnunet/README.md and
#     ../run_pipeline.sh's `cnisp-prep-dataset835-*` phases).
python scripts/03_infer.py -p configs/paths.yaml \
    -t configs/train_default.yaml -c configs/test_default.yaml \
    -m <model_name> \
    --test-label-source nnunet_pred --run-tag nnunet_pred

# 4. CNISP-only artifacts for each run_tag (cross-resolution heatmaps,
#    file-tree dump, native_space_step_XX/ audit). Per-step Dice trend
#    / per-class / per-case figures come from ../run_pipeline.sh's
#    `compare` phase, which writes the CNISP slice of the bundle to
#    output_basedir/<model_name>/viz/<run_tag>/<method_label>_recon_summary.png.
python scripts/04_visualization.py -p configs/paths.yaml \
    -t configs/train_default.yaml -c configs/test_default.yaml \
    -m <model_name> --run-tag atlas_gt
python scripts/04_visualization.py -p configs/paths.yaml \
    -t configs/train_default.yaml -c configs/test_default.yaml \
    -m <model_name> --run-tag nnunet_pred
```

The repo-level [run_pipeline.sh](../run_pipeline.sh) wraps every step above plus the nnUNet sparse-CT sweep and `compare` phase. Use it for the full end-to-end run; the per-script invocations here exist mainly for partial reruns of one curve at a time.
