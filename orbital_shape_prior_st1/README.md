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

## Execution order

```bash
# 1. Prepare aligned patches (once)
python scripts/01_prepare_data.py -p configs/paths.yaml

# 2. Train shape prior
python scripts/02_train.py -p configs/paths.yaml -c configs/train_default.yaml

# 3. Infer on test set (writes step_XX/, native_space/, native_space_step_XX/)
python scripts/03_infer.py -p configs/paths.yaml \
    -t configs/train_default.yaml -c configs/test_default.yaml -m <model_name>

# 4. Build the visualization bundle (recon_summary.png + heatmaps + sweep audit)
python scripts/04_visualization.py -p configs/paths.yaml \
    -t configs/train_default.yaml -c configs/test_default.yaml -m <model_name>
```
