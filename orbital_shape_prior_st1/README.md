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
│   └── io_utils.py             # NIfTI I/O, checkpoint management
├── diagnostics/
│   ├── __init__.py
│   ├── reconstruction_qc.py    # centroid shift, volume ratio, aligned dice
│   ├── latent_analysis.py      # latent space visualization (t-SNE, PCA)
│   └── report.py               # generate summary tables + figures
├── scripts/
│   ├── 01_prepare_data.py      # end-to-end data preparation
│   ├── 02_train.py             # launch training
│   ├── 03_infer.py             # launch inference
│   └── 04_diagnose.py          # run all diagnostics
└── README.md
```

## Execution order

```bash
# 1. Prepare aligned patches (once)
python scripts/01_prepare_data.py -p configs/paths.yaml

# 2. Train shape prior
python scripts/02_train.py -p configs/paths.yaml -c configs/train_default.yaml

# 3. Infer on test set
python scripts/03_infer.py -p configs/paths.yaml -c configs/eval_default.yaml -m <model_name>

# 4. Run diagnostics
python scripts/04_diagnose.py -p configs/paths.yaml -m <model_name>
```
