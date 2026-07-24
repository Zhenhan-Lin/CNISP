# Test Result CSV Reference

## Overview

Test results are collected through two parallel pipelines under `simulation/`.
Both pipelines store **per-case, per-method, per-structure** results.

---

## Pipeline 1: Comparison (`simulation/comparison/`)

Entry point: `compare_native.py`

### Primary outputs

| CSV | Granularity | Description |
|-----|-------------|-------------|
| `paired_per_source__{run_tag}__{experiment}.csv` | per case × method × step × structure | Long-format table with every test case's Dice for each method |
| `paired_summary__{run_tag}__{experiment}.csv` | per bucket × structure | Aggregated mean/std Dice by effective-resolution bucket |
| `paired_summary__{run_tag}__{experiment}.txt` | — | Human-readable wide table (not CSV) |

### `paired_per_source` columns

| Column | Description |
|--------|-------------|
| `source_id` | Test case identifier |
| `gt_source` | GT origin (atlas manual GT or chk_pseudo) |
| `method` | Method name (e.g. `nnUNet`, `CNISP-atlasGT`, `nnUNet-C`, `nnUNet-interp`) |
| `step_size` | Sparsification step |
| `slice_start_id` | Slice start offset (0, 1, 2 for coarse eff_res) |
| `eff_res_mm` | Effective resolution in mm |
| `structure` | Anatomical structure (Globe, Optic nerve, Recti, Fat, mean) |
| `dice` | Dice coefficient |

### `paired_summary` columns

| Column | Description |
|--------|-------------|
| `bucket` | `{method} {eff_res_range}` |
| `structure` | Anatomical structure |
| `mean_dice` | Mean Dice across sources in the bucket |
| `std_dice` | Standard deviation |
| `n_sources` | Number of contributing sources |

### Derived CSVs (from downstream scripts)

| Script | Output CSV | Description |
|--------|-----------|-------------|
| `method_summary.py` | `{method}_per_source.csv` | Per-method per-source Dice |
| `method_summary.py` | `{method}_summary_by_eff_res.csv` | Per-method Dice by eff_res bucket |
| `paired_summary.py` | `paired_summary_by_eff_res.csv` | Head-to-head delta table |
| `experiment_summary.py` | `experiment_summary.csv` | Cross-experiment (thin/thick/real) aggregation |

### Methods covered (2–4)

| Method label | Source |
|--------------|--------|
| `nnUNet` | nnUNet sparse-CT predictions (always present) |
| `CNISP-atlasGT` / `CNISP-nnUNetPred` | CNISP canonical-space Dice from `sweep_results.pkl` |
| `nnUNet-interp` | Taubin-smoothed nnUNet control (optional, needs `nnunet-interp` phase) |
| `nnUNet-C` | nnUNet-C corrector (optional, needs `--nnunet-c-eval-csv`) |

### Default output directory

`comparison/` (overridable via `--out-dir` or config key `comparison_out_dir`)

---

## Pipeline 2: Evaluation (`simulation/evaluation/`)

Entry point: `build_mask_index.py` → `build_metrics.py` → `*_summary.py` drivers

### Primary output

| CSV | Granularity | Description |
|-----|-------------|-------------|
| `metrics_long.csv` | per case × arm × step × structure | All metrics for all 5 arms |

### `metrics_long` columns

| Column | Description |
|--------|-------------|
| `case` | Test case identifier |
| `arm` | Method/arm name |
| `step` | Sparsification step |
| `mode` | Experiment mode (thin/thick/real) |
| `eff_res` | Effective resolution in mm |
| `structure` | Anatomical structure (Globe, Optic nerve, Recti, Fat) |
| `dice` | Dice coefficient |
| `vol_pred` | Predicted volume (mm³) |
| `vol_gt` | GT volume (mm³) |
| `assd` | Average Symmetric Surface Distance (mm) |
| `hd95` | 95th percentile Hausdorff Distance (mm) |
| `nsd` | Normalized Surface Dice (at τ = 1.0 mm) |
| `signed_pct` | Signed volume error (%) |

### Arms covered (5)

| Arm | Description |
|-----|-------------|
| `nnUNet` | Image-conditioned nnUNet on sparse CT (baseline) |
| `Cascade UNet` | nnU→nnU self-correction (control B) |
| `CNISP` | CNISP shape prior with nnUNet sparse pred as input |
| `Proposed` | nnU→CNISP→nnU corrector (control C) |
| `Oracle` | CNISP shape prior with GT as input (ceiling) |

### Derived CSVs (from `*_summary.py` drivers)

| Script | Output CSV | Description |
|--------|-----------|-------------|
| `volume_agreement_summary.py` | `bland_altman_bias_by_arm.csv` | Per-arm volume bias (Bland-Altman stats) |
| `volume_stability_summary.py` | `volume_stability_cov_summary.csv` | CoV summary by arm × structure |
| `volume_stability_summary.py` | `volume_stability_cov_detail.csv` | Per-case CoV detail |
| `volume_stability_summary.py` | `volume_stability_on_range_detail.csv` | Optic nerve range detail |
| `cross_resolution_summary.py` | `cross_resolution/{arm}/cross_res_dice_matrix.csv` | Cross-step Dice matrix (one per arm) |
| `plausibility_summary.py` | `plausibility/plausibility_long.csv` | Per-eye topology/continuity metrics |
| `plausibility_summary.py` | `plausibility/plausibility_tests.csv` | Paired statistical test results |

### Default output directory

`comparison/viz/evaluation__{mode}/` (e.g. `evaluation__thick/`)

---

## Upstream (outside `simulation/`)

| CSV | Location | Description |
|-----|----------|-------------|
| `test_results.csv` | `{output_basedir}/{model}/runs/{experiment}/{run_tag}/` | CNISP inference per-(case, step) metrics; produced by `orbital_shape_prior_st1/engine/infer.py` |

---

## Quick lookup: "Where do I find X?"

| I want to find... | Look at |
|--------------------|---------|
| Per-case Dice for nnUNet vs CNISP | `paired_per_source__*.csv` |
| Aggregated Dice by resolution bucket | `paired_summary__*.csv` |
| Per-case Dice + surface metrics for all 5 arms | `metrics_long.csv` |
| Volume agreement / Bland-Altman stats | `bland_altman_bias_by_arm.csv` |
| Volume stability (CoV) | `volume_stability_cov_*.csv` |
| Cross-resolution consistency | `cross_resolution/{arm}/cross_res_dice_matrix.csv` |
| Topology / plausibility | `plausibility/plausibility_long.csv` |
| Statistical tests (paired Wilcoxon) | `plausibility/plausibility_tests.csv` |
| CNISP-only inference results | `test_results.csv` (upstream) |
