# Evaluation Figures Book

Reference for every figure the `simulation/evaluation/` subsystem produces: **what
it shows, the exact formula, the data source, and where the plot + its numbers
land on disk.** Companion to the code in this directory. Line references are to the
files as of this writing — grep the function names if they drift.

---

## 0. The pipeline (how a figure is built)

```
build_mask_index.py   →  mask_index.json      (5-arm A–E mask registry)
build_metrics.py      →  metrics_long.csv     (tidy per-structure numbers; THE interface)
<name>_summary.py     →  <name>.png (+ CSVs)  (one figure per driver)
```

One-shot wrapper: `simulation/evaluation/make_eval_figures.sh <mask_index.json>`
runs `build_metrics` then all four `*_summary` drivers, writing under the
mask_index's own directory (e.g. `comparison/viz/evaluation__thick/`).

### The 5 arms (`metrics.py:42`, display order A→E)

| Letter | `METHODS` name | Meaning | Prediction source |
|---|---|---|---|
| A | `nnUNet` | image-conditioned nnUNet on the sparse CT (baseline) | `work_dir/prediction/.../sparse_step_XX_native` |
| B | `Cascade UNet` | nnU→nnU self-correction (Dataset855 corrector) | `predictions/PHOTON_CT_CORR_B_stacked/fold_0` |
| C | `CNISP` | CNISP shape prior on the nnUNet sparse pred | CNISP run `native_space_step_XX` (e.g. `corrector_gt`) |
| D | `Proposed` | nnU→CNISP→nnU corrector (Dataset845 corrector) | `predictions/PHOTON_CT_CORR_C_cnisp/fold_0` |
| E | `Oracle` | CNISP shape prior on the **GT** (ceiling) | CNISP run `atlas_gt` |

`GT` is **not** a plotted arm — every metric is computed *against* GT, but a
GT-vs-GT reference (Dice 1 / ASSD 0 / CoV 0) is intentionally omitted (`metrics.py:37-41`).

### Structures (`metrics.py:29`)

`["Globe", "Optic nerve", "Recti", "Fat"]`.

### `metrics_long.csv` columns (`metrics.py:212-216`)

`case, arm, step, mode, eff_res, structure, dice, vol_pred, vol_gt, assd, hd95,
nsd, signed_pct`.

### Volume definition (shared by BA + stability) — `metrics.py:159-180`

- If the prediction grid ≠ GT grid (always true for the iso-0.5 corrector arms B/D
  vs native GT), the prediction is **resampled onto the GT grid**, world-aware,
  order 0 (nearest, labels stay discrete) — `metrics.py:159-169`.
- `vol = voxel_count × prod(GT_spacing)` mm³, with the **same** `vv` for pred and
  GT — `metrics.py:173,179-180`. So `vol_pred` and `vol_gt` are always in the same
  units on the same grid. **There is no per-arm voxel-size scaling bug** (audited).
- `signed_pct = 100·(vol_pred − vol_gt)/vol_gt` — `metrics.py:216`.

### `--common-samples` (default **ON**) — `aggregate.py:23-51`

Restricts every figure to the `(case, step)` present for **all** arms except GT,
so a difference reflects method quality, not coverage. `make_eval_figures.sh`
leaves the driver default on unless you pass `COMMON_SAMPLES=…`; drivers accept
`--no-common-samples` to use each arm's full set.

---

## 1. Surface quality — `surface_quality_metrics.png`

- **Driver:** `surface_quality_summary.py` → `plots.surface_figure`.
- **Shows:** three boxplots per arm — ASSD (mm) ↓, HD95 (mm) ↓, Surface-Dice@τ ↑.
- **Aggregation** (`aggregate.surface`): per `(arm, case, step)`, mean of the metric
  **over the 4 structures**; the box is the distribution over those (case, step).
- **Source columns:** `assd, hd95, nsd` (τ = `--tau-mm`, default 1.0 mm).
- **Output:** `<out>/surface_quality_metrics.png`.

---

## 2. Volume agreement — `volume_agreement_bland_altman.png`  (+ per-arm table & panels)

- **Driver:** `volume_agreement_summary.py` → `plots.volume_agreement_figure`.
- **Structure:** `--ba-structure` (default **Globe**).
- **Panels:** (a) Bland–Altman for **nnUNet**, (b) Bland–Altman for **Proposed**,
  (c) signed-volume-error violins for **all 5 arms**. Panels (a)/(b) **share one
  y-axis** (`plots.py:129`) so they're directly comparable.
- **Bland–Altman math** (`plots._bland_altman`, `plots.py:104-118`):
  - x = `(V_pred + V_GT)/2`, y = `diff = V_pred − V_GT` (mm³).
  - **bias** = `mean(diff)`; **LoA** = `±1.96·std(diff)` (population std, ddof=0);
    **Lin's CCC** = `2·cov(V_pred,V_GT) / (var_pred + var_gt + (mean_pred−mean_gt)²)`.
  - Point color = `eff_res` (slice thickness).
- **Reading the bias:** a positive bias = the arm predicts a **larger** volume than
  GT (over-segmentation). The Proposed arm's large `+`bias on Globe is a **faithful
  measurement**, not a scale/grid artifact — the CNISP shape prior completes/inflates
  the globe. To chase the cause, look **upstream** (CNISP iso-0.5 export /
  corrector output), not in `metrics.py`/`aggregate.py`/`plots.py`.

### 2b. Per-arm bias table + panels (added)

Because the combined figure only draws nnUNet + Proposed, the driver also emits, on
the **same restricted sample as the figure**:

- **`bland_altman_bias_by_arm.csv`** — one row per arm: `n, bias_mm3, sd_diff_mm3,
  loa_lo_mm3, loa_hi_mm3, ccc, mean_vol_pred_mm3, mean_vol_gt_mm3, mean_signed_pct`
  (`aggregate.volume_agreement_per_arm`; same ddof/CCC conventions as the plot).
- **stdout table** — `arm | n | bias(mm³) | ±LoA | CCC | signed%` for all 5 arms.
- **`bland_altman_per_arm/bland_altman_<arm>.png`** — a standalone BA panel per arm
  (arms with < 2 paired points are skipped and reported).

---

## 3. Volume stability — `volume_stability_by_resolution.png`  (+ CoV/range CSVs)

- **Driver:** `volume_stability_summary.py` → `plots.stability_figure`.
- **⚠ Naming caveat:** despite the filename, **there are no effective-resolution
  buckets** in this figure. The two panels are:
  - **(a) CoV bars** — x = the 4 **structures**, grouped bars per arm.
  - **(b) Optic-nerve per-scan range violin** — x = the 5 **arms**. This is a
    **range**, not a CoV.
- **CoV** (`aggregate.stability`, `aggregate.py:61-63`):
  `CoV = 100 · std(ddof=0) / mean` of `vol_pred` across **step_sizes within a
  `(arm, structure, case)`**, then **averaged over cases**. The error bar is
  `std(ddof=1)` of the per-case CoVs.
- **Panel (b) range** (`aggregate.py:69-70`): `100·(max−min)/mean` of Optic-nerve
  `vol_pred` across step_sizes, one point per `(arm, case)`.
- **Known correctness caveats (audited; not silently changed):**
  1. **No minimum-step guard.** A `(arm, case)` with a single step → std 0 → CoV/range
     **0** ("perfectly stable"), diluting the distribution rather than being dropped.
  2. **ddof inconsistency** — CoV uses population std (ddof=0), its error bar uses
     sample std (ddof=1).
  3. **Near-zero-mean inflation** — Optic nerve is the smallest structure; a case with
     a few-voxel ON mask has a tiny mean, so `(max−min)/mean` (and CoV) can blow up.
     No clipping.

### 3b. CoV / range tables (added — this is where "Oracle variance" lives)

The figure kept these numbers in memory only; the driver now writes:

- **`volume_stability_cov_summary.csv`** — per `(arm, structure)`: `n_cases,
  cov_mean_pct, cov_sd_pct`. **This is the per-arm variance table.**
- **`volume_stability_cov_detail.csv`** — per `(arm, structure, case)`: `n_steps,
  mean_vol_mm3, cov_pct`. **This answers "how many case × step_size feed a CoV":**
  each row is one case; `n_steps` = the step_sizes for that case.
- **`volume_stability_on_range_detail.csv`** — per `(arm, case)` for Optic nerve:
  `n_steps, mean_vol_mm3, range_pct` (the panel-(b) points).
- **stdout** prints the **Oracle × Optic nerve** coverage: `n_cases`, how many are
  single-step (CoV 0), and the mean CoV.

> To find Oracle × ON: filter `volume_stability_cov_detail.csv` for
> `arm==Oracle, structure=="Optic nerve"` — row count = cases, `n_steps` per row =
> step_sizes per case; the summary CSV's `n_cases` is the count folded into the bar.

---

## 4. Plausibility — `plausibility/…`

- **Driver:** `plausibility_summary.py` → `plausibility_plots`.
- **Outputs** (under `<out>/plausibility/`): `topology_violation_rate_<metric>.png`,
  `cross_slice_continuity.png`, `cross_slice_area_stability.png`,
  `compactness_distribution.png`, and — **only** if the qualitative args are passed
  (`--qualitative-case --qualitative-step --ct-source --test-cases-map`) —
  `qualitative_comparison.png`.
- Computed directly from the masks in the MASK_INDEX (not from `metrics_long.csv`);
  optional `--plausibility-csv` caches the intermediate numbers.

---

## 5. `combined__thick/` — a SEPARATE, older track (not part of the above)

- **Driver:** `simulation/comparison/combined_summary.py` (the `run_pipeline.sh`
  compare / cnisp-viz phases), **not** `simulation/evaluation/`.
- **Shows:** Dice-vs-effective-resolution curves overlaying nnUNet-sparse, each CNISP
  run, and the nnUNet-C corrector(s). Reads `comparison/paired_per_source__<run_tag>__<exp>.csv`
  (built by the compare phase), **not** `metrics_long.csv`.
- **Label caveat:** it renders raw internal `method_label` strings with **no A–E
  mapping**. In particular the code's **`nnUNet-C (C)` = arm D (Proposed)** and
  **`nnUNet-C (B)` = arm B (Cascade)**, while arm **C (CNISP)** is the plain
  `CNISP-v6.5-gt` run — the letters do **not** line up with A–E.
- **Why it may show 4 not 5 curves:** a curve is dropped when its method has zero
  rows. The usual culprit is **Oracle (`atlas_gt`)** having no `thick` paired CSV
  (`configs_v6_5_gt.yaml:27-32` warns the `atlas_gt` thick run may never have been
  inferred). Generate/compare that run to restore the 5th curve.

For the clean 5-arm A–E comparison, prefer the `evaluation__thick/` figures above.

---

## Quick file map (per figure)

| Figure PNG | Driver | Aggregate fn | Added CSV(s) |
|---|---|---|---|
| `surface_quality_metrics.png` | `surface_quality_summary.py` | `aggregate.surface` | — |
| `volume_agreement_bland_altman.png` | `volume_agreement_summary.py` | `aggregate.volume_agreement` + `volume_agreement_per_arm` | `bland_altman_bias_by_arm.csv`, `bland_altman_per_arm/*.png` |
| `volume_stability_by_resolution.png` | `volume_stability_summary.py` | `aggregate.stability` + `stability_table` | `volume_stability_cov_summary.csv`, `..._cov_detail.csv`, `..._on_range_detail.csv` |
| `plausibility/*.png` | `plausibility_summary.py` | `plausibility.py` | `--plausibility-csv` (opt) |
| `combined__thick/*.png` | `simulation/comparison/combined_summary.py` | (paired CSVs) | — |
</content>
