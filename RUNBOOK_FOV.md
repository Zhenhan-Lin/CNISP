# FOV-Truncation Experiment Runbook (Part 2, "isolate FOV")

Companion to `RUNBOOK.md`. Studies whether the CNISP-conditioned corrector recovers
anatomy in an **imaged-but-empty** region: the native CT is FOV-truncated (a
contiguous region blanked with air), **with no slice thickening** — so the only
degradation is the missing field of view, and the corrector must defer to the
**completed** CNISP prior where ch0 has no evidence.

**Locked design choices:**
- ch0 = truncation-only (native CT truncated, NOT thickened) → isolates the FOV variable.
- Two truncation geometries (`--mode`): **slab** (type-1) blanks a through-plane slab;
  **box** (type-2) keeps one axis-aligned FOV box — globe/anterior axis kept, the
  up/down + left/right axes corner-clipped (mis-centred acquisition). See §1.
- Truncation level is encoded as a **pseudo-step** `PP = round(keep_fraction*100)`
  (keep 0.5→`_step50`), so the ENTIRE cascade pipeline + stratified loader are reused
  unchanged — now stratifying by FOV severity. Set `CORRECTOR_STRATA="50,65,80"`.
- The CNISP prior is **re-fit to the truncated observation** (stage-1 + CNISP on the
  truncated CT), not zeroed.

**What is new vs reused**
| New (Part 2 code) | Reused unchanged |
|---|---|
| `nnunet/sparsify_inputs.py::_truncate_one_ct` (slab, type-1) + `_truncate_one_ct_box` (box, type-2) | `build_corrector_dataset.py --layout cascade`, `build_finetune_plan.py --cascade`, `relocate_prevseg.py`, `run_train.sh`, `run_corrector_predict.sh` |
| `nnunet-c/scripts/build_fov_truncated_data.py` | the CNISP deployment flow (alignment + 835 stage-1 + CNISP 032) |
| `CORRECTOR_STRATA` env on the trainer | `eval_corrector.py` (extended with `--region`) |
| `eval_corrector.py --region visible\|truncated` | |

A separate dataset id keeps the FOV model apart from the thickness one; below uses
`847 / PHOTON_CT_CORR_C_fov` (prior `848`).

---

## 0. FOV config
Copy `nnunet-c/configs/corrector.yaml` → `nnunet-c/configs/corrector_fov.yaml`; change:
- `corrector_data.data_root` → the FOV root (default `nnunet-c/data_fov_pereye_test`),
- `corrector_data.steps` → `[50, 65, 80]`,
- control **C** `dataset_id`/`dataset_name` → `847` / `PHOTON_CT_CORR_C_fov`.
Pass `--config nnunet-c/configs/corrector_fov.yaml` (and `CONFIG=…` to wrappers) everywhere.

## 1. Build the truncated CTs (new)
Three geometries:
```bash
# type-1 (slab, default): blank a through-plane slab (top/bottom of FOV cut off)
python nnunet-c/scripts/build_fov_truncated_data.py \
    --keep-fractions 0.5,0.65,0.8 --side end     # --side random/both also available

# type-2 (box): keep one GLOBAL axis-aligned FOV box; globe/anterior axis never cut,
# the up/down + left/right axes corner-clipped -> a mis-centred acquisition.
python nnunet-c/scripts/build_fov_truncated_data.py \
    --keep-fractions 0.5,0.65,0.8 --mode box --corner SL   # SL|SR|IL|IR (S/I x L/R) or random

# min-retain (RECOMMENDED for the real experiment): the SAME single global corner box,
# but sized by binary search so BOTH orbits stay >= T of foreground AND >= T_on of ON
# (worst-eye binding). Physically a mis-centred FOV -> the near eye is clipped more, the
# floor keeps even it >= half visible. --min-retains gives the floor levels.
python nnunet-c/scripts/build_fov_truncated_data.py \
    --min-retains 0.5,0.65,0.8 --corner SL --min-retain-on 0.5   # floors -> steps 50,65,80
```
Writes `nnunet-c/data_fov_pereye_test/images/{case}_step{PP}_0000.nii.gz`, a
`corrector_data_manifest.json` (→ build_corrector_dataset), and a
`fov_truncation_manifest.json` sidecar (per (case,PP): `source_shape` + the visible
window — **slab**: `trunc_axis` + `visible_range`; **box** / **box_min_retain**:
`visible_box` (per-axis `[lo,hi]`) + `corner`. box also has `retained_*`;
box_min_retain also has `keep_fraction` (the calibrated global fraction) + `per_eye.{OD,OS}`
retention QC (`ret_total`/`ret_ON`/`ret_per_structure`/`binding_constraint`)). The region
eval reads `visible_box` (single box) for both box variants.

**Box specifics:**
- The globe/anterior axis is found from the CT affine (`aff2axcodes` → the A/P axis)
  and kept in full; the two orthogonal axes are the ones clipped.
- `keep_fraction` = TOTAL retained orbit-volume fraction (split `sqrt` across the two
  cut axes), so `_step50` ≈ half the eye out of FOV — comparable to the slab pseudo-step.
- The cut is anchored on the orbit bbox (from `gt_candidate_pred`, mapped through
  world coords so gt/CT grids need not match), so it reliably bites into the eye.
- `retained_per_structure` in the sidecar is the ">= half of every structure visible"
  QC — check it before training; raise `keep_fraction` if a structure drops too low.
- NOTE (deferred): under box truncation the observed globe centroid the CNISP
  re-fit estimates (§2) drifts more than under slab, since part of the globe is
  removed. We are running the pipeline with the existing observed-alignment
  estimator as-is; a truncation-robust globe-centre estimate is a later refinement.

**min-retain (`--min-retains`) specifics:**
- It is the **same single global corner box** as `--mode box` (blanked corner extends to
  the image edge → a real truncation, NOT an interior hole); only the sizing differs.
- Eyes are split via `canonical_align.separate_eyes` (globe CC → OD/OS) + the L-R midline
  **only to MEASURE** per-eye retention; a case is skipped if either eye's ON has
  `< --min-on-vox` voxels (default 10).
- The global `keep_fraction` is **binary-searched** to the DEEPEST single box still holding
  `ret_total >= T` and `ret_ON >= T_on` for **both** eyes (worst-eye binding), so `_step50`
  means "both eyes keep >= 50%" as a hard floor.
- Because it is ONE global cut plane, the eye nearer the clipped corner loses more; the far
  eye retains more (both >= T). That asymmetry is physically correct for a mis-centred FOV —
  if you need the two eyes truncated symmetrically, use a superior/inferior **slab** instead.
- `per_eye.{OD,OS}` in the sidecar reports each eye's achieved `ret_total`/`ret_ON`/per-structure
  and `binding_constraint`; check them (Stage 2) — all should be >= T.

Check: `ls nnunet-c/data_fov_pereye_test/images | head`; the sidecar has an entry per case/pseudo-step.

## 1b. Extreme-case scout test of the CNISP output (box mode)
Before a full training run, sanity-check **how the CNISP-completed prior behaves under
extreme box truncations** — aggressive `keep_fraction` and every corner — with
`nnunet-c/diagnostics/fov_extreme_test.py`. It targets the failure modes from the design
review: does CNISP complete the blanked FOV, does its centroid drift, does the eye run off
the fixed decode patch.

```bash
# (0) logic check, no model/data needed:
python nnunet-c/diagnostics/fov_extreme_test.py --self-test

# (1) build the extremes (reuses the box builder; small --max-cases to keep it a scout):
for C in SL SR IL IR; do
  python nnunet-c/scripts/build_fov_truncated_data.py --mode box --corner "$C" \
    --keep-fractions 0.25,0.35,0.5 --max-cases 3 \
    --out-data-root "nnunet-c/data_fov_extreme_${C}"
done
# (2) run the CNISP re-fit (Step 2 below) on each data_fov_extreme_* root to emit the
#     completed iso prior per (case, pseudo-step).
# (3) analyze one (case, step) at a time (append rows to a CSV):
python nnunet-c/diagnostics/fov_extreme_test.py --analyze \
    --ref  <untruncated gt_candidate_pred / prior for SRC, on the source grid> \
    --cnisp <CNISP completed iso prior for SRC step 35> \
    --trunc-manifest nnunet-c/data_fov_extreme_SL/fov_truncation_manifest.json \
    --source-id SRC --step 35 --out-csv /tmp/fov_extreme.csv
```

Diagnostics per case (JSON + a one-line `flag`):
- `recovery_trunc` — of the reference anatomy the box **blanked**, fraction the CNISP prior
  reproduced. **Low (< 0.5) = CNISP did not complete the missing FOV.**
- `globe_drift_mm` / `centroid_drift_mm` — mm between the CNISP prior's globe / all-fg
  centroid and the reference's. **Large = mislocated** (TTO cannot fix large drift).
- `extent_ratio` (per axis) + `cnisp_touches_boundary` — CNISP fg bbox extent vs reference;
  `< 1` = under-covers, and a `true` boundary flag = the prior likely ran off the fixed
  64 mm decode patch (the silent clip of a large / drifted eye — see the alignment note).
- `per_structure` — `vol_ratio` + `recovery_trunc` per ON/Recti/Globe/Fat.

`--ref` must be on the truncation **source grid** (== the sidecar `source_shape`, same guard
as `eval_corrector --region`); otherwise the box mask can't be applied and the case is
skipped. Use this scout to pick a sane `keep_fraction` floor before Step 3.

## 2. Re-fit the CNISP prior on the truncated CTs (reused box flow)
Same CNISP deployment path as the thickness run, pointed at the truncated CTs and the
pseudo-steps (it aligns each case, runs the 835 stage-1 model on the truncated CT for the
coarse seg, and runs CNISP 032 to decode the **completed** iso prior):
```bash
EMIT_ISO=1 BUILD_STEPS=50,65,80 CONFIG=nnunet-c/configs/corrector_fov.yaml \
    bash nnunet-c/run_corrector_predict.sh C 0     # RUN_CNISP=1 auto for control C
#   (or drive 03_infer.py / 032_cnisp_infer_corrector.py --steps 50,65,80
#    --emit-iso-prelabel-dir <fov iso root> directly on the truncated inputs)
```
Result: the completed CNISP iso prior per `(case, PP)` that `build_corrector_dataset.py
--layout cascade` reads.

## 3. Build + preprocess + train (reused cascade path)
Follow `RUNBOOK.md` §1 Stages 2–5 with the FOV config and pseudo-step strata.

First emit the **train-side** CNISP iso prelabels (control-C cascade ch1..4 read
them). `run_corrector_cnisp.sh` reads `CONFIG` to target the FOV tree — it derives
`EXPERIMENT=fov`, `STEPS=50,65,80`, and the `data_fov_pereye_test` aligned/out/iso
dirs from the config (env vars still override):
```bash
CONFIG=nnunet-c/configs/corrector_fov.yaml EMIT_ISO=1 \
    bash nnunet-c/run_corrector_cnisp.sh          # -> data_fov_pereye_test/cnisp_pred_train_iso
```
Then build the cascade dataset (reads those iso prelabels via `--prelabel-grid iso`):
```bash
python nnunet-c/scripts/build_corrector_dataset.py \
    --config nnunet-c/configs/corrector_fov.yaml --control C --layout cascade \
    --steps 50,65,80 --max-samples <N>
# fingerprint/plan 847 + 848; build_finetune_plan --cascade; copy plan to 848;
# preprocess x2; relocate_prevseg; check_preprocessed --cascade   (RUNBOOK.md Stages 3-4)
CASCADE=1 SKIP_PREPROCESS=1 CORRECTOR_STRATA=50,65,80 \
CORRECTOR_TRAINER=nnUNetTrainer_OrbitalCascade \
CONFIG=nnunet-c/configs/corrector_fov.yaml \
bash nnunet-c/run_train.sh C 0
```

## 4. Predict + region-restricted eval (new eval knob)
```bash
CASCADE=1 CORRECTOR_TRAINER=nnUNetTrainer_OrbitalCascade \
CONFIG=nnunet-c/configs/corrector_fov.yaml \
bash nnunet-c/run_corrector_predict.sh C 0

TM=nnunet-c/data_fov_pereye_test/fov_truncation_manifest.json
MAP=nnunet-c/test_input/PHOTON_CT_CORR_C_fov/test_cases_map.json
PRED=nnunet-c/predictions/PHOTON_CT_CORR_C_fov/fold_0
python nnunet-c/diagnostics/eval_corrector.py --map $MAP --pred-dir $PRED --full-metrics                                  # whole volume
python nnunet-c/diagnostics/eval_corrector.py --map $MAP --pred-dir $PRED --full-metrics --region truncated --trunc-manifest $TM   # recovery in the blanked FOV
python nnunet-c/diagnostics/eval_corrector.py --map $MAP --pred-dir $PRED --full-metrics --region visible   --trunc-manifest $TM   # fidelity where ch0 has signal
```

**Key alignment note:** the sidecar is keyed by the corrector_data **case_id** →
pseudo-step, and `--region` looks it up by the eval map's **source_id** + **step**. Those
must match (same source naming), and the region mask applies only when the GT grid equals
the recorded `source_shape` (else that case is skipped with a warning; score it whole-volume
with `--region all`).

## 5. Cleanup / checks
Same reset commands as `RUNBOOK.md` §4 with the FOV ids (`847`/`848`) and
`data_root = nnunet-c/data_fov_pereye_test`. To regenerate the truncated CTs: `rm -rf
nnunet-c/data_fov_pereye_test/images` (or `--force`). Verify the train log shows
`num_input_channels: 5` and the `[StepStratified]` loader lists the pseudo-steps
`{50,65,80}`.
