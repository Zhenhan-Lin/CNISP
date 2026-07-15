# FOV-truncation experiment runbook (Part 2, "isolate FOV")

Studies whether the CNISP-conditioned corrector recovers anatomy in an **imaged-but-
empty** region: the native CT is FOV-truncated along z (a contiguous fraction blanked
with air), **with no slice thickening** — so the only degradation is the missing field
of view. Where ch0 has no evidence, the corrector must defer to the **completed** CNISP
prior (CNISP decodes a complete shape from the visible portion — the capability naive
self-prediction cannot replicate).

**Design choices (locked):**
- ch0 = truncation-only (native CT truncated, NOT thickened) → isolates the FOV variable.
- Truncation level is encoded as a **pseudo-step** `PP = round(keep_fraction*100)`
  (keep 0.5→`_step50`), so the ENTIRE Phase-0.5 cascade pipeline + stratified loader are
  reused unchanged — now stratifying by FOV severity. Set `CORRECTOR_STRATA="50,65,80"`.
- The CNISP prior is **re-fit to the truncated observation** (run stage-1 + CNISP on the
  truncated CT), not zeroed.

**What's new (this commit) vs reused:**
| New | Reused unchanged |
|---|---|
| `nnunet/sparsify_inputs.py::_truncate_one_ct` (truncate + air-pad; returns visible range) | `build_corrector_dataset.py --layout cascade`, `build_finetune_plan.py --cascade`, `relocate_prevseg.py`, `run_train.sh`, `run_corrector_predict.sh` |
| `scripts/build_fov_truncated_data.py` (truncated CTs + manifest + sidecar) | the CNISP deployment flow (alignment + 835 stage-1 + CNISP 032) |
| `nnUNetTrainer_OrbitalCascade` `CORRECTOR_STRATA` env override | `eval_corrector.py` (extended with `--region`) |
| `eval_corrector.py --region visible\|truncated` (+ `--trunc-manifest`) | |

---

## 0. A FOV config
Copy `configs/corrector.yaml` → `configs/corrector_fov.yaml` and change:
- `corrector_data.data_root` → the FOV data root (default `<data_root>_fov`),
- `corrector_data.steps` → the pseudo-steps `[50, 65, 80]`,
- control **C**'s `dataset_id` / `dataset_name` → a NEW id (e.g. `847` /
  `PHOTON_CT_CORR_C_fov`) so the FOV model/dataset is separate from the thickness one.
Use `--config nnunet-c/configs/corrector_fov.yaml` everywhere below.

## 1. Build the truncated CTs (new)
```bash
python nnunet-c/scripts/build_fov_truncated_data.py \
    --keep-fractions 0.5,0.65,0.8 --side end        # --side random for mixed cut ends
```
Writes `<data_root>_fov/images/{case}_step{PP}_0000.nii.gz`,
`corrector_data_manifest.json` (→ build_corrector_dataset), and
`fov_truncation_manifest.json` (→ region eval).

## 2. Re-fit the CNISP prior on the truncated CTs (reused box flow)
Run the SAME CNISP deployment path used for the thickness experiment, but pointed at the
truncated CTs and the pseudo-steps — it aligns each case, runs the 835 stage-1 model on
the truncated CT to get the coarse seg, and runs CNISP 032 to decode the **completed**
iso prior:
```bash
EMIT_ISO=1 BUILD_STEPS=50,65,80 CONFIG=nnunet-c/configs/corrector_fov.yaml \
    bash nnunet-c/run_corrector_predict.sh C 0     # RUN_CNISP=1 auto for control C
#   (or drive 03_infer.py / 032_cnisp_infer_corrector.py --steps 50,65,80
#    --emit-iso-prelabel-dir <fov iso root> directly on the truncated inputs)
```
Result: the completed CNISP iso prior per `(case, PP)` under the FOV iso train root that
`build_corrector_dataset.py --layout cascade` reads.

## 3. Build + preprocess + train (reused cascade path)
Follow `CASCADE_RUNBOOK.md` A–B with the FOV config and pseudo-step strata:
```bash
python nnunet-c/scripts/build_corrector_dataset.py \
    --config nnunet-c/configs/corrector_fov.yaml --control C --layout cascade \
    --steps 50,65,80 --max-samples <N>
# fingerprint/plan 847 + 848(_prior); build_finetune_plan --cascade; copy plan to prior;
# preprocess x2; relocate_prevseg; check_preprocessed --cascade   (all as CASCADE_RUNBOOK.md)
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

# whole-volume + FOV-restricted, with refinement metrics:
TM=<data_root>_fov/fov_truncation_manifest.json
MAP=nnunet-c/test_input/PHOTON_CT_CORR_C_fov/test_cases_map.json
PRED=nnunet-c/predictions/PHOTON_CT_CORR_C_fov/fold_0
python nnunet-c/diagnostics/eval_corrector.py --map $MAP --pred-dir $PRED --full-metrics
python nnunet-c/diagnostics/eval_corrector.py --map $MAP --pred-dir $PRED --full-metrics \
    --region truncated --trunc-manifest $TM     # recovery in the blanked FOV
python nnunet-c/diagnostics/eval_corrector.py --map $MAP --pred-dir $PRED --full-metrics \
    --region visible   --trunc-manifest $TM     # fidelity where ch0 has signal
```

**Key alignment note:** the sidecar (`fov_truncation_manifest.json`) is keyed by the
corrector_data **case_id** → pseudo-step, and `--region` looks it up by the eval map's
**source_id** + **step**. Those must match (same source naming), and the region mask
applies only when the GT grid equals the recorded `source_shape` (else that case is
skipped with a warning and can be scored whole-volume with `--region all`).
