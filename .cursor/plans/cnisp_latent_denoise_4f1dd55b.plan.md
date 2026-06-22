---
name: CNISP latent denoise
overview: "Add a latent-space denoising/de-biasing mechanism to CNISP: a second per-case latent (alpha_nn) pinned to the nnUNet prediction, plus a shared lightweight Delta MLP that learns to navigate a noisy latent toward one that decodes the GT shape, all behind config flags for ablation."
todos:
  - id: phase1-data
    content: "engine/dataset.py: add 'dual' supervision returning both GT (dense sub) and nnUNet (sparse) samples per item; assert nnunet obs_source"
    status: completed
  - id: phase1-boundary
    content: "engine/dataset.py: add denoise.boundary_weighted_sampling (Problem 6) on disagreement region for GT-coord draws; built but default OFF"
    status: completed
  - id: phase1-train
    content: "engine/train.py: add latents_nn parameter table (INDEPENDENT init seed, Problem 2a) + param group; read denoise config block"
    status: completed
  - id: phase1-smoke
    content: "diagnostics/denoise_smoke.py --phase dual: shapes + Problem 3 frame/normalization assertion (shared origin+scale, same centroid) + alpha_GT/alpha_nn/F grad routing"
    status: pending
  - id: phase2-delta
    content: "models/denoise.py: LatentDenoiser MLP (latent->hidden->latent, ReLU, zero-init final layer)"
    status: completed
  - id: phase2-loss
    content: "engine/train.py: 3-term loss w/ stop-gradient; separate lambda_l2_nn (Problem 2b); add delta to optimizer; persist latents_nn+delta; log 3 losses + health scalars mean_alpha_gap & mean_delta_norm (Problem 2a)"
    status: completed
  - id: phase2-smoke
    content: "diagnostics/denoise_smoke.py --phase full: assert grad routing (alpha_nn unaffected by term3, delta only from term3) + zero-init no-op + prints health scalars"
    status: pending
  - id: phase2p5-gap
    content: "diagnostics/denoise_latent_gap.py (Problem 1): re-fit train cases' alpha_nn via test-time optimize_latent, report mean/max ||train-testfit|| vs norms; STOP & report if relative gap >20%"
    status: pending
  - id: phase3-infer
    content: "engine/infer.py: load+freeze delta with get('delta'); RAISE if use_delta but delta missing (Problem 4); wrap optimize_fn; alpha_hat = alpha_nn_test + Delta(...); keep label-smoothing soft-fit (Problem 5)"
    status: completed
  - id: phase3-smoke
    content: "diagnostics/denoise_smoke.py --phase infer: confirm alpha_hat shape + zero-init no-op + non-zero shift; confirm error raised when use_delta but no delta in ckpt"
    status: pending
  - id: config
    content: Add configs/train_v7_denoise.yaml (clone of train_v6_5) with the denoise block incl. lambda_l2_nn + boundary_weighted_sampling; thick-only (modes:[thick]), obs_sources:[nnunet], train_supervision:dual, model_name orbital_ad_v7
    status: pending
  - id: config-test
    content: Add configs/test_v7_denoise.yaml (clone of test_default) with start_offsets:[0], latent_fit_soft:true, sweep_mode:thick
    status: pending
  - id: config-nnunet
    content: Add nnunet/configs_v7.yaml (clone of configs_v6_5) with cnisp_model_name orbital_ad_v7, cnisp_train_yaml train_v7_denoise.yaml, sweep_degrade_mode thick, start_offsets:[0], cnisp_runs_to_compare:[nnunet_pred]
    status: pending
  - id: phase4-preflight
    content: "diagnostics/preflight_v7_thick.py: verify thick nnUNet train obs (labels_dataset835_thick_train_step_*), test deploy input (labels_dataset835_thick_step_*), dense targets (labels/atlas_*, labels_dataset835/chk_*, metadata_dataset835); STOP if any missing"
    status: pending
  - id: phase4-train
    content: "Run thick dual-latent training: scripts/02_train.py -c configs/train_v7_denoise.yaml (model orbital_ad_v7)"
    status: pending
  - id: phase4-infer
    content: "Run thick inference start=0: scripts/03_infer.py ... -m orbital_ad_v7 --test-label-source nnunet_pred --run-tag nnunet_pred --experiment thick"
    status: pending
  - id: phase4-viz-compare
    content: run_pipeline.sh --config nnunet/configs_v7.yaml --test-config configs/test_v7_denoise.yaml cnisp-viz compare -> CNISP viz + nnUNet head-to-head plots
    status: pending
isProject: false
---

## Survey summary (what exists today)

- Per-case latents: `engine/train.py` L228-239 — one `nn.Parameter latents_train` of shape `[n_train, latent_dim]`, indexed by `batch["caseids"]`. One row per dataset *item* (case x mode x step x offset).
- Decoder F: `models/multiclass_ad.py` `MultiClassAutoDecoder.forward(latents, coords) -> logits [B,*,C]`.
- Loss: `models/losses.py` `MultiClassShapeLoss` = CE + Dice (one-hot built internally from integer labels), used in `train_one_epoch` (`train.py` L50-115) + an L2 latent reg with a 100-epoch ramp.
- Test-time fit: `engine/infer.py` `optimize_latent` (L147-238) optimizes a fresh `[1,latent_dim]` latent with the net frozen; `diagnostics/resolution_sweep.py` `run_sweep` calls it (L241) then decodes via `net(latent, coords)` (L295).
- nnUNet labels are ALREADY in the pipeline. With `obs_sources: [nnunet]` (`configs/train_v6_5.yaml`), each item in `engine/dataset.py::_init_degradation_bank` carries both `labels_sparse[i]` (nnUNet sparse pred = noisy) and `labels_dense_sub[i]` (co-framed dense GT = clean) in the SAME 64 mm frame. `__getitem__` currently returns only ONE (selected by `train_supervision`). No new data loading needed — only expose both.

Confirmed decisions: alpha_GT <- dense co-framed GT (`labels_dense_sub`); alpha_nn <- sparse nnUNet pred (`labels_sparse`); per-item granularity (reuse `caseids`); build on the v6-5 `obs_sources:[nnunet]` path.

## Pre-implementation investigation (problems 1 and 5)

Problem 5 — current test-time fit target is a HARD mask, not nnUNet softmax.
- `engine/infer.py::optimize_latent` fits the latent against `labels_sparse`, which for the deployment path is loaded by `engine/test_label_sources.py::load_patch_as_label_tensor` (L278-295) from a canonical-aligned `.nii.gz`. That file stores INTEGER class labels (argmax of nnUNet), i.e. a hard mask.
- The `soft=True` / `latent_fit_soft` knob does NOT use nnUNet probabilities. It only label-smooths the hard integer target inside `MultiClassShapeLoss` (`models/losses.py` L146-159): `eps` mass spread uniformly over off-classes. Same story in training (the bank loads hard integer patches too).
- nnUNet's softmax/logits are not preserved anywhere in the aligned-patch pipeline. A true probability soft-fit would require a new data-gen stage that saves per-class probability patches (canonical-aligned), plus loss-path changes to consume soft targets. That is out of scope for the current plan.
- RECOMMENDATION (pending your call): keep the existing hard-mask + label-smoothing soft-fit for v1, and make label smoothing the default at test time. If you want a true nnUNet-softmax soft-fit, that is a separate prep task (flagged below as an open decision).

Problem 1 — train/test alpha_nn distribution gap is real and unmeasured today.
- Training alpha_nn will be a joint-optimized per-case embedding row (grad from term 2). Test-time alpha_nn_test is fit from scratch by `optimize_latent` (init `N(0, 1e-4)`, net frozen). Standard AutoDecoders have a train/test latent gap, so Delta learned on training latents may not transfer.
- This cannot be measured until a model is trained, so it becomes a dedicated validation step AFTER Phase 2 training, BEFORE committing to the design (new Phase 2.5 below). I will only report numbers; I will not change the training latent scheme without your go-ahead.

## Open decision for you

- Problem 5: keep hard-mask + label-smoothing soft-fit for v1 (recommended), or schedule a separate nnUNet-softmax probability-patch prep task now?

## Design (math)

Per item, sample GT coords `x_gt` (dense sub-patch grid) and nnUNet coords `x_nn` (sparse grid):
- L_recon_GT = DiceCE( F(alpha_GT, x_gt), onehot_GT )  -> grads to F + alpha_GT
- L_recon_nn = DiceCE( F(alpha_nn, x_nn), onehot_nn )  -> grads to F + alpha_nn
- L_denoise  = DiceCE( F( sg[alpha_nn] + Delta(sg[alpha_nn]), x_gt), onehot_GT ) + eta * ||Delta(sg[alpha_nn])||^2  -> grads to F + Delta only
- L_total = L_recon_GT + lambda_nn * L_recon_nn + lambda_denoise * L_denoise  (+ existing L2 latent reg on alpha_GT and alpha_nn)

`sg[.]` = `.detach()`. The detach on alpha_nn in term 3 is what routes grads correctly; this falls out automatically from the two separate forwards.

## Config block (all new behavior gated)

Add to the training config (e.g. a new `configs/train_v7_denoise.yaml` cloned from `train_v6_5.yaml`) and read in `train.py`:

```yaml
denoise:
  enabled: true          # master off -> original single-latent CNISP
  use_alpha_nn: true     # off -> single-latent CNISP (alpha_GT only)
  use_delta: true        # off -> dual-latent reconstruction only (no L_denoise)
  lambda_nn: 0.5          # try {0.3, 0.5, 1.0}
  lambda_denoise: 1.0     # try {0.5, 1.0, 2.0}
  eta: 1.0e-2             # Delta L2 residual reg; try {1e-3, 1e-2, 1e-1}
  delta_hidden_dim: null  # null -> = latent_dim
  delta_num_hidden_layers: 2
  # Problem 2b: alpha_nn must be driven by term 2, NOT pulled to origin by L2.
  # Separate (smaller) L2 weight for alpha_nn. null -> 0.5 * lat_reg_lambda.
  lambda_l2_nn: null
  # Problem 6: disagreement-region (nnUNet vs GT) upweighted sampling for the
  # GT-coord draws (term 1 / term 3). Built but OFF by default for the baseline.
  boundary_weighted_sampling: false
  boundary_weight: 4.0    # relative sampling weight for disagreement voxels
```

Ablation: `enabled:false` -> original; `use_alpha_nn:false` -> single-latent CNISP; `use_delta:false` -> pure dual-latent recon.

## Phase 1 — dual latent + dataloader (both one-hots per item)

- `engine/dataset.py`: add `train_supervision == "dual"` (or a `return_both` flag). In `__getitem__`, when active and `bank_obs_source[item]=="nnunet"`, return BOTH:
  - `labels_gt`,`coords_gt`,`spacings_gt`,`offsets_gt` sampled from `labels_dense_sub`/`spacings_dense_sub`/`offsets_dense_sub`
  - `labels_nn`,`coords_nn`,`spacings_nn`,`offsets_nn` sampled from `labels_sparse`/`spacings_sparse`/`offsets_sparse`
  Keep fixed `num_points` sampling so default collate works (both are `[num_pts,1,1]`). Add an assert/skip for non-nnunet items so a misconfigured `obs_sources` fails loudly.
- Problem 6 (boundary-weighted sampling, built but default OFF): in `__getitem__`, when `denoise.boundary_weighted_sampling`, the GT-coord draws (used by term 1 and term 3) sample voxels with probability upweighted by `boundary_weight` inside the disagreement region between `labels_dense_sub` and the nnUNet `labels_sparse` resampled onto the dense grid (host-vs-GT working ROI, like corrective learning). Implementation note: both sub-patches already share the dense frame (see Problem 3), so the disagreement mask is `argmax_nn_on_dense != labels_dense_sub`; precompute per item or compute on the fly. Off by default -> uniform sampling (baseline unchanged).
- `engine/train.py`: add `latents_nn = nn.Parameter([n_train, latent_dim])` alongside `latents_train` (rename usage to `latents_gt` internally). Problem 2a: initialize `latents_nn` with an INDEPENDENT RNG draw (its own `torch.Generator` seed), NOT a copy/share of `latents_gt`. Add a param group for it (lr = `learning_rate_lat`).
- Smoke test (Phase 1): `cd orbital_shape_prior_st1 && python -m diagnostics.denoise_smoke --phase dual --paths configs/paths.yaml --config configs/train_v7_denoise.yaml` -> builds a tiny 1-batch loader; then:
  - asserts `labels_gt`/`labels_nn` shapes match;
  - Problem 3 ASSERTION: the decoder normalization is `local = coords - latent_coords` with `latent_coords = image_size/2` (`models/multiclass_ad.py` L62, L74) — identical for both draws. Print and compare, per item: `offsets_gt` vs `offsets_nn`, the sub-patch frame origin (`sub_offset_dense_local` vs `sub_offset_sparse_local`), and `latent_coords`; assert coords_gt and coords_nn live in the SAME physical sub-patch frame (shared origin + scale) and that the inner-crop centroid used to center both came from the SAME source (the sparse view's visible-LCC centroid in `inner_crop_64mm`), so alpha_nn decoded on x_gt is not spatially shifted. Fail loudly if frames diverge.
  - runs L_recon_GT+L_recon_nn forward/backward, prints `.grad is None?` for alpha_GT, alpha_nn, F (expect alpha_GT grad only from term1, alpha_nn only from term2, F from both).

## Phase 2 — Delta module + three-term loss

- New `models/denoise.py`: `LatentDenoiser(latent_dim, hidden_dim, num_hidden_layers)` — MLP `latent_dim -> hidden -> ... -> latent_dim`, ReLU, FINAL LAYER ZERO-INIT (weight+bias = 0) so initial output ~= 0 (identity correction). `forward(z) -> residual`.
- `engine/train.py`:
  - Instantiate `delta` when `use_delta`; add to optimizer (lr = network lr group).
  - Rewrite `train_one_epoch` loss assembly to compute the three terms with `detach()` per the math above; sum into `L_total`. Log each term separately to TensorBoard.
  - Problem 2b (L2 that does not collapse alpha_nn): apply the existing ramped L2 to alpha_GT with `lat_reg_lambda`, and apply a SEPARATE, smaller L2 to alpha_nn with `denoise.lambda_l2_nn` (default `0.5 * lat_reg_lambda`). This keeps term 2 dominant over alpha_nn's position so it encodes the nnUNet noise rather than being pulled to the origin.
  - Problem 2a + health metrics (TensorBoard): each epoch log two CORE health scalars in addition to the three loss terms:
    - `health/mean_alpha_gap` = mean ||alpha_nn - alpha_GT|| (over the epoch's batch rows, plus their individual norms). If this trends toward 0, alpha_nn is being assimilated by GT -> framework collapse; this is the primary health signal.
    - `health/mean_delta_norm` = mean ||Delta(alpha_nn)|| (is Delta actually moving, or crushed by eta?).
  - Checkpoint dict: add `"latents_nn": latents_nn` and `"delta": delta.state_dict()` to `best_checkpoint.pth` and rolling checkpoints (so `model_state` carries them; loader stays back-compatible when keys are absent).
- Smoke test (Phase 2): `cd orbital_shape_prior_st1 && python -m diagnostics.denoise_smoke --phase full ...` -> runs all three terms + backward on one batch and asserts the gradient-routing contract:
  - alpha_GT.grad: non-None (term1 only)
  - alpha_nn.grad: non-None (term2 only); verify it is UNAFFECTED by term3 by checking grad equals a term2-only rerun
  - F params.grad: non-None
  - delta params.grad: non-None; and with `use_delta:false`, delta absent and L reduces to dual recon
  - At init, `||Delta(z)||` ~= 0 (zero-init check)
  - prints the two health scalars (`mean_alpha_gap`, `mean_delta_norm`) so the wiring is verified before any real run.

## Phase 2.5 — train/test alpha_nn gap validation (Problem 1; run before committing to full training)

- New `diagnostics/denoise_latent_gap.py` (lazy imports): after a Phase-2 model exists, take N training cases, RE-FIT their alpha_nn from scratch with the test-time `optimize_latent` against the same nnUNet observation, and compare to the stored training `latents_nn` row per case.
- Report: mean/max `||alpha_nn_train - alpha_nn_testfit||`, and the norms `||alpha_nn_train||`, `||alpha_nn_testfit||`, plus the ratio `gap / ||alpha_nn_train||`.
- Decision gate: if the relative gap is large (e.g. > 20%), STOP and report the numbers to you. Possible remedy (NOT applied unilaterally): obtain training alpha_nn via a test-time-like fit so Delta sees the same input distribution. Report first.
- One-line: `cd orbital_shape_prior_st1 && python -m diagnostics.denoise_latent_gap --paths configs/paths.yaml --config configs/train_v7_denoise.yaml --model-name <name> --n-cases 16`

## Phase 3 — test-time path

- `engine/infer.py`:
  - In `infer_test_set`, after loading the checkpoint, read `delta_state = model_state.get("delta", None)` and `latents_nn = model_state.get("latents_nn", None)`.
  - Problem 4 (no silent collapse): if `denoise.use_delta` is true but `delta_state is None`, RAISE a clear error ("config requests use_delta but the loaded checkpoint contains no 'delta' weights — re-train with denoise.use_delta or set use_delta:false"). Do not silently skip. Only build/load `delta` and freeze it (`eval()`, no grad) when both the config asks for it and the state is present.
  - Wrap the optimizer fn: `optimize_fn = functools.partial(optimize_latent, delta=delta)` (delta=None when use_delta is false).
  - `optimize_latent`: add `delta=None` kwarg. Unchanged fit of `alpha_nn_test` against the nnUNet observation. Problem 5: this fit uses the existing HARD-mask target + optional label-smoothing soft-fit (`latent_fit_soft`/`latent_fit_label_smoothing`); keep label smoothing ON by default at test time to retain boundary uncertainty (the nnUNet softmax itself is not in the pipeline — see investigation). AFTER the loop, if `delta is not None`: `alpha_hat = alpha_nn_test + delta(alpha_nn_test)`; return `alpha_hat.detach()`. Net + Delta both frozen; only `alpha_nn_test` is optimized.
  - Consequence: the saved `latents/<case>.npy` and all downstream (`predict_dense`, iso, native map, cache replay) use `alpha_hat`, so no other infer code changes and cache replay reproduces the corrected prediction without needing Delta at replay time.
- Smoke test (Phase 3): `cd orbital_shape_prior_st1 && python -m diagnostics.denoise_smoke --phase infer --model-dir <ckpt>` -> loads a trained/zero-init Delta, fits a latent on a tiny synthetic observation, confirms `alpha_hat` shape `[1,latent_dim]` and that with zero-init Delta `alpha_hat == alpha_nn_test` (no-op), then a non-zero Delta shifts it.

## Phase 4 — thick-mode end-to-end v7 run (after the code is implemented + smoke tests pass)

Goal: train + infer ONLY in thick mode. Training uses thick GT + thick nnUNet obs (dual-latent); test-time optimization uses ONLY the thick nnUNet output as input. First v7 result keeps start=0 only (no start>0 fan-out) to speed up inference.

Config files to author:
- `orbital_shape_prior_st1/configs/train_v7_denoise.yaml` (clone of `train_v6_5.yaml`):
  - `model_name: orbital_ad_v7`
  - `degradation_bank.modes: [thick]` (thick-only), `obs_sources: [nnunet]`, `nnunet_patch_prefix: labels_dataset835_{exp}_train_step_`
  - `train_supervision: dual` (new mode from Phase 1)
  - the full `denoise:` block (Phase-0 config section above, incl. `lambda_l2_nn`, `boundary_weighted_sampling: false`)
- `orbital_shape_prior_st1/configs/test_v7_denoise.yaml` (clone of `test_default.yaml`):
  - `adaptive_step_sweep.start_offsets: [0]` (START=0 ONLY; disables the fan-out)
  - `latent_fit_soft: true`, `latent_fit_label_smoothing: 0.1` (thick deployment soft-fit, Problem 5)
  - `sweep_mode: thick` (degradation matches the thick experiment)
  - keep `test_label_source`/`run_tag` overridable from CLI (set by the infer command below)
- `nnunet/configs_v7.yaml` (clone of `configs_v6_5.yaml`) for the viz/compare orchestration:
  - `cnisp_model_name: orbital_ad_v7`, `cnisp_train_yaml: train_v7_denoise.yaml`
  - `sweep_degrade_mode: thick` (drives `EXP=thick` in `run_pipeline.sh` L555)
  - `start_offsets: [0]` (MUST match test_v7; comment at `configs_v6_5.yaml` L116 requires this)
  - `cnisp_runs_to_compare: [{run_tag: nnunet_pred, method_label: CNISP-nnUNetPred-v7}]`

Pre-flight data existence check (Problem-style gate; report and STOP if missing, do NOT regenerate):
- New `diagnostics/preflight_v7_thick.py` (lazy imports) that verifies, under `aligned_dir` from `paths.yaml`:
  - TRAIN obs (nnUNet thick sparse pred on the train split): `labels_dataset835_thick_train_step_*/` dirs exist and are non-empty (consumed by `_init_degradation_bank` for `obs_sources:[nnunet]`).
  - TEST deployment input (nnUNet thick sparse pred per step): `labels_dataset835_thick_step_*/` dirs exist (read by `engine/infer.py::_build_label_obs_loader` via `step_input_patch_path`).
  - DENSE Dice targets: `labels/atlas_*.nii.gz` (atlas GT) and `labels_dataset835/chk_*.nii.gz` + `metadata_dataset835/` (chk_* dense pred targets).
  - Print a per-bucket count summary; exit non-zero with a clear message if any bucket is empty (these are produced by the nnUNet-side phases `nnunet-predict-sweep-train`, `cnisp-prep-dataset835-gt`, `cnisp-prep-dataset835-sparse` — we assume they already ran).
- One-line: `cd orbital_shape_prior_st1 && python -m diagnostics.preflight_v7_thick -p configs/paths.yaml -c configs/train_v7_denoise.yaml`

Run sequence (only after pre-flight passes):
1. Train (thick, dual-latent): `cd orbital_shape_prior_st1 && python scripts/02_train.py -p configs/paths.yaml -c configs/train_v7_denoise.yaml`
2. Phase 2.5 gap check on `orbital_ad_v7` (Problem 1) — STOP and report if relative gap > 20%.
3. Infer (thick, nnUNet-only input, start=0): `cd orbital_shape_prior_st1 && python scripts/03_infer.py -p configs/paths.yaml -t configs/train_v7_denoise.yaml -c configs/test_v7_denoise.yaml -m orbital_ad_v7 --checkpoint best --test-label-source nnunet_pred --run-tag nnunet_pred --experiment thick`
4. CNISP viz + cross-model comparison vs nnUNet (one pipeline call reusing existing orchestration):
   `bash run_pipeline.sh --config nnunet/configs_v7.yaml --test-config orbital_shape_prior_st1/configs/test_v7_denoise.yaml cnisp-viz compare`
   - produces the CNISP recon summary AND `nnunet/comparison/viz/paired__nnunet_pred__thick/paired_dice_vs_eff_res.png`-style head-to-head plots against the already-computed nnUNet thick results.
   - (Steps 1+3 can also be driven through `run_pipeline.sh ... cnisp-train cnisp-infer-nnunet-pred`, but the direct script calls above are the explicit path; the prep phases are intentionally NOT invoked so existing nnUNet outputs are reused, not regenerated.)

Note on environment: the data dirs in `paths.yaml` are server paths (`/fs5/...`), so this phase runs on the server/GPU host, not the local mac workspace.

## Diagnostics / hygiene

- All smoke/diagnostic code goes in `diagnostics/` (`denoise_smoke.py`, `denoise_latent_gap.py`), using function-body lazy imports (e.g. `from engine.train import ...` inside functions) to break circular imports.
- TensorBoard logs (per epoch): the three loss terms separately AND the two core health scalars `health/mean_alpha_gap` (||alpha_nn - alpha_GT||; collapse detector) and `health/mean_delta_norm` (||Delta(alpha_nn)||; is Delta moving / not crushed by eta). These two say more about whether the mechanism works than the loss values.
- Memory: dual forward roughly doubles activation memory. If OOM, use chunked forward + gradient accumulation over the point dimension (NOT random downsampling). The existing `point_sample_fraction` (0.75) and `batch_size_train` knobs are the first lever.
