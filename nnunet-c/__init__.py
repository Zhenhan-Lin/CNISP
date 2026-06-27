"""nnUNet-C ("corrector") experiment package.

A cascade segmentation experiment that asks: does the CNISP shape prior give
the second-stage segmenter information that re-running nnUNet cannot?

Three controls share ONE builder/finetune/predict codebase, switched by config
(`configs/corrector.yaml`):

  * A  -- pure nnUNet, single-channel degraded CT. ALREADY DONE = Dataset835.
  * B  -- stacked nnUNet (Dataset855): ch0 degraded CT + ch1..4 = nnUNet's own
          one-round prediction split into per-class binaries.
  * C  -- CNISP-conditioned (Dataset845): ch0 degraded CT + ch1..4 = CNISP's
          prediction split into per-class binaries.

B and C are structurally identical 5-channel datasets; the only difference is
whether the prelabel channels come from a second nnUNet pass (B) or CNISP (C).
Both finetune from the same Dataset835 weights (first-conv channel adaptation).

Layout (mirrors orbital_shape_prior_st1/ granularity):

  * lib/         -- single-responsibility utilities (config, resample, channels,
                    labels, caselist, prelabel resolution).
  * engine/      -- heavier drivers (the single convert.py converter, raw-dataset
                    helpers, checkpoint surgery, plan merge for finetune).
  * scripts/     -- thin CLIs over engine/lib (build_corrector_dataset = TRAIN,
                    build_corrector_testset = TEST; both call engine/convert.py).
  * diagnostics/ -- smoke tests + the post-preprocess hard gate (lazy imports).
  * configs/     -- corrector.yaml (controls + paths + versions).
  * splits/      -- corrector_train.txt (user-provided source_ids).
  * staging/     -- scratch dir (adapted checkpoint, temp casefiles).

NOTE: nnUNetv2 (training/preprocess/predict) runs on the GPU box. The Python
modules here depend only on numpy / nibabel / torch / json and are importable
locally; the nnUNet CLI steps live in the shell wrappers (run_corrector_data.sh,
run_corrector_cnisp.sh, run_train.sh, run_corrector_predict.sh).
"""
