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
                    labels, caselist, prelabel resolution, softlink staging).
  * engine/      -- heavier drivers (dataset assembly, checkpoint surgery, plan
                    merge for finetune).
  * scripts/     -- thin CLIs over engine/lib.
  * diagnostics/ -- smoke tests + the post-preprocess hard gate (lazy imports).
  * configs/     -- corrector.yaml (controls + paths + versions).
  * splits/      -- corrector_train.txt (user-provided source_ids).
  * staging/     -- per-case softlinks (ct/prelabel/gt) created at build time.

NOTE: nnUNetv2 (training/preprocess/predict) runs on the GPU box. The Python
modules here depend only on numpy / nibabel / torch / json and are importable
locally; the nnUNet CLI steps live in the shell wrappers + run_full_pipeline.sh.
"""
