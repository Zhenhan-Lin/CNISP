"""Inference-side and visualization drivers for the nnUNet pipeline.

Mirrors ``orbital_shape_prior_st1/engine/`` in role: everything that
*consumes* model predictions (or, in SMORE's case, runs an
inference-like super-resolution pass) and *emits* derived artifacts
lives here. The lightweight CLI wrappers stay one level up at
``nnunet/run_*.sh``.

Inference-side artifact builders:

  * ``build_smore_test_images.py``    - run SMORE on the 31 source CTs
                                        (writes ``${smore_out_root}``)
  * ``build_cnisp_native_sweep.py``   - backfill CNISP step_XX preds to
                                        ``native_space_step_XX/`` (no-op
                                        if engine/infer.py already wrote
                                        them)
  * ``upsample_sparse_preds.py``      - NN-upsample nnUNet sparse-CT
                                        preds back to the native CT
                                        grid, symlink step_01 to the
                                        dense baseline, emit the sweep
                                        manifest

Visualization / summary builders:

  * ``build_method_summary.py``       - per-method (CNISP or nnUNet)
                                        by-eff_res Dice tables and
                                        the configured PNG. Driven from
                                        ``${work_dir}/comparison/paired_per_source__<run_tag>.csv``
                                        so both methods share one
                                        source of truth.
"""
