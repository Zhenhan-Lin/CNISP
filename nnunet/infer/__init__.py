"""Inference-side artifact builders.

Each script in this folder produces a derived NIfTI artifact downstream
of nnUNet inference (or, in SMORE's case, of an inference-like
super-resolution pass). They are *not* the predict-time shell wrappers
(those live at ``nnunet/run_predict_*.sh``).

  * ``build_smore_test_images.py``    - run SMORE on the 31 source CTs
                                        (writes ${smore_out_root})
  * ``build_cnisp_native_sweep.py``   - backfill CNISP step_XX preds to
                                        native_space_step_XX (no-op if
                                        engine/infer.py already wrote them)
  * ``upsample_sparse_preds.py``      - NN-upsample nnUNet sparse-CT preds
                                        back to the native CT grid,
                                        symlink step_01 to the dense
                                        baseline, emit the sweep manifest
"""
