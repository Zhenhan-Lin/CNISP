"""Shared input-staging helpers for nnUNetv2_predict.

The per-source CT staging that turns a list of source CTs into a directory
of ``{source_id}_0000.nii.gz`` symlinks (or freshly written NIfTIs) under
``${work_dir}/input/`` now lives directly in the top-level scripts (they
were short enough that a separate ``data_prep/`` orchestrator was redundant):

  * ``nnunet/prepare_inputs.py``        - stage the original CTs (dense baseline)
  * ``nnunet/prepare_smore_inputs.py``  - stage the SMORE-super-resolved CTs
  * ``nnunet/sparsify_inputs.py``       - stage per-step sparsified CTs
                                          (set keyed off CNISP's sweep_results.pkl)

What remains here are the genuinely shared / heavier data-prep libraries
those scripts import:

  * ``resolve_gt.py``        - source/GT resolution shared across the pipeline
                               (``resolve_sources`` / ``fail_on_missing``).
  * ``synth_train_sweep.py`` - synthesize the (source, step) grid that drives
                               the v6 train-split sparsification.

These carry no argparse/CLI of their own; the top-level scripts wire them
together and share ``nnunet/configs.yaml`` with the rest of the pipeline.
"""
