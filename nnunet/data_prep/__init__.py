"""Input staging for nnUNetv2_predict.

Each script in this folder turns a list of source CTs into a directory
of ``{source_id}_0000.nii.gz`` symlinks (or freshly written NIfTIs)
under ``${work_dir}/input/`` that nnUNetv2's CLI can consume
directly.

  * ``prepare_inputs.py``        - stage the original CTs (dense baseline)
  * ``prepare_smore_inputs.py``  - stage the SMORE-super-resolved CTs
  * ``sparsify_inputs.py``       - stage per-step sparsified CTs
                                   (set keyed off CNISP's sweep_results.pkl)

All three are invoked as scripts (``python nnunet/data_prep/<name>.py``)
and share ``nnunet/configs.yaml`` with the rest of the pipeline.
"""
