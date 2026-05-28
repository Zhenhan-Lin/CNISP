"""Helpers shared by the nnUNet-side preprocessing/inference scripts.

Submodules:
    * :mod:`nnunet.helpers.config`      -- YAML loading, comma-separated
      CLI parsing, NIfTI-aware filename stem, and ``sys.path``
      bootstrapping (``add_repo_root_to_syspath`` /
      ``add_cnisp_src_to_syspath``) so the script entry points can be
      run as plain ``python nnunet/.../foo.py`` invocations.
    * :mod:`nnunet.helpers.buckets`     -- ``STRUCT_ORDER``,
      ``NNUNET_METHOD_LABEL``, ``DEFAULT_BUCKET_EDGES_MM``, plus
      ``assign_bucket`` / ``bucket_sort_key`` shared by
      ``compare_native.py`` and the two by-eff_res summary builders.
    * :mod:`nnunet.helpers.paired_csv`  -- readers/filters for
      ``comparison/paired_per_source__<run_tag>.csv``, used by both
      ``engine/build_method_summary.py`` and
      ``engine/build_paired_summary.py``.
    * :mod:`nnunet.helpers.patch_size`  -- canonical-align patch-size
      resolver used by ``engine/build_dataset835_canonical_patches.py``
      and ``engine/build_dataset835_sparse_patches.py`` to pin the
      patch extent to whatever the CNISP MLP was trained on.
    * :mod:`nnunet.helpers.smore`       -- IACL ``run-smore`` CLI
      wrappers, NIfTI-header compatibility check, and multi-host
      directory locks.
"""
