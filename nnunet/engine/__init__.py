"""Heavier inference-side drivers for the nnUNet pipeline.

Mirrors ``orbital_shape_prior_st1/engine/`` in role: everything that
*consumes* model predictions (or, in SMORE's case, runs an
inference-like super-resolution pass) and is too substantial to read as
a single top-level script lives here as a LIBRARY. Each module exposes a
``run(args)`` entry point and carries no argparse/CLI of its own; the thin
orchestration script that parses args and calls ``run`` lives one level up
at ``nnunet/<name>.py`` (e.g. ``python nnunet/compare_native.py``), and the
shell wrappers stay at ``nnunet/run_*.sh``.

The *calculation/rendering* primitives these orchestrators rely on live in
``nnunet/lib/`` (``lib.metrics`` for label IO / geometry / Dice / eff_res /
source resolution, ``lib.viz`` for the matplotlib + CSV/TXT writers,
``lib.predictor`` + ``lib.native_resample`` for the nnUNet inference path,
``lib.patches`` for patch-builder layouts), so each ``run(args)`` here just
wires those together. ``compare_native`` shares ``lib.metrics`` instead of
cross-importing private ``_`` names.

Only the substantial orchestrators that still warrant their own module keep
a presence here. The short patch- and summary-builders no longer have an
``engine/`` module: their orchestration is short enough to live directly in
the top-level ``nnunet/<name>.py`` script (which imports the same ``lib.*``
primitives), so there is nothing left to factor out.

Remaining modules:

  * ``compare_native.py``         - head-to-head native-space Dice between
                                    CNISP and nnUNet for one CNISP run;
                                    writes ``paired_per_source__<tag>.csv``.
                                    CLI: ``nnunet/compare_native.py``.
  * ``build_smore_test_images.py``- run SMORE on the 31 source CTs
                                    (writes ``${smore_out_root}``).
                                    CLI: ``nnunet/build_smore_test_images.py``.

Self-contained top-level scripts (no ``engine/`` counterpart):

  * ``nnunet/predict_sparse_iso.py``  (uses ``lib.predictor`` +
                                      ``lib.native_resample``)
  * ``nnunet/build_dataset835_canonical_patches.py``
  * ``nnunet/build_dataset835_sparse_patches.py``
  * ``nnunet/build_realpair_patches.py``
  * ``nnunet/build_method_summary.py``
  * ``nnunet/build_paired_summary.py``
  * ``nnunet/build_experiment_summary.py``
  * ``nnunet/build_nnunet_native_summary.py``
  * ``nnunet/build_cnisp_native_sweep.py``
"""
