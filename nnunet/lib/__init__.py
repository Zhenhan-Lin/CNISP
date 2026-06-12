"""Reusable calculation/rendering primitives for the nnUNet-side tooling.

Where ``nnunet/helpers/`` holds tiny shared utilities (config/YAML, bucket
constants, paired-CSV readers, patch-size resolution, SMORE CLI wrappers),
``nnunet/lib/`` holds the heavier *root functionality* extracted out of the
pipeline orchestrators so those scripts can stay thin:

* :mod:`nnunet.lib.metrics`         -- label-volume IO, world-aware resample,
  binary/per-structure Dice, prediction-offset detection, sweep eff_res
  indexing, test-source resolution, and the nnUNet native-Dice scorer.
  Shared by ``engine/compare_native.py`` and
  ``build_nnunet_native_summary.py`` (previously a cross-file private
  import).
* :mod:`nnunet.lib.native_resample` -- world-coordinate resampling of nnUNet
  plan/iso logits onto the native CT grid (used by
  ``nnunet/predict_sparse_iso.py``).
* :mod:`nnunet.lib.predictor`       -- nnUNet predictor construction +
  per-case inference helpers (model/fold resolution, axis-order detection,
  logits -> segmentation). Used by ``nnunet/predict_sparse_iso.py``.
* :mod:`nnunet.lib.viz`             -- matplotlib (Agg) plotting primitives,
  eff_res/step aggregation, and CSV/TXT table writers shared by the four
  ``build_*_summary.py`` drivers.
* :mod:`nnunet.lib.patches`         -- canonical-align output-directory
  layouts and per-(source, step) input iteration used by the
  ``build_*_patches.py`` drivers.

Orchestration that wires these primitives together lives either in the
substantial ``engine/*.py`` drivers (``compare_native``,
``build_smore_test_images``) or, for the shorter patch/summary builders and
the sparse-sweep predictor, directly in the top-level ``nnunet/<name>.py``
script.
"""
