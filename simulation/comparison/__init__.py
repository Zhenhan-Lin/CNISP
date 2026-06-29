"""Cross-method comparison subsystem (CNISP vs nnUNet-sparse vs nnUNet-C).

Moved here from ``nnunet/`` so the comparison/visualization layer lives in
the shared ``simulation`` package alongside the degradation operators it is
built to evaluate. Each module keeps a ``run(args)`` entry point plus a thin
argparse CLI, and the ``compare`` phase of ``run_pipeline.sh`` drives them:

  * :mod:`simulation.comparison.compare_native`   -- per-source paired Dice
        (nnUNet-sparse vs one CNISP run, plus optional nnUNet-C and
        nnUNet-interp columns), writes ``paired_per_source__<tag>__<exp>.csv``
        to the repo-level ``comparison/`` dir.
  * :mod:`simulation.comparison.method_summary`   -- per-method by-eff_res
        figure/CSV bundle.
  * :mod:`simulation.comparison.paired_summary`   -- head-to-head overlay
        (nnUNet-sparse vs CNISP vs nnUNet-C) figures + CSV.
  * :mod:`simulation.comparison.experiment_summary`-- cross thin/thick/real
        aggregation.
  * :mod:`simulation.comparison.nnunet_c`         -- loader that turns the
        nnUNet-C ``eval_corrector`` per-case CSV into paired rows.

The numeric/rendering primitives (Dice, eff_res index, bucketing, matplotlib
writers) still live under ``nnunet.lib`` / ``nnunet.helpers`` because the
non-comparison nnUNet phases (``nnunet-interp``, ``nnunet-native-summary``)
share them; this package imports them rather than duplicating them. The
package is intentionally NOT imported by ``simulation/__init__.py`` so the
lightweight degradation import path never pulls in matplotlib/nibabel.
"""
