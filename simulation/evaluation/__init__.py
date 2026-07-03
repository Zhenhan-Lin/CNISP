"""5-pipeline segmentation-quality evaluation (metrics + figures).

Companion to ``simulation.comparison`` (paired Dice-vs-eff_res curves). Where
comparison shows per-eff_res Dice for nnUNet-sparse / CNISP / nnUNet-C, this
subsystem computes the richer per-structure metrics (volume CoV, volume
agreement / Bland-Altman, ASSD / HD95 / Surface-Dice) across the five pipelines
(nnU-Net, CNISP, nnU->nnU, Proposed, Oracle) and renders the corresponding
figures. Run it alongside the ``compare`` phase.

Layered like ``simulation/comparison/`` for extensibility (adding a figure =
one new ``plots`` function + one thin ``*_summary`` driver, reusing the layers
below):

  computation   :mod:`simulation.evaluation.metrics`    masks -> per-structure rows
  aggregation   :mod:`simulation.evaluation.aggregate`  rows  -> plot-ready inputs
  rendering     :mod:`simulation.evaluation.plots`       inputs -> PNG
  fallback      :mod:`simulation.evaluation.synthetic`   placeholder aggregates

Interface (like comparison's paired CSV): ``metrics.build_metrics_table`` writes
``metrics_long.csv`` from a MASK_INDEX; the drivers consume it.

Drivers (thin ``run(args)`` + CLI):
  * :mod:`simulation.evaluation.build_metrics`             MASK_INDEX -> metrics_long.csv
  * :mod:`simulation.evaluation.volume_stability_summary`  cross-resolution CoV figure
  * :mod:`simulation.evaluation.volume_agreement_summary`  Bland-Altman figure
  * :mod:`simulation.evaluation.surface_quality_summary`   ASSD/HD95/Surface-Dice figure
"""
