"""Single-responsibility utilities for the nnUNet-C corrector experiment.

Modules (one job each, imported by engine/ and scripts/):

  * config.py    -- YAML loading, repo-root + sys.path bootstrap, corrector
                    config resolution (merges nnunet/configs_v7.yaml paths).
  * resample.py  -- build a per-case reference grid at the 835 plan spacing and
                    resample images/masks onto it (pothole-2 a-ii no-op trick).
  * labels.py    -- label-scheme remapping via nnunet/data_prep/resolve_gt.py.
  * channels.py  -- split a multi-class mask into per-class binaries; assemble +
                    geometry-assert a case's channels; ch0 degraded-source pin.
  * caselist.py  -- read split txts, source_id<->casename, leakage asserts.
  * prelabel.py  -- resolve B/C prelabel masks from ONE nnUNet prediction.
  * staging.py   -- softlink ct/prelabel/gt per case + staging_manifest.json.
"""
