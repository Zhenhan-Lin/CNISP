"""Heavier drivers for the nnUNet-C corrector experiment.

Called by the thin CLIs under scripts/:

  * convert.py       -- THE single per-(case,step) converter (CNISP/nnUNet mask
                        -> 5-ch nnUNet-C input); shared by the train + test builders.
  * build_dataset.py -- raw-dataset path + dataset.json helpers (_raw_root /
                        _dataset_dir / _write_dataset_json) used by the train builder.
  * finetune.py      -- first-conv 1ch->5ch checkpoint surgery so Dataset835
                        weights load into the 5-channel network (+ param report).
  * plan_merge.py    -- override a freshly-planned 855/845 plan's ch0 intensity
                        stats + target spacing + architecture to equal 835
                        (potholes 1 & 3), keeping its 5-entry per-channel lists.
"""
