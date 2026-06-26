"""Heavier drivers for the nnUNet-C corrector experiment.

Called by the thin CLIs under scripts/:

  * build_dataset.py -- assemble a control's 5-channel (or 1-channel) nnUNet raw
                        dataset (imagesTr/labelsTr/dataset.json) from staging/.
  * finetune.py      -- first-conv 1ch->5ch checkpoint surgery so Dataset835
                        weights load into the 5-channel network (+ param report).
  * plan_merge.py    -- override a freshly-planned 855/845 plan's ch0 intensity
                        stats + target spacing + architecture to equal 835
                        (potholes 1 & 3), keeping its 5-entry per-channel lists.
"""
