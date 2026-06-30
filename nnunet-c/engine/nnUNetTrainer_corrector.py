"""Short-finetune nnUNet trainer for the nnUNet-C corrector controls (B/C).

Identical to stock ``nnUNetTrainer`` except for the finetune schedule:

  * ``num_epochs``  (default 200)   -- we finetune from the Dataset835 weights,
                                       so a full 1000-epoch run is wasteful.
  * ``initial_lr``  (default 0.005) -- gentler than the stock 1e-2 because we
                                       start from pretrained weights rather than
                                       from scratch.

Both are overridable at runtime via the ``CORRECTOR_EPOCHS`` / ``CORRECTOR_LR``
environment variables (``run_train.sh`` exports them from
``corrector.yaml::finetune``), so the schedule is tunable without editing this
installed copy. Setting them in ``__init__`` is enough: nnUNet builds the SGD
optimizer + PolyLR schedule from ``self.initial_lr`` / ``self.num_epochs`` in
``configure_optimizers`` (called after ``__init__``).

This file is COPIED into the installed ``nnunetv2/training/nnUNetTrainer/``
package by ``run_train.sh`` / ``run_corrector_predict.sh`` so that
``nnUNetv2_train/predict -tr nnUNetTrainer_corrector`` can discover it.
"""

from __future__ import annotations

import os

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_corrector(nnUNetTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_epochs = int(os.environ.get("CORRECTOR_EPOCHS", "200"))
        self.initial_lr = float(os.environ.get("CORRECTOR_LR", "0.005"))
