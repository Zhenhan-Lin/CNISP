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
installed copy.

Why we override ``initialize()`` and NOT ``__init__``
----------------------------------------------------
Stock ``nnUNetTrainer.__init__`` records its constructor arguments with
``inspect.signature(self.__init__).parameters`` + ``locals()[k]``. Because
``self`` is the subclass instance, ``self.__init__`` resolves to OUR signature;
if we override ``__init__`` as ``(*args, **kwargs)`` the parameter names become
``args``/``kwargs`` and the parent's ``locals()['args']`` raises
``KeyError: 'args'``. Mirroring the exact parent signature is brittle (it
differs across nnUNet versions), so instead we leave ``__init__`` untouched and
set the schedule in ``initialize()`` -- which runs ``configure_optimizers()``
(it reads ``self.initial_lr`` / ``self.num_epochs`` and builds the PolyLR
horizon from ``self.num_epochs``). ``initialize()`` is called before the
training loop for both fresh runs and ``--c`` (continue), so the schedule is in
effect in time, and the checkpoint only restores ``current_epoch`` / weights /
optimizer state (never ``num_epochs``).

This file is COPIED into the installed ``nnunetv2/training/nnUNetTrainer/``
package by ``run_train.sh`` / ``run_corrector_predict.sh`` so that
``nnUNetv2_train/predict -tr nnUNetTrainer_corrector`` can discover it.
"""

from __future__ import annotations

import os

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_corrector(nnUNetTrainer):
    def initialize(self):
        self.num_epochs = int(os.environ.get("CORRECTOR_EPOCHS", "200"))
        self.initial_lr = float(os.environ.get("CORRECTOR_LR", "0.005"))
        super().initialize()
