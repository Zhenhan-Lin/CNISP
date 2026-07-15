"""Step-stratified nnUNet dataloader for the corrector (design §1.3).

Default nnUNet random case sampling makes each batch's `step_size` composition
random, which (with `step_size` a hidden difficulty variable) drives gradient
variance and the averaged trust policy. This subclass fixes the batch's
step composition: each batch draws **one case per step stratum {3,6,9}** plus
**one arbitrary case** (position 0), so with `oversample_foreground_percent=0.75`
nnUNet forces the last 3 positions to foreground and leaves position 0 random ->
"3 foreground strata + 1 background patch".

Only `get_indices()` (the per-batch case-key picker that
`nnUNetDataLoader.generate_train_batch` calls) is overridden; everything else —
patch cropping, oversampling, prev-stage seg loading, transforms — is inherited
unchanged. This is a PURE subclass: the trainer's `get_dataloaders` builds this
class instead of the stock `nnUNetDataLoader`; nothing in site-packages is
touched, so only `-tr nnUNetTrainer_OrbitalCascade` runs are affected.

The `case_id -> step_size` key is parsed from the preprocessed identifier
`corr_{sid}_step{XX}` with a trailing-anchored regex (`sid` itself contains
underscores, so anchor on the final `_step\d+`).

Deps: numpy + nnunetv2 (installed on the GPU box).
"""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np

from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader

_STEP_RE = re.compile(r"_step(\d+)$")


def step_of(identifier: str):
    """step_size from a `corr_{sid}_step{XX}` identifier, or None."""
    m = _STEP_RE.search(str(identifier))
    return int(m.group(1)) if m else None


class StepStratifiednnUNetDataLoader(nnUNetDataLoader):
    """nnUNetDataLoader that fixes each batch's step-stratum composition.

    batch layout per `get_indices()`: [arbitrary(bg), step=strata[0], strata[1], ...].
    Set the trainer's ``oversample_foreground_percent = len(strata) / batch_size``
    (0.75 for batch 4 / 3 strata) so positions 1.. are forced foreground and
    position 0 is a random-location (background) patch. ``batch_size`` MUST equal
    ``1 + len(strata)`` (the trainer overrides the plan's batch_size accordingly).
    """

    def __init__(self, *args, strata: Sequence[int] = (3, 6, 9), **kwargs):
        super().__init__(*args, **kwargs)
        self.strata = tuple(int(s) for s in strata)
        # group preprocessed identifiers by step_size
        self._by_step: dict[int, list] = {s: [] for s in self.strata}
        n_unmatched = 0
        for k in self.indices:                       # set by nnUNetDataLoader.__init__
            s = step_of(k)
            if s in self._by_step:
                self._by_step[s].append(k)
            elif s is None:
                n_unmatched += 1
        empty = [s for s in self.strata if not self._by_step[s]]
        if empty:
            raise RuntimeError(
                f"StepStratified: no training cases for step(s) {empty}. "
                f"Identifiers must end in `_step{{XX}}` and cover {list(self.strata)}."
            )
        if self.batch_size != 1 + len(self.strata):
            raise RuntimeError(
                f"StepStratified expects batch_size == 1 + len(strata) = "
                f"{1 + len(self.strata)}, got {self.batch_size}. Set the plan's "
                f"batch_size accordingly in the trainer."
            )
        if n_unmatched:
            print(f"[StepStratified] {n_unmatched} identifier(s) had no _step tag "
                  f"(usable only as the position-0 background draw).")

    def get_indices(self):
        # position 0 = arbitrary case (random-location => background under the
        # 0.75 oversample); positions 1.. = one case per step stratum (forced fg).
        keys = [self.indices[np.random.randint(len(self.indices))]]
        for s in self.strata:
            pool = self._by_step[s]
            keys.append(pool[np.random.randint(len(pool))])
        return keys
