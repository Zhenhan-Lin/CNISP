"""Diagnostics + gates for the nnUNet-C experiment (lazy imports inside fns).

  * smoke_dataset.py      -- validate a few assembled raw cases (shapes/affines/
                             channel counts/label ranges) before preprocessing.
  * check_preprocessed.py -- pothole-4 HARD GATE: load one preprocessed 855/845
                             case and assert CT-normalization consistency with
                             835, binary ch1-4, identical shapes, labels {0..4}.

Heavy/optional deps (torch, blosc2, nnunetv2 readers) are imported lazily inside
function bodies so importing this package never triggers a circular/heavy import.
"""
