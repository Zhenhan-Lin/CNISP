"""nnUNet-side comparison helpers for the CNISP test set.

This package collects the scripts that:
  * resolve the source CT files for CNISP's 62-eye test set,
  * run nnUNetv2_predict on those CTs,
  * map CNISP's per-step sweep predictions to native head space,
  * compute paired per-source Dice scores against the same native GT,
  * pre-compute SMORE super-resolved CTs for the deferred iso comparison.

Each module is runnable as a standalone script and shares the same
configs.yaml so paths stay consistent across stages.
"""
