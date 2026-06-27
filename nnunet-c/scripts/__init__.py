"""Thin CLI entry points over engine/ + lib/ for the nnUNet-C experiment.

  * build_corrector_dataset.py -- CLI: build TRAIN Dataset855/845 (data/ tree).
  * build_corrector_testset.py -- CLI: build TEST imagesTs + eval map (same converter).
  * adapt_checkpoint.py    -- CLI: 1ch->5ch first-conv checkpoint surgery.
  * build_finetune_plan.py -- CLI: merge 835 stats/spacing/arch into 855/845 plan.
  * gen_prelabels.sh       -- CNISP test-optimization for a caselist (shell).

The single per-(case,step) conversion both builders share lives in
``engine/convert.py::convert_case`` (train writes a GT label, test does not).
"""
