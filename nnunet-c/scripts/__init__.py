"""Thin CLI entry points over engine/ + lib/ for the nnUNet-C experiment.

  * build_dataset.py       -- CLI: build Dataset855/845 for a control.
  * adapt_checkpoint.py    -- CLI: 1ch->5ch first-conv checkpoint surgery.
  * build_finetune_plan.py -- CLI: merge 835 stats/spacing/arch into 855/845 plan.
  * predict_cascade.py     -- CLI: degraded CT -> nnUNet -> CNISP -> 5ch -> predict.
  * gen_prelabels.sh       -- CNISP test-optimization for a caselist (shell).
"""
