#!/usr/bin/env python3
"""CLI: enforce no-leakage asserts and derive the corrector_train casenames file.

Expands corrector_train source_ids -> casenames (both eyes) and writes
``{casefiles_dir}/{corrector_train_casefile}`` so the casename-based CNISP/nnUNet
machinery (03_infer.py, prepare_inputs.py) can consume the corrector split.

Run before Stage 1 (staging CTs) and Stage 3 (CNISP prelabels).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.config import add_repo_to_syspath, load_corrector_config  # noqa: E402

add_repo_to_syspath(__file__)

from lib.caselist import assert_no_leakage, derive_train_casefile  # noqa: E402

_DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "corrector.yaml"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(_DEFAULT_CONFIG))
    args = ap.parse_args()

    cfg = load_corrector_config(args.config, caller_file=__file__)
    assert_no_leakage(cfg)             # aborts on any leakage
    out = derive_train_casefile(cfg)
    n = sum(1 for _ in open(out))
    print(f"[derive] no-leakage OK; wrote {n} casenames -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
