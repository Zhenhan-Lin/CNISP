"""
Self-test for the FOV keep-all fix (whole visible-eye centroid + no
largest-CC stripping under truncation).

Proves, on synthetic masks (no data):
  * INTACT eye (one connected blob): keep_all=True is BYTE-IDENTICAL to the
    default single-largest-CC path -- same mask, same centroid, same counts.
    So thin/thick and the undegraded dense target frame are unchanged.
  * TRUNCATED eye (fragmented into islands): the default keeps only the largest
    fragment and its centroid is biased toward it; keep_all=True keeps every
    fragment and takes the centroid over the WHOLE visible eye -- the convention
    the prior was trained on. Both properties hold at the two centroid sites:
      - data_prep.canonical_align.extract_single_eye_lcc  (disk crop + obs mask)
      - engine.dataset.compute_visible_lcc_centroid_mm     (64 mm inner crop)

Run as a plain script from orbital_shape_prior_st1/ (NOT ``-m``: the diagnostics
package __init__ pulls in resolution_sweep, which triggers a package-init
circular import only under ``-m``):
    python diagnostics/fov_keep_all_selftest.py
"""

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main():
    import numpy as np
    import torch
    from data_prep.canonical_align import extract_single_eye_lcc
    from engine.dataset import compute_visible_lcc_centroid_mm

    # 1) intact single blob -> keep_all must equal the default exactly
    fg = np.zeros((30, 30, 30), bool)
    fg[10:20, 10:20, 10:20] = True
    m0, c0, n0, t0 = extract_single_eye_lcc(fg, keep_all=False)
    m1, c1, n1, t1 = extract_single_eye_lcc(fg, keep_all=True)
    assert np.array_equal(m0, m1), "intact mask changed"
    assert np.allclose(c0, c1), "intact centroid changed"
    assert n0 == n1 == t0 == t1, "intact counts changed"
    print(f"intact:    default == keep_all  (centroid={np.round(c0, 2)}, "
          f"kept={n1}/{t1})")

    # 2) truncated -> two islands. default keeps largest; keep_all keeps all.
    fg = np.zeros((40, 40, 40), bool)
    fg[5:12, 18:22, 18:22] = True      # small island
    fg[20:35, 15:25, 15:25] = True     # large island
    md, cd, nd, td = extract_single_eye_lcc(fg, keep_all=False)
    ma, ca, na, ta = extract_single_eye_lcc(fg, keep_all=True)
    whole = np.array(np.nonzero(fg)).mean(1)
    assert nd < td, "default should drop the small island"
    assert na == ta, "keep_all should keep everything"
    assert np.allclose(ca, whole), "keep_all centroid must be the whole-fg mean"
    assert not np.allclose(cd, ca), "centroids must differ under fragmentation"
    print(f"truncated: default keeps {nd}/{td} (largest), centroid={np.round(cd, 2)}")
    print(f"truncated: keep_all keeps {na}/{ta} (all),     centroid={np.round(ca, 2)}"
          f"  == whole-fg {np.round(whole, 2)}")

    # 3) the inference-side wrapper behaves the same
    vol = torch.from_numpy(fg.astype(np.float32))
    sp = torch.tensor([0.5, 0.5, 0.5])
    of = torch.tensor([0.25, 0.25, 0.25])
    cm_d, _, _ = compute_visible_lcc_centroid_mm(vol, sp, of, keep_all=False)
    cm_a, _, _ = compute_visible_lcc_centroid_mm(vol, sp, of, keep_all=True)
    exp_a = torch.from_numpy(whole.astype(np.float32)) * sp + of
    assert torch.allclose(cm_a, exp_a, atol=1e-4), (cm_a, exp_a)
    assert not torch.allclose(cm_d, cm_a), "inference centroid must differ"
    print(f"inference: default_mm={np.round(cm_d.numpy(), 2)}  "
          f"keep_all_mm={np.round(cm_a.numpy(), 2)} (== whole-fg)")

    print("\nFOV KEEP-ALL SELF-TEST PASSED")


if __name__ == "__main__":
    main()
