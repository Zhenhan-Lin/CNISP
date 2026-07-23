"""
Self-test for the FOV valid-mask path in engine/infer.py::optimize_latent.

Proves, on a synthetic MultiClassAutoDecoder (no data):
  * valid_mask=None runs the unchanged full-patch fit (thin/thick path).
  * with a valid_mask, corrupting the observation OUTSIDE the mask does NOT change
    the fitted latent (the truncated region contributes nothing to CE or Dice).
  * WITHOUT the mask, the same corruption DOES change the fit -> the mask matters.

Needs torch (+ the model deps engine/infer pulls in). Run as a plain script from
orbital_shape_prior_st1/ (NOT ``-m``: the diagnostics package __init__ imports
resolution_sweep, which engine/infer also imports -> a package-init circular import
only when loaded via ``-m``):
    python diagnostics/fov_valid_mask_selftest.py
"""

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main():
    import torch
    from models.multiclass_ad import MultiClassAutoDecoder
    from engine.infer import optimize_latent

    torch.manual_seed(0)
    dev = torch.device("cpu")
    D = 12
    g = torch.linspace(2.0, 62.0, D)                     # coords in [0,64] mm
    coords = torch.stack(torch.meshgrid(g, g, g, indexing="ij"), -1).unsqueeze(0)
    ball = (((coords[0] - 32.0) ** 2).sum(-1) < 12.0 ** 2).long()
    labels = ball.unsqueeze(0)                           # [1, D, D, D]

    def net():
        torch.manual_seed(1)
        return MultiClassAutoDecoder(
            latent_dim=8, spatial_dim=3, image_size=torch.tensor([64.0] * 3),
            num_classes=2, num_layers=4, layers_with_coords=[0, 2]).eval()

    kw = dict(latent_dim=8, lr=1e-2, lat_reg_lambda=1e-4, num_iters=60,
              max_num_const_dsc=-1, device=dev, verbose=False)

    # 1) None path runs (the unchanged full-patch fit)
    z_none = optimize_latent(net(), labels, coords, **kw, valid_mask=None)
    assert torch.isfinite(z_none).all()
    print(f"valid_mask=None runs; |z|={z_none.norm().item():.3f}")

    # 2) masked: corruption OUTSIDE the mask must not move the fit
    mask = (coords[0, ..., 0].reshape(-1) < 32.0).float()   # keep x<32 half
    flat = labels.reshape(-1).clone()
    flat[mask == 0] = 1 - flat[mask == 0]                   # flip labels outside mask
    labels_corrupt = flat.reshape(labels.shape)
    z_clean = optimize_latent(net(), labels, coords, **kw, valid_mask=mask)
    z_corr = optimize_latent(net(), labels_corrupt, coords, **kw, valid_mask=mask)
    d_masked = (z_clean - z_corr).norm().item()
    print(f"masked  : |z_clean - z_corrupt(outside)| = {d_masked:.5f}  (expect ~0)")
    assert d_masked < 1e-3, d_masked

    # 3) unmasked: the same corruption DOES move the fit -> masking matters
    z_c_u = optimize_latent(net(), labels, coords, **kw, valid_mask=None)
    z_x_u = optimize_latent(net(), labels_corrupt, coords, **kw, valid_mask=None)
    d_unmasked = (z_c_u - z_x_u).norm().item()
    print(f"unmasked: |z_clean - z_corrupt|          = {d_unmasked:.5f}  (expect >>0)")
    assert d_unmasked > 1e-2, d_unmasked

    print("\nFOV VALID-MASK SELF-TEST PASSED")


if __name__ == "__main__":
    main()
