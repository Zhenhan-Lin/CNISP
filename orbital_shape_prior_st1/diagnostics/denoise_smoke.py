"""
Smoke tests for the CNISP latent-denoise framework.

Self-contained: the gradient-routing / zero-init / inference checks run on
SYNTHETIC tensors so they work anywhere torch is installed (no data needed).
When ``--paths`` + ``--config`` are also given AND the data is reachable, the
``dual`` phase additionally builds one real training batch and runs the
Problem-3 frame/normalization assertion on it.

Usage (run from orbital_shape_prior_st1/):
    python -m diagnostics.denoise_smoke --phase dual  [-p configs/paths.yaml -c configs/train_v7_denoise.yaml]
    python -m diagnostics.denoise_smoke --phase full
    python -m diagnostics.denoise_smoke --phase infer

All imports are lazy (inside functions) to avoid circular-import issues.
"""

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _tiny_net(latent_dim, num_classes, image_mm=64.0):
    import torch
    from models.multiclass_ad import MultiClassAutoDecoder
    return MultiClassAutoDecoder(
        latent_dim=latent_dim, spatial_dim=3,
        image_size=torch.tensor([image_mm] * 3, dtype=torch.float32),
        num_classes=num_classes, num_layers=4, layers_with_coords=[0, 2],
    )


def _synthetic_batch(B, P, C, image_mm=64.0):
    import torch
    coords_gt = torch.rand(B, P, 1, 1, 3) * image_mm
    coords_nn = torch.rand(B, P, 1, 1, 3) * image_mm
    labels_gt = torch.randint(0, C, (B, P, 1, 1))
    labels_nn = torch.randint(0, C, (B, P, 1, 1))
    return coords_gt, coords_nn, labels_gt, labels_nn


def _check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}")
    if not cond:
        raise AssertionError(name)


def phase_dual(args):
    """Two-term recon: grad routing for alpha_GT / alpha_nn / F."""
    import torch
    import torch.nn as nn
    from models.losses import MultiClassShapeLoss

    print("== phase dual: two-term recon grad routing (synthetic) ==")
    torch.manual_seed(0)
    Z, C, N, B, P = 16, 5, 4, 2, 64
    net = _tiny_net(Z, C)
    crit = MultiClassShapeLoss()
    latents_gt = nn.Parameter(torch.randn(N, Z) * 0.1)
    latents_nn = nn.Parameter(torch.randn(N, Z) * 0.1)
    ids = torch.tensor([0, 1])
    coords_gt, coords_nn, labels_gt, labels_nn = _synthetic_batch(B, P, C)

    z_gt, z_nn = latents_gt[ids], latents_nn[ids]
    l1 = crit(net(z_gt, coords_gt), labels_gt)
    l2 = crit(net(z_nn, coords_nn), labels_nn)
    (l1 + 0.5 * l2).backward()

    g_gt = latents_gt.grad
    g_nn = latents_nn.grad
    f_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                     for p in net.parameters())
    _check("alpha_GT rows in batch receive grad (term1)",
           g_gt is not None and g_gt[ids].abs().sum() > 0)
    _check("alpha_GT rows NOT in batch receive no grad",
           float(g_gt[2:].abs().sum()) == 0.0)
    _check("alpha_nn rows in batch receive grad (term2)",
           g_nn is not None and g_nn[ids].abs().sum() > 0)
    _check("F receives grad (both terms)", f_has_grad)

    if args.paths and args.config:
        _dataset_frame_assertion(args)
    else:
        print("  (skip dataset frame assertion: pass -p/-c with reachable data)")
    print("phase dual: OK")


def _dataset_frame_assertion(args):
    """Problem 3: coords_gt and coords_nn share the same physical sub-patch
    frame + decoder normalization. Builds ONE real training batch."""
    import yaml
    import torch
    from engine.dataset import create_data_loader, PhaseType

    print("-- Problem 3: real-batch frame/normalization assertion --")
    with open(args.paths) as f:
        params = yaml.safe_load(f)
    with open(args.config) as f:
        params.update(yaml.safe_load(f))
    params.setdefault("train_supervision", "dual")

    dl = create_data_loader(params, PhaseType.TRAIN, verbose=True)
    image_size = dl.dataset.image_size  # [3] mm
    batch = next(iter(dl))
    for k in ("coords_gt", "coords_nn", "spacings", "offsets",
              "spacings_nn", "offsets_nn"):
        _check(f"batch has key '{k}'", k in batch)

    cg = batch["coords_gt"].reshape(-1, 3)
    cn = batch["coords_nn"].reshape(-1, 3)
    print(f"  image_size (mm)      : {image_size.tolist()}")
    print(f"  latent_coords (=/2)  : {(image_size / 2).tolist()}")
    print(f"  spacings_gt[0]       : {batch['spacings'][0].tolist()}")
    print(f"  offsets_gt[0]        : {batch['offsets'][0].tolist()}")
    print(f"  spacings_nn[0]       : {batch['spacings_nn'][0].tolist()}")
    print(f"  offsets_nn[0]        : {batch['offsets_nn'][0].tolist()}")
    print(f"  coords_gt range      : {cg.min(0).values.tolist()} .. {cg.max(0).values.tolist()}")
    print(f"  coords_nn range      : {cn.min(0).values.tolist()} .. {cn.max(0).values.tolist()}")

    # Same frame: both coordinate clouds live inside [0, image_size] (the
    # decoder subtracts the SAME latent_coords for both), so neither is
    # shifted into a different octant.
    ext = image_size.max().item()
    in_frame = (cg.min() >= -2.0 and cg.max() <= ext + 2.0
                and cn.min() >= -2.0 and cn.max() <= ext + 2.0)
    _check("coords_gt and coords_nn lie in the same [0,image_size] frame",
           bool(in_frame))
    # Same center: foreground centroids of the two views agree to within a
    # few mm (both inner-cropped around the SAME sparse visible-LCC centroid).
    lg = batch["labels_gt"].reshape(-1)
    ln = batch["labels_nn"].reshape(-1)
    if (lg > 0).any() and (ln > 0).any():
        cen_gt = cg[lg > 0].mean(0)
        cen_nn = cn[ln > 0].mean(0)
        d = torch.norm(cen_gt - cen_nn).item()
        print(f"  fg-centroid gap (mm) : {d:.2f}")
        _check("GT/nn foreground centroids agree within 12 mm", d < 12.0)
    print("-- frame assertion: OK --")


def phase_full(args):
    """Three-term loss: full grad-routing contract + zero-init + health."""
    import torch
    import torch.nn as nn
    from models.losses import MultiClassShapeLoss
    from models.denoise import LatentDenoiser

    print("== phase full: three-term grad routing + zero-init (synthetic) ==")
    torch.manual_seed(0)
    Z, C, N, B, P = 16, 5, 4, 2, 64
    eta, lam_nn, lam_dn = 1e-2, 0.5, 1.0
    net = _tiny_net(Z, C)
    delta = LatentDenoiser(Z, Z, 2)
    crit = MultiClassShapeLoss()
    latents_gt = nn.Parameter(torch.randn(N, Z) * 0.1)
    latents_nn = nn.Parameter(torch.randn(N, Z) * 0.1)
    ids = torch.tensor([0, 1])
    coords_gt, coords_nn, labels_gt, labels_nn = _synthetic_batch(B, P, C)

    # Zero-init Delta is a no-op.
    z = torch.randn(3, Z)
    _check("zero-init Delta(z) == 0", float(delta(z).abs().max()) == 0.0)

    # Reference: alpha_nn grad from term 2 ALONE (no latent reg).
    l2_only = lam_nn * crit(net(latents_nn[ids], coords_nn), labels_nn)
    l2_only.backward()
    ref_nn_grad = latents_nn.grad[ids].clone()
    net.zero_grad(); latents_gt.grad = None; latents_nn.grad = None
    delta.zero_grad()

    # Full three-term loss (no latent reg, to isolate routing).
    z_gt, z_nn = latents_gt[ids], latents_nn[ids]
    l1 = crit(net(z_gt, coords_gt), labels_gt)
    l2 = crit(net(z_nn, coords_nn), labels_nn)
    z_nn_sg = z_nn.detach()
    resid = delta(z_nn_sg)
    l3 = crit(net(z_nn_sg + resid, coords_gt), labels_gt) \
        + eta * resid.pow(2).sum(1).mean()
    (l1 + lam_nn * l2 + lam_dn * l3).backward()

    _check("alpha_GT receives grad (term1)",
           latents_gt.grad is not None and latents_gt.grad[ids].abs().sum() > 0)
    _check("alpha_nn receives grad",
           latents_nn.grad is not None and latents_nn.grad[ids].abs().sum() > 0)
    _check("alpha_nn grad UNAFFECTED by term3 (== term2-only grad)",
           torch.allclose(latents_nn.grad[ids], ref_nn_grad, atol=1e-6))
    _check("F receives grad",
           any(p.grad is not None and p.grad.abs().sum() > 0
               for p in net.parameters()))
    # At zero-init only Delta's LAST layer gets grad (upstream W=0 blocks body).
    _check("Delta.last_layer receives grad (term3)",
           delta.last_layer.weight.grad is not None
           and delta.last_layer.weight.grad.abs().sum() > 0)

    # Health scalars compute.
    with torch.no_grad():
        gap = torch.mean(torch.norm(z_nn - z_gt, dim=1)).item()
        dnorm = torch.mean(torch.norm(delta(z_nn.detach()), dim=1)).item()
    print(f"  health: mean_alpha_gap={gap:.4f}  mean_delta_norm={dnorm:.6f}")
    _check("mean_delta_norm == 0 at zero-init", dnorm == 0.0)
    print("phase full: OK")


def phase_infer(args):
    """Test-time optimize_latent + Delta: shape, zero-init no-op, non-zero shift."""
    import torch
    from models.denoise import LatentDenoiser
    from engine.infer import optimize_latent

    print("== phase infer: optimize_latent + Delta (synthetic) ==")
    torch.manual_seed(0)
    Z, C, P = 16, 5, 200
    device = torch.device("cpu")
    net = _tiny_net(Z, C).to(device)
    coords = (torch.rand(1, P, 1, 1, 3) * 64).to(device)
    labels = torch.randint(0, C, (1, P, 1, 1)).to(device)

    delta0 = LatentDenoiser(Z, Z, 2).to(device).eval()  # zero-init -> no-op
    lat_noop = optimize_latent(
        net, labels, coords, latent_dim=Z, lr=1e-2, lat_reg_lambda=1e-4,
        num_iters=15, max_num_const_dsc=-1, device=device, verbose=False,
        soft=True, label_smoothing=0.1, delta=delta0,
    )
    _check("alpha_hat shape == [1, Z]", tuple(lat_noop.shape) == (1, Z))

    # Re-fit with delta=None to recover the raw test-fit latent (same seed/iters).
    torch.manual_seed(0)
    lat_raw = optimize_latent(
        net, labels, coords, latent_dim=Z, lr=1e-2, lat_reg_lambda=1e-4,
        num_iters=15, max_num_const_dsc=-1, device=device, verbose=False,
        soft=True, label_smoothing=0.1, delta=None,
    )
    # Same seed path -> the optimisation is identical; zero-init Delta adds 0.
    torch.manual_seed(0)
    lat_noop2 = optimize_latent(
        net, labels, coords, latent_dim=Z, lr=1e-2, lat_reg_lambda=1e-4,
        num_iters=15, max_num_const_dsc=-1, device=device, verbose=False,
        soft=True, label_smoothing=0.1, delta=delta0,
    )
    _check("zero-init Delta is a no-op (alpha_hat == alpha_nn_test)",
           torch.allclose(lat_noop2, lat_raw, atol=1e-6))

    # Non-zero Delta shifts the latent.
    delta1 = LatentDenoiser(Z, Z, 2).to(device).eval()
    with torch.no_grad():
        delta1.last_layer.weight.normal_(0, 0.1)
        delta1.last_layer.bias.normal_(0, 0.1)
    torch.manual_seed(0)
    lat_shift = optimize_latent(
        net, labels, coords, latent_dim=Z, lr=1e-2, lat_reg_lambda=1e-4,
        num_iters=15, max_num_const_dsc=-1, device=device, verbose=False,
        soft=True, label_smoothing=0.1, delta=delta1,
    )
    _check("non-zero Delta shifts the latent",
           not torch.allclose(lat_shift, lat_raw, atol=1e-4))
    print("phase infer: OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["dual", "full", "infer"])
    ap.add_argument("-p", "--paths", default=None)
    ap.add_argument("-c", "--config", default=None)
    args = ap.parse_args()
    {"dual": phase_dual, "full": phase_full, "infer": phase_infer}[args.phase](args)
    print("\nSMOKE OK")


if __name__ == "__main__":
    main()
