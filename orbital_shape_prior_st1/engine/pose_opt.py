"""Joint latent-translation test-time optimization for partial-FOV CNISP fitting.

Implements the "pose-aware CNISP fitting" plan: alongside the subject latent ``z``,
optimize a bounded shared 3-D translation ``tau`` of the canonical query coordinates,
so centroid drift from FOV truncation is absorbed by ``tau`` instead of corrupting the
shape-manifold projection of ``z``. See the plan doc for the full derivation.

Coordinate convention (matches ``models.multiclass_ad``): the decoder is called
``net(latent, coords)`` where ``coords`` are the canonical patch coordinates in mm
(the current pipeline's ``x0 = R0 (u - c_hat)``). The correction is simply
``coords + tau`` -- ``tau`` in the SAME mm frame; ``tau = tau_max * tanh(tau_raw)`` keeps
it bounded per axis.

The observation loss is evaluated ONLY inside the acquired-FOV mask ``valid_mask``
(C1: the geometric kept-region mask, NOT the segmentation), so the shape manifold is
free to extrapolate anatomy beyond the FOV boundary.

This module is decoder-agnostic and has no repo imports, so it is unit-testable with a
synthetic decoder (see engine tests). Wiring into the real test-opt loop lives elsewhere.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


# ── masked observation losses (hard multiclass; C2) ───────────────────────────
def masked_cross_entropy(logits: torch.Tensor, target_labels: torch.Tensor,
                         valid_mask: torch.Tensor) -> torch.Tensor:
    """Mean CE over voxels with ``valid_mask==1``. ``logits`` [N, K], ``target_labels``
    [N] (int), ``valid_mask`` [N]."""
    per_point = F.cross_entropy(logits, target_labels, reduction="none")
    m = valid_mask.reshape(-1).float()
    return (per_point * m).sum() / m.sum().clamp_min(1.0)


def masked_soft_dice_loss(probs: torch.Tensor, target_onehot: torch.Tensor,
                          valid_mask: torch.Tensor, eps: float = 1e-6,
                          include_background: bool = False) -> torch.Tensor:
    """1 - mean foreground soft-Dice, masked. ``probs``/``target_onehot`` [N, K],
    ``valid_mask`` [N] or [N,1]."""
    m = valid_mask.reshape(-1, 1).float()
    p, t = probs * m, target_onehot * m
    inter = (p * t).sum(dim=0)
    denom = p.sum(dim=0) + t.sum(dim=0)
    dice = (2.0 * inter + eps) / (denom + eps)
    dice = dice if include_background else dice[1:]      # drop channel 0 (background)
    return 1.0 - dice.mean()


# ── joint latent-translation optimizer ────────────────────────────────────────
def optimize_latent_translation(
    net,
    coords: torch.Tensor,             # [N, 3] canonical patch coords (mm) = R0 (u - c_hat)
    observed_labels: torch.Tensor,    # [N] int class labels (hard; C2)
    valid_mask: torch.Tensor,         # [N] 1 ONLY inside the acquired-FOV box (C1)
    latent_dim: int,
    tau_max_mm,                       # [3] per-axis translation bound (mm)
    tau_sigma_mm,                     # [3] per-axis drift scale for the tau penalty (mm)
    *,
    z_init: Optional[torch.Tensor] = None,   # warm-start latent (e.g. latent-only fit); None -> 0
    optimize_tau: bool = True,        # False -> pure latent-only (Condition A / identity)
    optimize_z: bool = True,          # False -> tau-only, z frozen at z_init (Condition E)
    num_translation_steps: int = 75,
    num_joint_steps: int = 500,
    num_latent_refine_steps: int = 150,
    lr_z: float = 1e-3,
    lr_tau: float = 5e-3,
    lambda_dice: float = 1.0,
    lambda_ce: float = 1.0,
    lambda_z: float = 1e-4,
    lambda_tau: float = 1e-2,
    grad_clip: float = 1.0,
    verbose: bool = False,
) -> dict:
    """Fit ``z`` (+ bounded ``tau``) to a masked partial-FOV observation.

    Phased schedule: (1) tau warm-up with z frozen, (2) joint z+tau, (3) latent-only
    refinement with tau frozen. Toggles ``optimize_tau``/``optimize_z`` select the
    ablation conditions (A: tau off; E: z off). Returns the best-objective state:
    ``{z_star [1,Z], tau_star [1,3] (mm), best_loss, tau_hit_bound}``.
    """
    device = coords.device
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)

    num_classes = int(net.num_classes)
    z0 = (torch.zeros(1, latent_dim, device=device) if z_init is None
          else z_init.detach().to(device).reshape(1, latent_dim).clone())
    z = z0.clone().requires_grad_(optimize_z)
    tau_raw = torch.zeros(1, 3, device=device, requires_grad=optimize_tau)
    tau_max = torch.as_tensor(tau_max_mm, dtype=coords.dtype, device=device).reshape(1, 3)
    tau_sigma = torch.as_tensor(tau_sigma_mm, dtype=coords.dtype, device=device).reshape(1, 3)

    labels = observed_labels.reshape(-1).long()
    onehot = F.one_hot(labels, num_classes=num_classes).float()
    m = valid_mask.reshape(-1).float()

    def _loss():
        tau = tau_max * torch.tanh(tau_raw) if optimize_tau else torch.zeros_like(tau_raw)
        # Decoder wants coords as [B, *spatial, 3] (it derives the spatial shape from
        # coords.shape[1:-1] and expands z to match); add the batch dim for [N, 3].
        logits = net(z, (coords + tau).unsqueeze(0))  # net(latent, coords) convention
        logits = logits.reshape(-1, num_classes)
        probs = torch.softmax(logits, dim=-1)
        ce = masked_cross_entropy(logits, labels, m)
        dice = masked_soft_dice_loss(probs, onehot, m)
        reg_z = lambda_z * (z ** 2).mean() if optimize_z else torch.zeros((), device=device)
        reg_tau = (lambda_tau * ((tau / tau_sigma.clamp_min(1e-6)) ** 2).mean()
                   if optimize_tau else torch.zeros((), device=device))
        total = lambda_ce * ce + lambda_dice * dice + reg_z + reg_tau
        return total, tau

    best = {"loss": float("inf"), "z": z.detach().clone(), "tau": torch.zeros(1, 3, device=device)}

    def _track(loss, tau):
        v = float(loss.detach())
        if v < best["loss"]:
            best.update(loss=v, z=z.detach().clone(), tau=tau.detach().clone())

    def _run(params, n, lr_map):
        if not params or n <= 0:
            return
        opt = torch.optim.Adam([{"params": [p], "lr": lr_map[id(p)]} for p in params])
        for _ in range(n):
            opt.zero_grad(set_to_none=True)
            loss, tau = _loss()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
            opt.step()
            _track(loss, tau)

    lrs = {id(z): lr_z, id(tau_raw): lr_tau}
    # Phase 1: tau warm-up (z frozen)
    if optimize_tau:
        z.requires_grad_(False)
        _run([tau_raw], num_translation_steps, lrs)
    # Phase 2: joint
    if optimize_z:
        z.requires_grad_(True)
    joint = ([z] if optimize_z else []) + ([tau_raw] if optimize_tau else [])
    _run(joint, num_joint_steps, lrs)
    # Phase 3: latent-only refinement (tau frozen)
    if optimize_z:
        if optimize_tau:
            tau_raw.requires_grad_(False)
        lrs_ref = dict(lrs); lrs_ref[id(z)] = lr_z * 0.25
        _run([z], num_latent_refine_steps, lrs_ref)

    tau_star = best["tau"]
    hit = bool((tau_star.abs() >= 0.99 * tau_max).any()) if optimize_tau else False
    if verbose:
        print(f"  [pose_opt] best_loss={best['loss']:.4f} tau*={tau_star.reshape(-1).tolist()} "
              f"hit_bound={hit}")
    return {"z_star": best["z"], "tau_star": tau_star, "best_loss": best["loss"],
            "tau_hit_bound": hit}


@torch.no_grad()
def decode_with_tau(net, coords: torch.Tensor, z_star: torch.Tensor,
                    tau_star: torch.Tensor, chunk: int = 200_000) -> torch.Tensor:
    """Decode class probabilities on ``coords`` shifted by the fitted ``tau_star``
    (native-grid decoding, plan §10). Returns [N, K] softmax probs."""
    net.eval()
    shifted = coords + tau_star
    out = []
    for s in range(0, shifted.shape[0], chunk):
        logits = net(z_star, shifted[s:s + chunk].unsqueeze(0))  # [B, N, 3] convention
        logits = logits.reshape(-1, int(net.num_classes))
        out.append(torch.softmax(logits, dim=-1))
    return torch.cat(out, dim=0)
