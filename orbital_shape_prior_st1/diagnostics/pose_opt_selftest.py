"""
Self-tests for engine/pose_opt.py (joint latent-translation FOV optimizer).

Self-contained: every check runs on a SYNTHETIC decoder (a differentiable ball
indicator), so no data / trained model is needed -- works anywhere torch is
installed. Covers the plan's §14 acceptance checks:

    - known-translation recovery (tau-only, Condition E) -> tau* ~ -drift
    - complete-FOV / no drift                            -> tau* ~ 0
    - identity (optimize_tau=False, Condition A)         -> tau* == 0 exactly
    - masked loss                                        -> corruption OUTSIDE
                                                            the FOV mask does not
                                                            move tau*
    - z gradient flow (decoder depends on z)             -> optimize_z lowers loss
    - decode_with_tau                                    -> valid softmax probs

Usage (run from anywhere):
    python orbital_shape_prior_st1/diagnostics/pose_opt_selftest.py

``pose_opt`` is loaded by file path (not ``from engine.pose_opt import ...``) so this
self-test needs only torch -- it does not trigger the ``engine``/``diagnostics``
package __init__ import chains (tensorboard, data deps, etc.).
"""

import importlib.util
from pathlib import Path

import torch

_POSE_OPT = Path(__file__).resolve().parents[1] / "engine" / "pose_opt.py"
_spec = importlib.util.spec_from_file_location("pose_opt", _POSE_OPT)
_po = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_po)
optimize_latent_translation = _po.optimize_latent_translation
decode_with_tau = _po.decode_with_tau


class _SynthDecoder(torch.nn.Module):
    """fg (class 1) logit high inside a ball of ``radius`` around ``c0``;
    optional +z bias so the decoder actually depends on the latent (z-grad test)."""

    num_classes = 2

    def __init__(self, c0, radius, z_gain=0.0):
        super().__init__()
        self.register_buffer("c0", torch.as_tensor(c0, dtype=torch.float32))
        self.radius = float(radius)
        self.z_gain = float(z_gain)
        self.frozen = torch.nn.Parameter(torch.zeros(1))   # non-empty .parameters()

    def forward(self, latent, coords):
        d2 = ((coords - self.c0) ** 2).sum(-1)
        fg = (self.radius ** 2 - d2) * 0.5 + self.z_gain * latent.sum()
        return torch.stack([torch.zeros_like(fg), fg], dim=-1)     # [.., 2]


def _grid(n=13, half=8.0):
    g = torch.linspace(-half, half, n)
    return torch.stack(torch.meshgrid(g, g, g, indexing="ij"), dim=-1).reshape(-1, 3)


def main():
    torch.manual_seed(0)

    coords = _grid()
    c0, R = [0.0, 0.0, 0.0], 5.0
    net = _SynthDecoder(c0, R)
    mask_all = torch.ones(coords.shape[0])

    def template_labels(shift):
        c = torch.as_tensor(c0) + torch.as_tensor(shift, dtype=torch.float32)
        return (((coords - c) ** 2).sum(-1) < R ** 2).long()

    kw = dict(latent_dim=8, tau_max_mm=[8.0, 8.0, 8.0], tau_sigma_mm=[3.0, 3.0, 3.0],
              num_translation_steps=400, num_joint_steps=0, num_latent_refine_steps=0,
              lr_tau=0.1, lambda_z=0.0, lambda_tau=1e-3)

    # known-translation recovery (tau-only) -> tau* ~ -drift
    d = [3.0, -2.0, 0.0]
    r = optimize_latent_translation(net, coords, template_labels(d), mask_all,
                                    optimize_z=False, **kw)
    tau = r["tau_star"].reshape(-1)
    print("known-shift d=", d, "tau*=", [round(x, 2) for x in tau.tolist()],
          "(expect ~", [-x for x in d], ")")
    assert torch.allclose(tau, -torch.as_tensor(d), atol=0.8), tau
    assert not r["tau_hit_bound"]

    # complete-FOV / no drift -> tau* ~ 0
    r0 = optimize_latent_translation(net, coords, template_labels([0., 0., 0.]),
                                     mask_all, optimize_z=False, **kw)
    print("no-drift tau*=", [round(x, 2) for x in r0["tau_star"].reshape(-1).tolist()],
          "(expect ~0)")
    assert r0["tau_star"].abs().max() < 0.8, r0["tau_star"]

    # identity: optimize_tau=False (Condition A) -> tau stays exactly 0
    obs = template_labels(d)
    rA = optimize_latent_translation(net, coords, obs, mask_all, optimize_tau=False,
                                     optimize_z=True, **{**kw, "num_joint_steps": 50})
    assert float(rA["tau_star"].abs().max()) == 0.0, rA["tau_star"]
    print("Condition-A (tau off): tau*==0 exactly:", rA["tau_star"].reshape(-1).tolist())

    # masked loss: corruption OUTSIDE the FOV mask must not move tau*
    mask_half = (coords[:, 2] >= -2).float()
    obs_g = obs.clone()
    obs_g[mask_half == 0] = 1 - obs_g[mask_half == 0]
    r_m = optimize_latent_translation(net, coords, obs_g, mask_half, optimize_z=False, **kw)
    r_c = optimize_latent_translation(net, coords, obs, mask_half, optimize_z=False, **kw)
    print("masked tau*(corrupt-outside)=",
          [round(x, 2) for x in r_m["tau_star"].reshape(-1).tolist()],
          "tau*(clean)=", [round(x, 2) for x in r_c["tau_star"].reshape(-1).tolist()])
    assert torch.allclose(r_m["tau_star"], r_c["tau_star"], atol=0.3), \
        (r_m["tau_star"], r_c["tau_star"])

    # z gradient flow (decoder depends on z) -> optimize_z lowers loss, z moves
    net_z = _SynthDecoder(c0, R, z_gain=0.5)
    rz0 = optimize_latent_translation(net_z, coords, obs, mask_all, optimize_tau=False,
                                      optimize_z=True, **{**kw, "num_joint_steps": 1,
                                                          "num_translation_steps": 0})
    rz1 = optimize_latent_translation(net_z, coords, obs, mask_all, optimize_tau=False,
                                      optimize_z=True, **{**kw, "num_joint_steps": 200,
                                                          "num_translation_steps": 0})
    print(f"z-grad: loss@1step={rz0['best_loss']:.4f} -> loss@200={rz1['best_loss']:.4f}")
    assert rz1["best_loss"] < rz0["best_loss"] - 1e-4, (rz0["best_loss"], rz1["best_loss"])
    assert not torch.allclose(rz1["z_star"], torch.zeros_like(rz1["z_star"])), "z must move"

    # decode_with_tau -> valid softmax probs
    p = decode_with_tau(net, coords, r["z_star"], r["tau_star"])
    assert p.shape == (coords.shape[0], 2)
    assert torch.allclose(p.sum(-1), torch.ones(coords.shape[0]), atol=1e-4)

    print("\nALL POSE-OPT SELF-TESTS PASSED")


if __name__ == "__main__":
    main()
