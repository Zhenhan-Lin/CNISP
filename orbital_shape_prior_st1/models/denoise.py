"""
Latent-space denoiser (Delta) for the CNISP denoise framework.

A small shared MLP that maps a (noisy) latent code to a latent RESIDUAL:

    alpha_hat = alpha_nn + Delta(alpha_nn)

It is trained (term 3 of the denoise loss) to navigate a latent that
faithfully encodes the nnUNet observation toward one that decodes the GT
shape under the SAME frozen decoder F. The final layer is zero-initialised so
that at the start of training ``Delta(z) ~= 0`` (identity correction / no-op),
which keeps the early shape-prior learning undisturbed; the residual grows as
the decoder manifold forms.
"""

from typing import Optional

import torch
import torch.nn as nn


class LatentDenoiser(nn.Module):
    """latent_dim -> hidden -> ... -> latent_dim residual MLP (shared, all cases).

    Args:
        latent_dim: dimensionality of the latent code.
        hidden_dim: hidden width (defaults to ``latent_dim`` when None).
        num_hidden_layers: number of hidden ReLU layers (default 2).
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: Optional[int] = None,
        num_hidden_layers: int = 2,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        h = int(hidden_dim) if hidden_dim else int(latent_dim)
        n_hidden = max(1, int(num_hidden_layers))

        layers = []
        in_ch = self.latent_dim
        for _ in range(n_hidden):
            layers.append(nn.Linear(in_ch, h))
            layers.append(nn.ReLU(inplace=True))
            in_ch = h
        self.body = nn.Sequential(*layers)
        self.last_layer = nn.Linear(in_ch, self.latent_dim)

        # Zero-init the FINAL layer so Delta(z) == 0 at init (identity start).
        nn.init.zeros_(self.last_layer.weight)
        nn.init.zeros_(self.last_layer.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """[*, latent_dim] -> [*, latent_dim] residual."""
        return self.last_layer(self.body(z))
