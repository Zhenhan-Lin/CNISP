"""
Multi-class AutoDecoder for orbital shape prior.

Changes from Amiranashvili (binary → multi-class), following Jansen et al.:
    - OccupancyPredictor last layer: Linear(latent_dim, num_classes)
    - Forward returns [B, *ST, C] class logits
    - Downstream uses softmax + CE + multi-class Dice

Coordinate convention:
    - We use RAS (Amiranashvili uses LPS), but the MLP is agnostic:
      it only sees local_coords = coords - latent_coords, which is
      symmetric around zero regardless of coordinate system.
    - image_size and latent_coords are registered buffers saved with
      the model checkpoint, so they persist across train→inference.
"""

from typing import List

import torch
import torch.nn as nn


class MultiClassOccupancyPredictor(nn.Module):
    def __init__(self, latent_dim, spatial_dim, num_classes, num_layers, layers_with_coords):
        super().__init__()
        self.layers_with_coords = layers_with_coords
        self.num_classes = num_classes

        def block(ch_in, ch_out):
            return nn.Sequential(nn.Linear(ch_in, ch_out), nn.ReLU(True))

        in_ch = [latent_dim] * num_layers
        for lid in layers_with_coords:
            in_ch[lid] = latent_dim + spatial_dim

        self.res_layers = nn.ModuleList(
            [block(in_ch[i], latent_dim) for i in range(num_layers - 1)]
        )
        self.last_layer = nn.Linear(in_ch[-1], num_classes)

    def forward(self, latents, local_coords):
        """[B, *, Z] + [B, *, 3] → [B, *, C]"""
        features = latents
        for i, layer in enumerate(self.res_layers):
            inject = i in self.layers_with_coords
            if inject:
                features = torch.cat([features, local_coords], dim=-1)
            out = layer(features)
            features = out if inject else features + out
        return self.last_layer(features)


class MultiClassAutoDecoder(nn.Module):
    def __init__(self, latent_dim, spatial_dim, image_size, num_classes,
                 num_layers=8, layers_with_coords=None):
        super().__init__()
        if layers_with_coords is None:
            layers_with_coords = [0, 4]
        self.num_classes = num_classes

        self.register_buffer("image_size", image_size)
        self.register_buffer("latent_coords", image_size / 2.0)

        self.occp_pred = MultiClassOccupancyPredictor(
            latent_dim, spatial_dim, num_classes, num_layers, layers_with_coords
        )

    def forward(self, latents, coordinates):
        """
        latents:     [B, Z]
        coordinates: [B, *ST, 3] physical coords in mm
        Returns:     [B, *ST, C] class logits
        """
        local_coords = coordinates - self.latent_coords
        spatial_shape = coordinates.shape[1:-1]
        z = latents
        for _ in range(len(spatial_shape)):
            z = z.unsqueeze(1)
        z = z.expand(-1, *spatial_shape, -1)
        return self.occp_pred(z, local_coords)

    def predict_dense(self, latent, target_shape, spacing,
                      batch_size_coords=2_000_000,
                      autocast_dtype=None):
        """
        Dense prediction on a full grid, following Amiranashvili's
        generate_sampling_grid with cmin=0, cmax=image_size.

        Coordinate generation (align_corners=False):
            spacing_grid = image_size / target_shape  (= voxel spacing)
            start = spacing_grid / 2
            coords = start, start+spacing_grid, start+2*spacing_grid, ...

        NOTE: we use the case's actual spacing (not image_size/target_shape)
        because in our pipeline spacing is given, not derived.
        The result is identical when target_shape * spacing == image_size,
        which holds for training cases. For test cases it may differ slightly
        if the patch was edge-clipped.

        Args:
            latent: [1, Z]
            target_shape: [3] integer tensor
            spacing: [3] float tensor (mm per voxel)
            batch_size_coords: voxels per forward chunk (no_grad, so chunk
                size only bounds peak intermediate activations, not gradient
                buffers; 2M is safe on 8GB+ GPUs for a 128-wide MLP).
            autocast_dtype: if not None, run forward under
                ``torch.autocast(device.type, dtype=autocast_dtype)`` for
                ~2x speedup; bf16 is recommended on Ampere+.
        Returns:
            class_map: [D1, D2, D3] int32 CPU tensor (argmax of logits)
        """
        device = latent.device
        target_shape = target_shape.long()

        # Check bounds: warn if physical extent exceeds trained image_size
        physical_extent = target_shape.float() * spacing.to(device)
        train_size = self.image_size.detach()
        eps = 0.1  # mm tolerance
        if torch.any(physical_extent > train_size + eps):
            print(f"  WARN: sampling outside trained volume: "
                  f"{physical_extent.tolist()} > {train_size.tolist()}")

        # Generate coordinate grid (align_corners=False)
        coords_1d = [
            torch.arange(int(target_shape[d]), device=device, dtype=torch.float32)
            * float(spacing[d]) + float(spacing[d]) / 2.0
            for d in range(3)
        ]
        grid = torch.meshgrid(coords_1d, indexing="ij")
        coords_flat = torch.stack(grid, dim=-1).reshape(-1, 3)  # [N, 3]
        n_total = coords_flat.shape[0]

        use_amp = (autocast_dtype is not None
                   and autocast_dtype != torch.float32
                   and device.type == "cuda")

        # Pre-allocate GPU label buffer (uint8 fits class_id < 256)
        preds_gpu = torch.empty(n_total, dtype=torch.int32, device=device)

        self.eval()
        with torch.no_grad():
            for start in range(0, n_total, batch_size_coords):
                end = min(start + batch_size_coords, n_total)
                bc = coords_flat[start:end].unsqueeze(0).unsqueeze(2).unsqueeze(2)
                with torch.autocast(device_type=device.type,
                                    dtype=autocast_dtype or torch.float32,
                                    enabled=use_amp):
                    bl = self.forward(latent, bc)
                bl = bl.squeeze(0).squeeze(1).squeeze(1)
                preds_gpu[start:end] = bl.argmax(dim=-1).to(torch.int32)

        return preds_gpu.reshape(
            int(target_shape[0]), int(target_shape[1]), int(target_shape[2])
        ).cpu()
