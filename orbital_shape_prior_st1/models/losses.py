"""
Multi-class loss functions for orbital shape prior.

Following Jansen et al.: L = L_CE + L_Dice + λ·||z||²

Key design decisions:
    - Dice is computed per-class (excluding background) then averaged
    - Optional per-class Dice weights (`dice_class_weights`) to upweight
      small structures (ON, Recti); plain CE on the remaining term.
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def multiclass_dice_coeff(
    probs: torch.Tensor,
    targets_onehot: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
    exclude_bg: bool = True,
    smooth: float = 1e-5,
) -> torch.Tensor:
    """
    Compute mean Dice coefficient across classes.

    Args:
        probs:         [B, *, C] softmax probabilities
        targets_onehot: [B, *, C] one-hot encoded targets
        class_weights: [C] optional per-class weights
        exclude_bg:    skip class 0 (background)
        smooth:        smoothing for numerical stability

    Returns:
        scalar mean Dice coefficient
    """
    num_classes = probs.shape[-1]
    start_class = 1 if exclude_bg else 0

    dice_per_class = []
    for c in range(start_class, num_classes):
        p = probs[..., c].flatten()
        t = targets_onehot[..., c].flatten()
        intersection = (p * t).sum()
        denom = p.sum() + t.sum()
        dice = (2.0 * intersection + smooth) / (denom + smooth)
        dice_per_class.append(dice)

    dice_stack = torch.stack(dice_per_class)

    if class_weights is not None:
        w = class_weights[start_class:]
        w = w / w.sum()
        return (dice_stack * w).sum()

    return dice_stack.mean()


class MultiClassDiceLoss(nn.Module):
    """Soft Dice loss for multi-class segmentation (from logits)."""

    def __init__(
        self,
        class_weights: Optional[List[float]] = None,
        exclude_bg: bool = True,
    ):
        super().__init__()
        self.exclude_bg = exclude_bg
        if class_weights is not None:
            self.register_buffer(
                "class_weights", torch.tensor(class_weights, dtype=torch.float32)
            )
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  [B, *, C] raw logits
            targets: [B, *]   integer class labels
        """
        num_classes = logits.shape[-1]
        probs = F.softmax(logits, dim=-1)
        targets_onehot = F.one_hot(targets.long(), num_classes).float()

        dice = multiclass_dice_coeff(
            probs, targets_onehot, self.class_weights, self.exclude_bg
        )
        return 1.0 - dice


class MultiClassShapeLoss(nn.Module):
    """
    Combined loss: CE + Dice + latent L2.

    L = w_ce * L_CE + w_dice * L_Dice + λ * ||z||²

    The latent regularization term is added externally (in the training loop)
    because it depends on the current epoch (ramp-up schedule).
    This module only handles CE + Dice.
    """

    def __init__(
        self,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        dice_class_weights: Optional[List[float]] = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        # ``label_smoothing`` > 0 turns the hard one-hot CE target into a
        # soft target (1-eps on the observed class, eps/C elsewhere). This
        # is the "soft latent-fit" knob: at test time it stops the latent
        # optimiser from being forced to reproduce every (possibly wrong)
        # observed voxel exactly, which matters when the observation is a
        # noisy nnUNet prediction on a degraded image. Default 0.0 keeps
        # the original hard-label behaviour bit-for-bit.
        self.label_smoothing = float(label_smoothing)
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = MultiClassDiceLoss(dice_class_weights)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        voxel_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            logits:  [B, *, C] raw logits from AutoDecoder
            targets: [B, *]   integer class labels {0, 1, 2, 3, 4}
            voxel_weights: [B, *] optional per-voxel reliability weights for
                the CE term (e.g. nnUNet confidence). When provided, the CE
                term becomes a weighted mean. ``None`` reproduces the plain
                (unweighted) behaviour.
        """
        # CE expects [N, C] logits and [N] targets
        C = logits.shape[-1]
        logits_flat = logits.reshape(-1, C)
        targets_flat = targets.reshape(-1).long()

        if self.label_smoothing > 0.0 or voxel_weights is not None:
            log_probs = F.log_softmax(logits_flat, dim=-1)
            onehot = F.one_hot(targets_flat, C).float()
            if self.label_smoothing > 0.0:
                eps = self.label_smoothing
                soft_target = onehot * (1.0 - eps) + eps / C
            else:
                soft_target = onehot
            ce_per_voxel = -(soft_target * log_probs).sum(dim=-1)
            if voxel_weights is not None:
                w = voxel_weights.reshape(-1).to(ce_per_voxel.dtype)
                loss_ce = (ce_per_voxel * w).sum() / (w.sum() + 1e-8)
            else:
                loss_ce = ce_per_voxel.mean()
        else:
            loss_ce = self.ce_loss(logits_flat, targets_flat)

        loss_dice = self.dice_loss(logits, targets)

        return self.ce_weight * loss_ce + self.dice_weight * loss_dice


class MultiClassDiceMetric(nn.Module):
    """
    Reports per-class and mean Dice for monitoring (not for backprop).
    Takes logits, applies argmax, computes hard Dice.
    """

    def __init__(self, num_classes: int, exclude_bg: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.exclude_bg = exclude_bg

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> dict:
        """
        Returns dict: {"mean": float, "per_class": [float, ...]}
        """
        preds = logits.argmax(dim=-1)  # [B, *]
        start = 1 if self.exclude_bg else 0

        dice_per_class = []
        for c in range(start, self.num_classes):
            p = (preds == c).float().flatten()
            t = (targets == c).float().flatten()
            inter = (p * t).sum()
            denom = p.sum() + t.sum()
            dice = (2.0 * inter + 1e-5) / (denom + 1e-5)
            dice_per_class.append(float(dice))

        return {
            "mean": float(sum(dice_per_class) / max(len(dice_per_class), 1)),
            "per_class": dice_per_class,
        }
