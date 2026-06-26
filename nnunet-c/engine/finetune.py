"""First-conv 1ch->Nch checkpoint surgery for finetuning Dataset835 weights.

nnUNet's stock ``-pretrained_weights`` loader asserts every non-seg weight
matches shape, so it fails on the first convolution when the target network has
5 input channels but the pretrained one has 1. This module expands the first
conv weight to N input channels:

  * channel 0  <- the pretrained CT weights (so ch0 keeps its learned behaviour)
  * channels 1..N-1 <- zeros (mask channels start as a no-op => the 5-channel net
    is initially equivalent to the original single-channel nnUNet).
    Use ``mask_init='small_random'`` (x0.01) if zero-init causes dead gradients.

Everything else is copied verbatim, so after surgery the standard
``-pretrained_weights`` path loads cleanly.

torch is imported lazily (only needed when actually running surgery).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

_CKPT_WEIGHTS_KEY = "network_weights"


def _is_first_conv_key(key: str, tensor) -> bool:
    """First-stage input conv weight with a single input channel.

    Matches both PlainConvUNet (`encoder.stages.0.0.convs.0.conv.weight` and its
    `...all_modules.0.weight` alias) and ResidualEncoderUNet
    (`encoder.stages.0.blocks.0.conv1.conv.weight`), while avoiding deeper
    1-in-channel convs by requiring the key to live in encoder stage 0.
    """
    if not key.endswith(".weight"):
        return False
    if "encoder.stages.0" not in key:
        return False
    if tensor.ndim not in (4, 5):   # 2D or 3D conv weight [out, in, *k]
        return False
    return int(tensor.shape[1]) == 1


def adapt_first_conv_state_dict(
    state_dict: Dict,
    n_new_channels: int,
    mask_init: str = "zero",
    scale: float = 0.01,
) -> Tuple[Dict, Dict]:
    """Return (new_state_dict, report) with the first conv expanded to N channels."""
    import torch  # lazy

    if mask_init not in ("zero", "small_random"):
        raise ValueError(f"mask_init must be zero|small_random, got {mask_init!r}")

    new_sd = dict(state_dict)
    adapted: List[Dict] = []
    total_params = sum(int(t.numel()) for t in state_dict.values())
    newly_init = 0

    for key, tensor in state_dict.items():
        if not _is_first_conv_key(key, tensor):
            continue
        out_ch = int(tensor.shape[0])
        kernel = tuple(int(x) for x in tensor.shape[2:])
        new_shape = (out_ch, n_new_channels, *kernel)
        new_w = torch.zeros(new_shape, dtype=tensor.dtype, device=tensor.device)
        new_w[:, 0:1, ...] = tensor                       # ch0 = pretrained CT
        if mask_init == "small_random" and n_new_channels > 1:
            new_w[:, 1:, ...] = torch.randn(
                (out_ch, n_new_channels - 1, *kernel),
                dtype=tensor.dtype, device=tensor.device,
            ) * scale
        new_sd[key] = new_w
        added = out_ch * (n_new_channels - 1) * int(torch.tensor(kernel).prod())
        newly_init += int(added)
        adapted.append({
            "key": key,
            "old_shape": list(tensor.shape),
            "new_shape": list(new_shape),
            "added_params": int(added),
        })

    if not adapted:
        raise RuntimeError(
            "no first-conv (single-input-channel) weight found to adapt; "
            "inspect the checkpoint keys (expected something like "
            "encoder.stages.0.0.convs.0.conv.weight)."
        )

    report = {
        "n_new_channels": n_new_channels,
        "mask_init": mask_init,
        "adapted_layers": adapted,
        "total_params_after": total_params + newly_init,
        "loaded_params": total_params,          # all original values reused
        "newly_init_params": newly_init,        # the added mask-channel slices
    }
    return new_sd, report


def adapt_checkpoint(
    in_path: Path,
    out_path: Path,
    n_new_channels: int = 5,
    mask_init: str = "zero",
) -> Dict:
    """Load an nnUNet checkpoint, expand its first conv, save the adapted copy."""
    import torch  # lazy

    in_path, out_path = Path(in_path), Path(out_path)
    ckpt = torch.load(str(in_path), map_location="cpu", weights_only=False)
    if _CKPT_WEIGHTS_KEY not in ckpt:
        raise KeyError(
            f"checkpoint {in_path} has no '{_CKPT_WEIGHTS_KEY}'; keys="
            f"{list(ckpt.keys())}"
        )
    new_sd, report = adapt_first_conv_state_dict(
        ckpt[_CKPT_WEIGHTS_KEY], n_new_channels, mask_init=mask_init
    )
    ckpt[_CKPT_WEIGHTS_KEY] = new_sd
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, str(out_path))
    report["in_path"] = str(in_path)
    report["out_path"] = str(out_path)
    return report


def print_report(report: Dict) -> None:
    print("── first-conv channel adaptation ──────────────────────────")
    print(f"  in  : {report.get('in_path')}")
    print(f"  out : {report.get('out_path')}")
    print(f"  n_new_channels: {report['n_new_channels']} | "
          f"mask_init: {report['mask_init']}")
    for a in report["adapted_layers"]:
        print(f"  adapted {a['key']}: {a['old_shape']} -> {a['new_shape']} "
              f"(+{a['added_params']} params)")
    print(f"  loaded params      : {report['loaded_params']:,}")
    print(f"  newly-init params  : {report['newly_init_params']:,}")
    print(f"  total params after : {report['total_params_after']:,}")
    print("───────────────────────────────────────────────────────────")
