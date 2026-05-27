"""
I/O utilities adapted from Amiranashvili et al.
"""

import sys
from io import TextIOWrapper
from pathlib import Path
from typing import Tuple

import numpy as np
import torch


class Logger(TextIOWrapper):
    """Dual logger: stdout + file."""
    def __init__(self, filepath: Path, mode: str):
        super().__init__(sys.__stdout__.buffer)
        self.file = open(filepath, mode)

    def __del__(self):
        self.file.close()

    def write(self, data):
        self.file.write(data)
        sys.__stdout__.write(data)

    def flush(self):
        self.file.flush()
        sys.__stdout__.flush()


class RollingCheckpointWriter:
    """Write checkpoints with automatic deletion of old ones."""

    def __init__(self, base_dir: Path, base_name: str,
                 max_ckpts: int, ext: str = "pth"):
        self.base_dir = base_dir
        self.base_name = base_name
        self.max_ckpts = max_ckpts
        self.ext = ext

    def write_checkpoint(self, model_state: dict, optim_state: dict,
                         n_steps: int, n_epochs: int):
        state = {
            "model_state": model_state,
            "optimizer_state": optim_state,
            "num_steps_trained": n_steps,
            "num_epochs_trained": n_epochs,
        }
        path = self.base_dir / f"{self.base_name}_{n_steps}.{self.ext}"
        torch.save(state, path)

        # Prune old checkpoints
        paths = sorted(
            self.base_dir.glob(f"{self.base_name}_*.{self.ext}"),
            key=lambda p: int(p.stem.split("_")[-1]),
        )
        for p in paths[:-self.max_ckpts]:
            p.unlink()


def load_latest_checkpoint(
    base_dir: Path, base_name: str, ext: str = "pth", verbose: bool = False
) -> Tuple[dict, dict, int, int]:
    """Load the most recent checkpoint."""
    paths = sorted(
        base_dir.glob(f"{base_name}_*.{ext}"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    if not paths:
        raise FileNotFoundError(f"No checkpoints in {base_dir}")

    latest = paths[-1]
    if verbose:
        print(f"Loading checkpoint: {latest}")
    state = torch.load(latest, map_location="cpu")
    return (state["model_state"], state["optimizer_state"],
            state["num_steps_trained"], state["num_epochs_trained"])


