"""Device utilities."""

import torch


def get_device() -> str:
    """Get the best available device (MPS, CUDA, or CPU)."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"
