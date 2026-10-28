from pathlib import Path

import torch


def load_model(
    checkpoint: str | Path,
    model: torch.nn.Module,
    device: str | torch.device = "cpu",
) -> torch.nn.Module:
    """Load a state dict from ``checkpoint`` into ``model`` and set it to eval mode."""
    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    return model.to(device).eval()
