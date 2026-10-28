import torch


@torch.inference_mode()
def predict(model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """Run ``model`` on ``inputs``, moved to the model's device."""
    device = next(model.parameters()).device
    return model(inputs.to(device))
