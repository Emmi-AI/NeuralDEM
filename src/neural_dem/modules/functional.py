import torch


def modulate_scale_shift(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)
    return x * (1 + scale) + shift


def modulate_gate(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        gate = gate.unsqueeze(1)
    return gate * x
