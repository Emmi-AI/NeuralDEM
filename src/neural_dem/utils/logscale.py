import torch


def to_logscale(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


def from_logscale(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (x.abs().exp() - 1)
