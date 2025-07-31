import torch
from torch import nn

from .continuous_sincos_embed import ContinuousSincosEmbed


class ScalarConditioner(nn.Module):
    def __init__(
        self,
        dim: int,
        cond_dim: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        cond_dim = cond_dim or dim * 4
        self.cond_dim = cond_dim
        self.dim = dim
        self.embed = ContinuousSincosEmbed(dim=dim, ndim=1)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, cond_dim),
            nn.GELU(),
        )

    def forward(self, scalar: torch.Tensor) -> torch.Tensor:
        # checks + preprocess
        assert scalar.numel() == len(scalar)
        # embed
        embed = self.mlp(self.embed(scalar))
        return embed
