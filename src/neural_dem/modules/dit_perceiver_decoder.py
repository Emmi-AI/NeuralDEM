from typing import Any

import einops
import torch
from torch import nn

from .continuous_sincos_embed import ContinuousSincosEmbed
from .dit_perceiver_block import DitPerceiverBlock


class DitPerceiverDecoder(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ndim: int,
        output_dim: int,
        cond_dim: int,
        eps=1e-6,
    ):
        super().__init__()
        # create query
        self.pos_embed = ContinuousSincosEmbed(dim=dim, ndim=ndim)
        self.query = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        # perceiver
        self.perc = DitPerceiverBlock(
            dim=dim,
            num_heads=num_heads,
            cond_dim=cond_dim,
            eps=eps,
        )
        # to final output
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.pred = nn.Linear(dim, output_dim)

    def forward(self, x: torch.Tensor, pos: torch.Tensor, block_kwargs: dict[str, Any] | None = None):
        if pos is None:
            return None
        assert x.ndim == 3
        assert pos.ndim == 3

        # create query
        query = self.query(self.pos_embed(pos))

        # perceiver
        x = self.perc(q=query, kv=x, **(block_kwargs or {}))

        # predict value
        x = self.norm(x)
        x = self.pred(x)

        # dense tensor (batch_size, max_num_points, dim) -> sparse tensor (batch_size * num_points, dim)
        x = einops.rearrange(x, "batch_size max_num_points dim -> (batch_size max_num_points) dim")
        return x
