import torch
from torch import nn

from .continuous_sincos_embed import ContinuousSincosEmbed


class ScalarsConditioner(nn.Module):
    def __init__(
        self,
        dim: int,
        num_scalars: int,
        cond_dim: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        cond_dim = cond_dim or dim * 4
        self.dim = dim
        self.num_scalars = num_scalars
        self.cond_dim = cond_dim
        # sin-cos embedding of individual scalars
        self.embed = ContinuousSincosEmbed(dim=dim, ndim=1)
        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, dim),
                    nn.GELU(),
                )
                for _ in range(num_scalars)
            ],
        )
        # combine conditions
        self.shared_mlp = nn.Sequential(
            nn.Linear(dim * num_scalars, dim),
            nn.GELU(),
            nn.Linear(dim, self.cond_dim),
            nn.GELU(),
        )

    def forward(self, *args, **kwargs):
        # checks + preprocess
        scalars = [*args] + list(kwargs.values())
        assert len(scalars) == self.num_scalars, f"got {len(scalars)} scalars but num_scalars == {self.num_scalars}"
        expected_len = None
        for i in range(self.num_scalars):
            scalar = scalars[i]
            if isinstance(scalar, float | int):
                scalar = torch.tensor([scalar], device=next(self.parameters()).device, dtype=torch.float)
            if expected_len is None:
                expected_len = scalar.numel()
            assert scalar.numel() == len(scalar) and scalar.ndim <= 2, \
                f"scalar should be (batch_size,) or (batch_size, 1), got {scalar.shape}"
            assert len(scalar) == expected_len, f"got scalars of different size ({len(scalars)} != {expected_len})"
            if scalar.ndim == 1:
                scalars[i] = scalar.unsqueeze(1)

        # embed all scalars at once
        embeds = self.embed(torch.concat(scalars)).chunk(self.num_scalars)
        # project embeds
        projs = [self.mlps[i](embeds[i]) for i in range(self.num_scalars)]
        # combine embeds
        embed = self.shared_mlp(torch.concat(projs, dim=1))

        return embed
