from typing import Sequence

import einops
import torch
import torch.nn.functional as F
from torch import nn


class MultimodalDotProductAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_main_modalities: int,
        num_aux_modalities: int,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        num_modalities = num_main_modalities + num_aux_modalities
        self.num_heads = num_heads
        self.num_modalities = num_modalities
        self.num_main_modalities = num_main_modalities
        self.num_aux_modalities = num_aux_modalities

        # attention is done in main_branch dim
        self.qkv = nn.ModuleList(
            [
                nn.Linear(dim, dim * 3)
                for _ in range(num_modalities)
            ],
        )
        self.proj = nn.ModuleList(
            [
                nn.Linear(dim, dim)
                for _ in range(num_modalities)
            ],
        )

    def forward(self, *args: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        assert len(args) == self.num_modalities
        assert all(args[i].ndim == 3 for i in range(self.num_modalities))
        x = list(args)
        seqlens = [x[i].size(1) for i in range(self.num_modalities)]

        qs = []
        ks = []
        vs = []
        for i in range(self.num_modalities):
            q, k, v = einops.rearrange(
                self.qkv[i](x[i]),
                "bs seqlen (three num_heads head_dim) -> three bs num_heads seqlen head_dim",
                three=3,
                num_heads=self.num_heads,
            ).unbind(0)
            qs.append(q)
            ks.append(k)
            vs.append(v)
        # concat main in sequence dimension
        main_q = torch.concat(qs[:self.num_main_modalities], dim=2)
        main_k = torch.concat(ks[:self.num_main_modalities], dim=2)
        main_v = torch.concat(vs[:self.num_main_modalities], dim=2)
        # main attention
        main_x = F.scaled_dot_product_attention(main_q, main_k, main_v)
        main_x = einops.rearrange(main_x, "bs num_heads seqlen head_dim -> bs seqlen (num_heads head_dim)")
        main_x = list(main_x.split(seqlens[:self.num_main_modalities], dim=1))

        # optional attention
        optional_x = []
        for i in range(self.num_main_modalities, self.num_modalities):
            q = qs[i]
            k = torch.concat([main_k.detach(), ks[i]], dim=2)
            v = torch.concat([main_v.detach(), vs[i]], dim=2)
            opt_x = F.scaled_dot_product_attention(q, k, v)
            opt_x = einops.rearrange(opt_x, "bs num_heads seqlen head_dim -> bs seqlen (num_heads head_dim)")
            optional_x.append(opt_x)

        # postprocess
        x = main_x + optional_x
        for i in range(self.num_modalities):
            x[i] = self.proj[i](x[i])
        return x
