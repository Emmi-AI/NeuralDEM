from functools import partial

import torch
from torch import nn

from .functional import modulate_scale_shift, modulate_gate
from .mlp import Mlp
from .perceiver_attention import PerceiverAttention


class DitPerceiverBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        kv_dim=None,
        cond_dim=None,
        eps=1e-6,
    ):
        super().__init__()
        norm_ctor = partial(nn.LayerNorm, elementwise_affine=False)
        cond_dim = cond_dim or dim * 4
        # modulation
        self.modulation = nn.Linear(cond_dim, dim * 8)
        # attention
        self.norm1q = norm_ctor(dim, eps=eps)
        self.norm1kv = norm_ctor(kv_dim or dim, eps=eps)
        self.attn = PerceiverAttention(dim=dim, num_heads=num_heads)
        # mlp
        self.norm2 = norm_ctor(dim, eps=eps)
        self.mlp = Mlp(dim)

    def _attn_residual_path(self, q, kv, q_scale, q_shift, kv_scale, kv_shift, gate):
        q = modulate_scale_shift(self.norm1q(q), scale=q_scale, shift=q_shift)
        kv = modulate_scale_shift(self.norm1kv(kv), scale=kv_scale, shift=kv_shift)
        x = self.attn(q=q, kv=kv)
        return modulate_gate(x, gate=gate)

    def _mlp_residual_path(self, x, scale, shift, gate):
        return modulate_gate(self.mlp(modulate_scale_shift(self.norm2(x), scale=scale, shift=shift)), gate=gate)

    def forward(self, q: torch.Tensor, kv: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        modulation = self.modulation(cond).chunk(chunks=8, dim=-1)
        q_scale, q_shift, kv_scale, kv_shift, attn_gate, mlp_scale, mlp_shift, mlp_gate = modulation
        q = q + self._attn_residual_path(
            q=q,
            kv=kv,
            q_scale=q_scale,
            q_shift=q_shift,
            kv_scale=kv_scale,
            kv_shift=kv_shift,
            gate=attn_gate,
        )
        q = q + self._mlp_residual_path(
            x=q,
            scale=mlp_scale,
            shift=mlp_shift,
            gate=mlp_gate,
        )
        return q
