from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .functional import modulate_scale_shift, modulate_gate
from .mlp import Mlp
from .multimodal_dot_product_attention import MultimodalDotProductAttention


class MultimodalDitBlock(nn.Module):
    """Multilpe DiT blocks where main modalities/branches interact with each other via concatenating all tokens of main
    modalities/branches before the attention. Additionally, auxiliary modalities/branches can be used which use
    self-attention within its own modality and detached cross-attention to the main modalities."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_main_modalities: int = 2,
        num_aux_modalities: int = 1,
        cond_dim: int | None = None,
        eps=1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.num_main_modalities = num_main_modalities
        self.num_aux_modalities = num_aux_modalities
        num_modalities = num_main_modalities + num_aux_modalities
        self.num_modalities = num_modalities
        self.eps = eps
        # properties
        cond_dim = cond_dim or dim * 4
        # modulation
        self.modulation = nn.ModuleList(
            [
                nn.Linear(cond_dim, dim * 6)
                for _ in range(num_modalities)
            ],
        )
        # attn
        self.attn = MultimodalDotProductAttention(
            dim=dim,
            num_heads=num_heads,
            num_main_modalities=num_main_modalities,
            num_aux_modalities=num_aux_modalities,
        )
        # mlp
        self.mlp = nn.ModuleList(
            [
                Mlp(dim)
                for _ in range(num_modalities)
            ],
        )

    def _attn_residual_path(self, x, scales, shifts, gates):
        assert isinstance(x, list)
        assert isinstance(scales, list)
        assert isinstance(shifts, list)
        assert isinstance(gates, list)
        assert len(x) == len(scales)
        assert len(x) == len(shifts)
        assert len(x) == len(gates)
        for i in range(len(x)):
            x[i] = F.layer_norm(x[i], [self.dim], eps=self.eps)
            x[i] = modulate_scale_shift(x[i], scale=scales[i], shift=shifts[i])
        x = self.attn(*x)
        for i in range(len(x)):
            x[i] = modulate_gate(x[i], gate=gates[i])
        return x

    def _mlp_residual_path(self, x, scales, shifts, gates):
        assert isinstance(x, list)
        assert isinstance(scales, list)
        assert isinstance(shifts, list)
        assert isinstance(gates, list)
        assert len(x) == len(scales)
        assert len(x) == len(shifts)
        assert len(x) == len(gates)
        for i in range(len(x)):
            x[i] = F.layer_norm(x[i], [self.dim], eps=self.eps)
            x[i] = modulate_scale_shift(x[i], scale=scales[i], shift=shifts[i])
            x[i] = self.mlp[i](x[i])
            x[i] = modulate_gate(x[i], gate=gates[i])
        return x

    def forward(self, x: Sequence[torch.Tensor], cond: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
        assert isinstance(x, (list, tuple))
        assert isinstance(cond, (list, tuple))

        # dit modulation vectors
        scales_shifts_gates = [self.modulation[i](cond[i]).chunk(chunks=6, dim=-1) for i in range(self.num_modalities)]

        # attention
        scales = [scales_shifts_gates[i][0] for i in range(self.num_modalities)]
        shifts = [scales_shifts_gates[i][1] for i in range(self.num_modalities)]
        gates = [scales_shifts_gates[i][2] for i in range(self.num_modalities)]
        og_x = [x[i] for i in range(len(x))]
        x = self._attn_residual_path(
            x=x,
            scales=scales,
            shifts=shifts,
            gates=gates,
        )
        # attention residual
        x = [og_x[i] + x[i] for i in range(len(x))]

        # mlp
        scales = [scales_shifts_gates[i][3] for i in range(self.num_modalities)]
        shifts = [scales_shifts_gates[i][4] for i in range(self.num_modalities)]
        gates = [scales_shifts_gates[i][5] for i in range(self.num_modalities)]
        og_x = [x[i] for i in range(len(x))]
        x = self._mlp_residual_path(
            x=x,
            scales=scales,
            shifts=shifts,
            gates=gates,
        )
        # mlp residual
        x = [og_x[i] + x[i] for i in range(len(x))]

        return x
