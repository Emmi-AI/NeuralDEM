import torch.nn.functional as F
from torch import nn

from .dot_product_attention import DotProductAttention
from .functional import modulate_scale_shift, modulate_gate
from .mlp import Mlp


class MultiDitBlock(nn.Module):
    """Multilpe independent DiT blocks where modalities/branches dont interact with each other."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_modalities: int = 4,
        cond_dim: int | None = None,
        eps=1e-6,
    ):
        super().__init__()
        self.dim = dim
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
        self.attn = nn.ModuleList(
            [
                DotProductAttention(
                    dim=dim,
                    num_heads=num_heads,
                )
                for _ in range(num_modalities)
            ],
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
        for i in range(len(x)):
            x[i] = self.attn[i](x[i])
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

    def forward(self, x, cond):
        assert isinstance(x, (list, tuple))
        assert isinstance(cond, (list, tuple))

        scales_shifts_gates = [self.modulation[i](cond[i]).chunk(6, dim=-1) for i in range(self.num_modalities)]
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
        x = [og_x[i] + x[i] for i in range(len(x))]
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
        x = [og_x[i] + x[i] for i in range(len(x))]
        return x
