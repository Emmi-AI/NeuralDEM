import einops
import torch
from torch import nn


class ContinuousSincosEmbed(nn.Module):
    def __init__(self, dim: int, ndim: int, max_wavelength: int = 10000, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.dim = dim
        self.ndim = ndim
        # if dim is not cleanly divisible -> cut away trailing dimensions
        self.ndim_padding = dim % ndim
        dim_per_ndim = (dim - self.ndim_padding) // ndim
        self.sincos_padding = dim_per_ndim % 2
        self.max_wavelength = max_wavelength
        self.padding = self.ndim_padding + self.sincos_padding * ndim
        effective_dim_per_wave = (self.dim - self.padding) // ndim
        assert effective_dim_per_wave > 0
        self.register_buffer(
            "omega",
            1. / max_wavelength ** (torch.arange(0, effective_dim_per_wave, 2, dtype=dtype) / effective_dim_per_wave),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        out_dtype = coords.dtype
        ndim = coords.shape[-1]
        assert self.ndim == ndim
        out = coords.unsqueeze(-1).to(self.omega.dtype) @ self.omega.unsqueeze(0)
        emb = torch.concat([torch.sin(out), torch.cos(out)], dim=-1)
        if coords.ndim == 3:
            emb = einops.rearrange(emb, "bs num_points ndim dim -> bs num_points (ndim dim)")
        elif coords.ndim == 2:
            emb = einops.rearrange(emb, "num_points ndim dim -> num_points (ndim dim)")
        else:
            raise NotImplementedError
        emb = emb.to(out_dtype)
        if self.padding > 0:
            padding = torch.zeros(*emb.shape[:-1], self.padding, device=emb.device, dtype=emb.dtype)
            emb = torch.concat([emb, padding], dim=-1)
        return emb
