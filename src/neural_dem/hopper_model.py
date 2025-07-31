import einops
import torch
from torch import nn

from neural_dem.modules import ScalarsConditioner, MultiDitBlock, DitPerceiverDecoder


class HopperModel(nn.Module):
    def __init__(self, dim: int = 384, num_heads: int = 6, num_latent_tokens: int = 32, depth: int = 1):
        super().__init__()
        # embed scalar conditions (hopper angle, material friction, ...)
        self.conditioner = ScalarsConditioner(dim=dim, num_scalars=4)

        # the hopper model is not trained autoregressively -> start from learnable initial tokens
        self.initial_latent = nn.ParameterList(
            [
                nn.Parameter(torch.randn(1, num_latent_tokens, dim))
                for _ in range(4)
            ],
        )
        # as the hopper is a steady system, interactions between modalities are
        # optional as the whole system can be described via scalars alone
        self.blocks = nn.ModuleList(
            [
                MultiDitBlock(
                    dim=dim,
                    num_heads=num_heads,
                    # 4 modalities (displacement, occupancy, transport, residence time)
                    num_modalities=4,
                    cond_dim=self.conditioner.cond_dim,
                )
                for _ in range(depth)
            ],
        )
        # modality-specific decoders
        self.displacement_decoder = DitPerceiverDecoder(
            dim=dim,
            num_heads=num_heads,
            cond_dim=self.conditioner.cond_dim,
            output_dim=3,
            ndim=3,
        )
        self.occupancy_decoder = DitPerceiverDecoder(
            dim=dim,
            num_heads=num_heads,
            cond_dim=self.conditioner.cond_dim,
            output_dim=1,
            ndim=3,
        )
        self.transport_decoder = DitPerceiverDecoder(
            dim=dim,
            num_heads=num_heads,
            output_dim=3,
            cond_dim=self.conditioner.cond_dim,
            ndim=3,
        )
        self.residence_decoder = DitPerceiverDecoder(
            dim=dim,
            num_heads=num_heads,
            output_dim=1,
            cond_dim=self.conditioner.cond_dim,
            ndim=3,
        )
        self.reset_weights()

    def reset_weights(self) -> None:
        for name, p in self.named_parameters():
            if isinstance(p, nn.Linear):
                if "modulation" in name:
                    # DiT modulation layers are initialized to 0
                    nn.init.zeros_(p.weight)
                else:
                    nn.init.trunc_normal_(p.weight, mean=0, std=0.02)
                nn.init.zeros_(p.bias)

    def forward(
        self,
        # inputs
        timestep: int | torch.Tensor,
        geometry_angle: float | torch.Tensor,
        mupp: float | torch.Tensor,
        rfpp: float | torch.Tensor,
        # output positions
        output_displacement_pos: torch.Tensor,
        output_occupancy_pos: torch.Tensor,
        output_transport_pos: torch.Tensor,
        output_residence_pos: torch.Tensor,
    ):
        condition = self.conditioner(
            timestep=timestep,
            geometry_angle=geometry_angle,
            mupp=mupp,
            rfpp=rfpp,
        )
        batch_size = len(output_occupancy_pos)
        x = [
            einops.repeat(
                self.initial_latent[i],
                "1 num_latent_tokens dim -> batch_size num_latent_tokens dim",
                batch_size=batch_size,
            )
            for i in range(len(self.initial_latent))
        ]

        # apply blocks
        for blk in self.blocks:
            x = blk(x, cond=[condition] * 4)

        # decode
        displacement = self.displacement_decoder(
            x=x[0],
            pos=output_displacement_pos,
            block_kwargs=dict(cond=condition),
        )
        occupancy = self.occupancy_decoder(
            x=x[1],
            pos=output_occupancy_pos,
            block_kwargs=dict(cond=condition),
        )
        transport = self.transport_decoder(
            x=x[2],
            pos=output_transport_pos,
            block_kwargs=dict(cond=condition),
        )
        residence = self.residence_decoder(
            x=x[3],
            pos=output_residence_pos,
            block_kwargs=dict(cond=condition),
        )
        return dict(
            displacement=displacement,
            occupancy=occupancy,
            transport=transport,
            residence=residence,
        )
