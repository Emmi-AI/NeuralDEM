import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from neural_dem.modules import (
    ScalarConditioner,
    MultimodalDitBlock,
    DitPerceiverDecoder,
    ContinuousSincosEmbed,
    SupernodePooling,
)
from neural_dem.utils import patch_utils, sampling_utils


class FbedModel(nn.Module):
    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 12,
        radius: int | float = 66,
        patch_size: tuple[int, int, int] = (4, 4, 4),
        resolution: tuple[int, int, int] = (40, 40, 100),
        depth: int = 12,
        normalized_domain_min: tuple[float, float, float] = (0., 0., 0.),
        normalized_domain_max: tuple[float, float, float] = (1000., 1000., 2500.),
        normalized_particle_radius: float = 6.25,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.radius = radius
        self.patch_size = patch_size
        self.resolution = resolution

        # embed condition (inlet velocity)
        self.conditioner = ScalarConditioner(dim)

        # positional embedding (used in all branches)
        self.pos_embed = ContinuousSincosEmbed(dim=dim, ndim=3)

        # embed particles (displacement)
        self.supernode_pooling = SupernodePooling(radius=radius, input_dim=3, ndim=3, hidden_dim=dim)
        self.to_particle_token = nn.Linear(dim, dim)

        # embed fluid (velocity + void fraction)
        self.to_fluid_token = nn.Linear(int(np.prod(patch_size)) * 4, dim)

        # embed mixing (concentration)
        self.to_mixing_token = nn.Linear(int(np.prod(patch_size)), dim)

        # transformer blocks
        self.blocks = nn.ModuleList(
            [
                MultimodalDitBlock(
                    dim=dim,
                    num_heads=num_heads,
                    # 2 main modalities: particles (displacement) and fluid cells (velocity + void fraction)
                    num_main_modalities=2,
                    # 1 auxiliary modality: mixing
                    num_aux_modalities=1,
                    cond_dim=self.conditioner.cond_dim,
                )
                for _ in range(depth)
            ],
        )

        # decoders
        self.particle_decoder = DitPerceiverDecoder(
            dim=dim,
            num_heads=num_heads,
            ndim=3,
            output_dim=3,
            cond_dim=self.conditioner.cond_dim,
        )
        self.fluid_decoder = DitPerceiverDecoder(
            dim=dim,
            num_heads=num_heads,
            ndim=3,
            output_dim=int(np.prod(patch_size)) * 4,
            cond_dim=self.conditioner.cond_dim,
        )
        self.mixing_decoder = DitPerceiverDecoder(
            dim=dim,
            num_heads=num_heads,
            ndim=3,
            output_dim=int(np.prod(patch_size)),
            cond_dim=self.conditioner.cond_dim,
        )

        # bounding boxes for particle position sampling
        self.register_buffer(
            "normalized_domain_boundary",
            torch.stack(
                [
                    torch.Tensor(normalized_domain_min),
                    torch.Tensor(normalized_domain_max),
                ],
            ),
            persistent=False,
        )
        self.register_buffer(
            "padded_normalized_domain_boundary",
            torch.stack(
                [
                    torch.Tensor(normalized_domain_min) + normalized_particle_radius,
                    torch.Tensor(normalized_domain_max) - normalized_particle_radius,
                ],
            ),
            persistent=False,
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
        # conditioning
        inlet_velocity: torch.Tensor,
        # particles
        input_particle_coords: torch.Tensor,
        input_particle_displacement: torch.Tensor,
        supernode_idx: torch.Tensor,
        # fluid
        input_fluid_coords: torch.Tensor,
        input_fluid_velocity: torch.Tensor,
        input_fluid_voidfraction: torch.Tensor,
        # mixing
        input_mixing_coords: torch.Tensor,
        input_mixing_concentration: torch.Tensor,
        # output coords
        output_fluid_coords: torch.Tensor,
        output_mixing_coords: torch.Tensor,
        # stuff for sampling particle positions from voidfraction
        fluid_voidfraction_min: torch.Tensor,
        num_output_particles: torch.Tensor,
    ):
        condition = self.conditioner(inlet_velocity)

        # particle encoding
        particle_tokens = self.supernode_pooling(
            input_feat=input_particle_displacement,
            input_pos=input_particle_coords,
            supernode_idx=supernode_idx,
        )
        particle_tokens = self.to_particle_token(particle_tokens)

        # fluid encoding
        fluid_tokens = self.to_fluid_token(torch.concat([input_fluid_velocity, input_fluid_voidfraction], dim=-1))
        fluid_tokens = fluid_tokens + self.pos_embed(input_fluid_coords)

        # mixing encoding
        mixing_tokens = self.to_mixing_token(input_mixing_concentration) + self.pos_embed(input_mixing_coords)

        # transformer
        x = [particle_tokens, fluid_tokens, mixing_tokens]
        for i, block in enumerate(self.blocks):
            # split transformer blocks logically into encoder/approximator/decoder with non-affine
            # layernorms before/after approximator to stabilize potential latent rollout
            if i in [4, 8]:
                for j in range(len(x)):
                    x[j] = F.layer_norm(x[j], [self.dim], eps=1e-6)
            x = block(x=x, cond=(condition, condition, condition))
        particle_tokens, fluid_tokens, mixing_tokens = x

        # fluid decoding
        fluid_output = self.fluid_decoder(
            x=fluid_tokens,
            pos=output_fluid_coords,
            block_kwargs=dict(cond=condition),
        )

        # mixing decoding
        mixing_output = self.mixing_decoder(
            x=mixing_tokens,
            pos=output_mixing_coords,
            block_kwargs=dict(cond=condition),
        )

        # sample particle positions from predicted voidfraction field
        voidfraction = fluid_output[:, int(np.prod(self.patch_size)) * 3:]
        # ground truth voidfraction is normalized to [0, 1] -> clamp to range to avoid out-of-bounds predictions
        solidfraction = (1 - voidfraction).clamp(min=0, max=1)
        # denormalize
        max_solidfraction = 1 - fluid_voidfraction_min
        solidfraction = solidfraction / max_solidfraction
        # convert patch tokens to a coherent 3d volume
        solidfraction = patch_utils.depatchify(
            volume=solidfraction,
            patch_size=self.patch_size,
            resolution=self.resolution,
        )
        # sample output particle position
        output_particle_coords = sampling_utils.sample_particles_from_solidfraction(
            solidfraction=solidfraction,
            # use same number of outputs as inputs for simplicity
            num_particles=num_output_particles,
            bbox=self.normalized_domain_boundary,
            sampling_bbox=self.padded_normalized_domain_boundary,
            volume_shape=self.resolution,
        ).unsqueeze(0)

        # particle decoding
        particle_displacement = self.particle_decoder(
            x=particle_tokens,
            pos=output_particle_coords,
            block_kwargs=dict(cond=condition),
        )

        return dict(
            particle_displacement=particle_displacement,
            particle_coords=output_particle_coords,
            fluid=fluid_output,
            mixing=mixing_output,
        )
