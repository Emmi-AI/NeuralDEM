import einops
import torch


def patchify(volume: torch.Tensor, patch_size: tuple[int, int, int]) -> torch.Tensor:
    return einops.rearrange(
        volume,
        "(h ph) (w pw) (d pd) c -> (h w d) (ph pw pd) c",
        ph=patch_size[0],
        pw=patch_size[1],
        pd=patch_size[2],
    )


def depatchify(
    volume: torch.Tensor,
    patch_size: tuple[int, int, int],
    resolution: tuple[int, int, int],
) -> torch.Tensor:
    seqlen_h = resolution[0] // patch_size[0]
    seqlen_w = resolution[1] // patch_size[1]
    seqlen_d = resolution[2] // patch_size[2]
    return einops.rearrange(
        volume,
        "(h w d) (ph pw pd c) -> (h ph) (w pw) (d pd) c",
        ph=patch_size[0],
        pw=patch_size[1],
        pd=patch_size[2],
        h=seqlen_h,
        w=seqlen_w,
        d=seqlen_d,
    )


def get_patch_center(position: torch.Tensor, patch_size: tuple[int, int, int]) -> torch.Tensor:
    ndim = len(patch_size)
    assert position.ndim == ndim + 1, f"positions.ndim={position.ndim} != {ndim + 1}"
    # check that each dim is divisible by the patch size
    for i in range(ndim):
        assert (
            position.shape[i] % patch_size[i] == 0
        ), f"positions.shape[{i}]={position.shape[i]} not divisible by patch_size[{i}]={patch_size[i]}"
    patchs_per_dim = [position.shape[i] // patch_size[i] for i in range(ndim)]

    # create a mesh grid of the patch centers
    processed_grid_linspace = [torch.zeros(x) for x in patchs_per_dim]
    for i in range(ndim):
        for j in range(patchs_per_dim[i]):
            # average all the coordinates in the patch
            start = j * patch_size[i]
            end = (j + 1) * patch_size[i]
            if i == 0:
                processed_grid_linspace[i][j] = position[start:end, ..., i].mean()
            elif i == 1:
                processed_grid_linspace[i][j] = position[:, start:end, ..., i].mean()
            else:
                processed_grid_linspace[i][j] = position[..., start:end, i].mean()

    patch_centers = torch.meshgrid(*processed_grid_linspace, indexing="ij")
    patch_centers = torch.stack(patch_centers, dim=-1)

    return patch_centers.reshape(-1, ndim)
