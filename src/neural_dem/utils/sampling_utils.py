import torch


def sample_particles_from_solidfraction(
    solidfraction,
    num_particles: int,
    bbox: torch.Tensor,
    sampling_bbox: torch.Tensor,
    volume_shape: tuple[int, int, int] = (40, 40, 100),
):
    cur_num_points = 0
    i = 1
    sampled_points = []
    solidfraction = solidfraction.flatten()
    while cur_num_points < num_particles and i < 100:
        points = _sample_random_points(
            num_points=2 * num_particles // i,
            bbox=sampling_bbox,
            device=solidfraction.device,
        )
        # perform rejection sampling to sample points inside the fluid based on the solid fraction
        voxel_idxs = _coords_to_voxel_idx(points, bbox, volume_shape)
        sampling_prob = torch.rand((len(voxel_idxs),), device=voxel_idxs.device)
        selected_points = points[solidfraction[voxel_idxs] > sampling_prob]
        sampled_points.append(selected_points)
        cur_num_points += len(selected_points)
        i += 1

    return torch.concat(sampled_points)[-num_particles:]


def _sample_random_points(num_points: int, bbox: torch.Tensor, device: torch.device) -> torch.Tensor:
    points = torch.rand(size=(num_points, 3), device=device)
    points = points * (bbox[1] - bbox[0]) + bbox[0]
    return points


def _coords_to_voxel_idx(
    coords: torch.Tensor,
    bbox: torch.Tensor,
    volume_shape: tuple[int, int, int],
) -> torch.Tensor:
    voxel_idx = (
        (coords - bbox[0])
        / (bbox[1] - bbox[0])
        * torch.tensor(volume_shape, device=coords.device)
    )
    voxel_idx = voxel_idx.to(torch.long)
    # convert to flatten idx
    voxel_idx = (
        voxel_idx[:, 0] * volume_shape[1] * volume_shape[2]
        + voxel_idx[:, 1] * volume_shape[2]
        + voxel_idx[:, 2]
    )
    return voxel_idx
