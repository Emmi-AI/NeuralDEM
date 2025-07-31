import numpy as np
import scipy


def gaussian_kernel_3d(size: int, sigma: float):
    kernel = np.fromfunction(
        lambda x, y, z: (1 / (2 * np.pi * sigma ** 2)) * np.exp(
            -((x - size // 2) ** 2 + (y - size // 2) ** 2 + (z - size // 2) ** 2) / (2 * sigma ** 2)
        ),
        (size, size, size)
    )
    return kernel / np.sum(kernel)


def smooth_field(field: np.array, sigma: float, kernel_size: int = 9):
    if sigma == 0:
        return field
    else:
        kernel = gaussian_kernel_3d(kernel_size, sigma)
        padded = np.pad(field, kernel_size // 2, mode="reflect")
        field_smoothed = scipy.signal.convolve(padded, kernel, mode="valid")
        return field_smoothed


def coords_to_voxel_idx_numpy(coords, bbox, volume_shape):
    voxel_idx = (
        (coords - bbox[0])
        / (bbox[1] - bbox[0])
        * np.array(volume_shape)
    )
    voxel_idx = voxel_idx.astype(np.long)
    # convert to flatten idx
    voxel_idx = (
        voxel_idx[:, 0] * volume_shape[1] * volume_shape[2]
        + voxel_idx[:, 1] * volume_shape[2]
        + voxel_idx[:, 2]
    )
    return voxel_idx


def inner_loop_numpy(voxel_idxs, selected_particles, num_voxels):
    particle_concentration = np.zeros(num_voxels)
    unique, counts = np.unique(voxel_idxs, return_counts=True)
    counts_dict = dict(zip(unique, counts))
    unique_selected, counts_selected = np.unique(voxel_idxs[selected_particles], return_counts=True)
    counts_dict_selected = dict(zip(unique_selected, counts_selected))

    for i in range(num_voxels):
        particles_per_voxel = counts_dict.get(i, 0)
        if particles_per_voxel > 0:
            particle_concentration[i] = counts_dict_selected.get(i, 0) / particles_per_voxel
        else:
            particle_concentration[i] = 0

    return particle_concentration


def compute_particle_concentration_numpy(
    particle_coords: np.array,
    particle_class: np.array,
    bbox: np.array,
    volume_shape: tuple[int, int, int],
):
    voxel_idx = coords_to_voxel_idx_numpy(particle_coords, bbox, volume_shape)
    selected_particles = particle_class.squeeze() > 0.5

    particle_concentration = inner_loop_numpy(voxel_idx, selected_particles, np.prod(volume_shape))

    return particle_concentration.reshape(volume_shape + (1,))
