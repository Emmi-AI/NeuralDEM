import fnmatch
import os
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import gaussian_blur


def plot_hopper_timesteps(
    # displacement
    displacement_pos: list[torch.Tensor],
    displacement_value: list[torch.Tensor],
    # occupancy
    occupancy_pos: list[torch.Tensor],
    occupancy_value: list[torch.Tensor],
    # transport
    transport_pos: list[torch.Tensor],
    transport_value: list[torch.Tensor],
    # residence
    residence_pos: list[torch.Tensor],
    residence_value: list[torch.Tensor],
    # which timesteps to plot
    timesteps: list[int],
    # how many transport bins
    num_transport_bins: int,
) -> None:
    _, axs = plt.subplots(nrows=len(timesteps), ncols=4, figsize=(20, len(timesteps) * 5), constrained_layout=True)
    axs = np.atleast_2d(axs)

    # coordinate min/max values
    xmin = min(
        displacement_pos[0][:, 0].min(),
        transport_pos[0][:, 0].min(),
        residence_pos[0][:, 0].min(),
    )
    xmax = max(
        displacement_pos[0][:, 0].max(),
        transport_pos[0][:, 0].max(),
        residence_pos[0][:, 0].max(),
    )
    zmin = min(
        displacement_pos[0][:, 2].min(),
        transport_pos[0][:, 2].min(),
        residence_pos[0][:, 2].min(),
    )
    zmax = max(
        displacement_pos[0][:, 2].max(),
        transport_pos[0][:, 2].max(),
        residence_pos[0][:, 2].max(),
    )
    # displacement norm min/max
    displacement_vmin = min([displacement_value[timestep].norm(dim=1).min() for timestep in timesteps])
    displacement_vmax = max([displacement_value[timestep].norm(dim=1).max() for timestep in timesteps])
    # z-transport min/max
    ztransport_vmin = min([transport_value[timestep][:, 2].min() for timestep in timesteps])
    ztransport_vmax = max([transport_value[timestep][:, 2].max() for timestep in timesteps])
    # residence min/max
    residence_vmin = min([residence_value[timestep].min() for timestep in timesteps])
    residence_vmax = max([residence_value[timestep].max() for timestep in timesteps])

    for i, timestep in enumerate(timesteps):
        # displacement
        x, y, z = displacement_pos[timestep].unbind(-1)
        is_center_slice = y.abs() < 0.01
        scatter = axs[i, 0].scatter(
            x[is_center_slice],
            z[is_center_slice],
            c=displacement_value[timestep][is_center_slice].norm(dim=1),
            s=3,
            vmin=displacement_vmin,
            vmax=displacement_vmax,
        )
        if i == 0:
            axs[i, 0].set_title("norm of displacement")
        axs[i, 0].set_xlim(xmin, xmax)
        axs[i, 0].set_ylim(zmin, zmax)
        axs[i, 0].set_ylabel(f"timestep {timestep}")
        plt.colorbar(scatter, ax=axs[i, 0])

        # transport
        x, y, z = transport_pos[timestep].unbind(-1)
        is_center_slice = y.abs() < 0.01
        cur_target = transport_value[timestep][is_center_slice, 2]
        # bin transport
        if num_transport_bins > 0:
            cur_target = (cur_target - ztransport_vmin) / ztransport_vmax * num_transport_bins
            cur_target = cur_target.long().float().clamp(max=num_transport_bins - 1)
            cur_target = cur_target / (num_transport_bins - 1)
            cur_target = cur_target * ztransport_vmax + ztransport_vmin
        scatter = axs[i, 1].scatter(
            x[is_center_slice],
            z[is_center_slice],
            c=cur_target,
            s=3,
            vmin=ztransport_vmin,
            vmax=ztransport_vmax,
        )
        plt.colorbar(scatter, ax=axs[i, 1])
        if i == 0:
            axs[i, 1].set_title(f"{num_transport_bins}-bin z component of transport")
        axs[i, 1].set_xlim(xmin, xmax)
        axs[i, 1].set_ylim(zmin, zmax)

        # residence
        x, y, z = residence_pos[timestep].unbind(-1)
        is_center_slice = y.abs() < 0.01
        scatter = axs[i, 2].scatter(
            x[is_center_slice],
            z[is_center_slice],
            c=residence_value[timestep][is_center_slice],
            s=3,
            vmin=residence_vmin,
            vmax=residence_vmax,
        )
        plt.colorbar(scatter, ax=axs[i, 2])
        if i == 0:
            axs[i, 2].set_title("residence time")
        axs[i, 2].set_xlim(xmin, xmax)
        axs[i, 2].set_ylim(zmin, zmax)

        # occupancy
        x, y, z = occupancy_pos[timestep].unbind(-1)
        is_center_slice = y.abs() < 0.01
        scatter = axs[i, 3].scatter(
            x[is_center_slice],
            z[is_center_slice],
            c=occupancy_value[timestep][is_center_slice],
            s=3,
            vmin=0,
            vmax=1,
        )
        plt.colorbar(scatter, ax=axs[i, 3])
        if i == 0:
            axs[i, 3].set_title("occupancy")
        axs[i, 3].set_xlim(xmin, xmax)
        axs[i, 3].set_ylim(zmin, zmax)

    # disable ticks
    for row in axs:
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
    plt.show()


def plot_hopper_predictions_randompos(
    target_pos: list[torch.Tensor],
    prediction_pos: list[torch.Tensor],
    target: list[torch.Tensor],
    prediction: list[torch.Tensor],
    name: str,
    num_bins: int = 0,
) -> None:
    _, axs = plt.subplots(nrows=len(target_pos), ncols=2, figsize=(10, len(target_pos) * 5), constrained_layout=True)
    axs = np.atleast_2d(axs)

    # coordinate min/max values
    xmin = prediction_pos[0][:, 0].min().item()
    xmax = prediction_pos[0][:, 0].max().item()
    zmin = prediction_pos[0][:, 2].min().item()
    zmax = prediction_pos[0][:, 2].max().item()
    # min/max
    vmin = min([target[i].min().item() for i in range(len(target_pos))])
    vmax = max([target[i].max().item() for i in range(len(target_pos))])

    for i in range(len(target_pos)):
        # target
        x, y, z = target_pos[i].unbind(-1)
        is_center_slice = y.abs() < 0.01
        cur_target = target[i][is_center_slice]
        if num_bins > 0:
            cur_target = (cur_target - vmin) / vmax * num_bins
            cur_target = cur_target.long().float().clamp(max=num_bins - 1)
            cur_target = cur_target / (num_bins - 1)
            cur_target = cur_target * vmax + vmin
        scatter = axs[i, 0].scatter(
            x[is_center_slice],
            z[is_center_slice],
            c=cur_target,
            s=3,
            vmin=vmin,
            vmax=vmax,
        )
        axs[i, 0].set_xlim(xmin, xmax)
        axs[i, 0].set_ylim(zmin, zmax)
        plt.colorbar(scatter, ax=axs[i, 0])

        # prediction
        x, y, z = prediction_pos[i].unbind(-1)
        is_center_slice = y.abs() < 0.01
        cur_prediction = prediction[i][is_center_slice]
        if num_bins > 0:
            cur_prediction = (cur_prediction - vmin) / vmax * num_bins
            cur_prediction = cur_prediction.long().float().clamp(max=num_bins - 1)
            cur_prediction = cur_prediction / (num_bins - 1)
            cur_prediction = cur_prediction * vmax + vmin
        scatter = axs[i, 1].scatter(
            x[is_center_slice],
            z[is_center_slice],
            c=cur_prediction,
            s=3,
            vmin=vmin,
            vmax=vmax,
        )
        axs[i, 1].set_xlim(xmin, xmax)
        axs[i, 1].set_ylim(zmin, zmax)
        if i == 0:
            axs[i, 0].set_title(f"target {name}")
            axs[i, 1].set_title(f"predicted {name}")
        plt.colorbar(scatter, ax=axs[i, 1])

    # disable ticks
    for row in axs:
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
    plt.show()


def plot_fbed_timesteps(
    particle_position: torch.Tensor,
    particle_displacement: torch.Tensor,
    fluid_position: torch.Tensor,
    fluid_velocity: list[torch.Tensor],
    fluid_voidfraction: list[torch.Tensor],
    mixing: list[torch.Tensor],
    timesteps: list[int] | None = None,
    mixing_transparency_threshold: float = 0.65,
    cmap: str = "magma",
    ylabel_mode: Literal["timestep", "type"] = "timestep",
    output_uri: str | None = None,
) -> None:
    num_timesteps = len(fluid_velocity)
    _, axs = plt.subplots(nrows=num_timesteps, ncols=4, figsize=(13, num_timesteps * 5), constrained_layout=True)
    axs = np.atleast_2d(axs)

    xmin = fluid_position[:, :, :, 0].min()
    xmax = fluid_position[:, :, :, 0].max()
    zmin = fluid_position[:, :, :, 2].min()
    zmax = fluid_position[:, :, :, 2].max()

    y_center_from = fluid_position[0, fluid_position.size(1) // 2, 0, 1].item()
    y_center_to = fluid_position[0, fluid_position.size(1) // 2 + 1, 0, 1].item()

    for i in range(num_timesteps):
        # particle positions (plot center slice)
        x, y, z = particle_position[i].unbind(-1)
        is_center_slice = torch.logical_and(y_center_from < y, y < y_center_to)
        scatter = axs[i, 0].scatter(
            x[is_center_slice],
            z[is_center_slice],
            c=particle_displacement[i][is_center_slice].norm(dim=-1),
            s=0.5,
            vmin=0,
            vmax=0.008,
            cmap=cmap,
        )
        if i == 0:
            axs[i, 0].set_title("Norm of particle\ndisplacement", fontsize=20)
        axs[i, 0].set_xlim(xmin, xmax)
        axs[i, 0].set_ylim(zmin, zmax)
        if ylabel_mode == "timestep":
            axs[i, 0].set_ylabel(f"Timestep {timesteps[i]}", fontsize=20)
        elif ylabel_mode == "type":
            if i == 0:
                axs[i, 0].set_ylabel("Ground truth DEM", fontsize=20)
            elif i == 1:
                axs[i, 0].set_ylabel("NeuralDEM prediction", fontsize=20)
            else:
                raise NotImplementedError
        axs[i, 0].set_aspect("equal")
        plt.colorbar(scatter, ax=axs[i, 0])

        # fluid velocity (plot center slice)
        center_fluid_velocity = fluid_velocity[i][:, fluid_velocity[0].size(1) // 2].norm(dim=-1)
        imshow = axs[i, 1].imshow(
            center_fluid_velocity.T,
            origin="lower",
            vmin=0,
            vmax=7,
            cmap=cmap,
        )
        if i == 0:
            axs[i, 1].set_title("Norm of fluid\nvelocity", fontsize=20)
        plt.colorbar(imshow, ax=axs[i, 1])

        # fluid voidfraction (plot center slice)
        center_fluid_voidfraction = fluid_voidfraction[i][:, fluid_voidfraction[0].size(1) // 2]
        imshow = axs[i, 2].imshow(
            center_fluid_voidfraction.T,
            origin="lower",
            vmin=0,
            vmax=1,
            cmap=cmap,
        )
        if i == 0:
            axs[i, 2].set_title("Fluid\nvoidfraction", fontsize=20)
        plt.colorbar(imshow, ax=axs[i, 2])

        # mixing (plot center slice)
        center_mixing = mixing[i][:, mixing[0].size(1) // 2]
        # format nicely such that empty areas have a different color and border is smooth
        center_mixing[center_fluid_voidfraction > mixing_transparency_threshold] += 2
        center_mixing = gaussian_blur(center_mixing.unsqueeze(0), kernel_size=9, sigma=2).squeeze(0)
        imshow = axs[i, 3].imshow(
            center_mixing.T,
            vmin=0,
            vmax=2,
            origin="lower",
            cmap=cmap,
        )
        if i == 0:
            axs[i, 3].set_title("Particle\nmixing", fontsize=20)
        plt.colorbar(imshow, ax=axs[i, 3])

    # disable ticks
    for row in axs:
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
    # show/save
    if output_uri is None:
        plt.show()
    else:
        plt.savefig(output_uri)
    plt.close()


def merge_pngs_to_gif(
    png_folder_uri: str,
    png_pattern: str,
    output_uri: str,
) -> None:
    fnames = [fname for fname in os.listdir(png_folder_uri) if fnmatch.fnmatch(fname, png_pattern)]
    fnames = list(sorted(fnames))
    imgs = [Image.open(Path(png_folder_uri) / fname) for fname in fnames]
    imgs[0].save(
        fp=output_uri,
        format="GIF",
        append_images=imgs[1:],
        save_all=True,
        duration=100,
        loop=0,
    )


def plot_fbed_statistics(
    target: list[torch.Tensor],
    prediction: list[torch.Tensor],
    mode: Literal["mean", "std"],
    name: str,
    vmin: float,
    vmax: float,
    cmap: str = "magma",
) -> None:
    fig, axs = plt.subplots(ncols=2, figsize=(5, 5), constrained_layout=True)

    stacked_target = torch.stack(target)
    stacked_prediction = torch.stack(prediction)
    if stacked_target.size(-1) == 3:
        stacked_target = stacked_target.norm(dim=-1)
        stacked_prediction = stacked_prediction.norm(dim=-1)

    if mode == "mean":
        target = stacked_target.mean(dim=0)
        prediction = stacked_prediction.mean(dim=0)
    elif mode == "std":
        target = stacked_target.std(dim=0)
        prediction = stacked_prediction.std(dim=0)
    else:
        raise NotImplementedError

    # fluid voidfraction (plot center slice)
    center_target = target[:, target.size(1) // 2]
    center_prediction = prediction[:, prediction.size(1) // 2]
    imshow = axs[0].imshow(
        center_target.T,
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    axs[0].set_title("CFD-DEM", fontsize=20)
    plt.colorbar(imshow, ax=axs[0])
    imshow = axs[1].imshow(
        center_prediction.T,
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    axs[1].set_title("NeuralDEM", fontsize=20)
    plt.colorbar(imshow, ax=axs[1])
    if mode == "mean":
        fig.suptitle(f"Mean {name}", fontsize=30)
    elif mode == "std":
        fig.suptitle(f"Std dev. {name}", fontsize=30)
    else:
      raise NotImplementedError

    # disable ticks
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
