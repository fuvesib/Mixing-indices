"""
Mixing Indices - Part of the open-access article:
"Mixing indices in up-scaled simulations" (Powder Technology, 2025)
DOI: https://doi.org/10.1016/j.powtec.2025.120775

Author: Balázs Füvesi
License: GPT3 - Please cite the original article if used.

This script is for the article and creates mixing data on a lattice.
"""

import numpy as np
import matplotlib.pyplot as plt


def _create_grid(n_particles_per_dim, segregated=True) -> np.ndarray:
    n_particles_per_dim = np.array(n_particles_per_dim)
    if n_particles_per_dim.shape != (3,):
        raise TypeError(f"Incorrect array shape {n_particles_per_dim.shape}, should be (3,).")
    if n_particles_per_dim.dtype != int:
        raise TypeError(f"Incorrect array dtype {n_particles_per_dim.dtype}, should be int.")

    i, j, k = np.meshgrid(
        np.arange(n_particles_per_dim[2]),
        np.arange(n_particles_per_dim[1]),
        np.arange(n_particles_per_dim[0]),
        indexing="ij",
    )
    stacked = np.column_stack((k.ravel(), j.ravel(), i.ravel()))

    if segregated:
        pos = stacked.astype(np.float64)
    else:
        common_mask = stacked.sum(axis=1) % 2
        mask_A = common_mask != 0
        pos_A = np.column_stack((k.ravel()[mask_A], j.ravel()[mask_A], i.ravel()[mask_A])).astype(np.float64)
        mask_B = common_mask == 0
        pos_B = np.column_stack((k.ravel()[mask_B], j.ravel()[mask_B], i.ravel()[mask_B])).astype(np.float64)
        pos = np.concatenate((pos_A, pos_B))

    pos += [0.5, 0.5, 0.5]
    pos /= n_particles_per_dim.astype(np.float64)

    return pos


def create_grid_segregated(n_particles_per_dim) -> np.ndarray:
    return _create_grid(n_particles_per_dim, True)


def create_grid_mixed(n_particles_per_dim) -> np.ndarray:
    return _create_grid(n_particles_per_dim, False)


def mixing_permutation(
    pos0: np.ndarray, n_saves: int, n_iterations: int = None, seed_a: int = 0, seed_b: int = 1
) -> np.ndarray:
    n_particles = pos0.shape[0]
    half = n_particles // 2  # Round down
    if n_iterations is None:
        n_iterations = half
    iterations_per_save = n_iterations / n_saves

    pos = np.zeros((n_particles, pos0.shape[1], n_saves + 1))
    pos[:, :, 0] = pos0

    selected_a = np.random.default_rng(seed_a).permutation(half)
    selected_b = half + np.random.default_rng(seed_b).permutation(half)

    n_iterations_done = 0
    remainder = 0.0
    for idx_save in range(n_saves):
        remainder += iterations_per_save
        (remainder, iterations_to_do) = np.modf(remainder)
        iterations_to_do = int(iterations_to_do)

        start = n_iterations_done
        stop = start + iterations_to_do
        pos[:, :, idx_save + 1] = pos[:, :, idx_save]
        pos[selected_a[start:stop], :, idx_save + 1] = pos[selected_b[start:stop], :, idx_save]
        pos[selected_b[start:stop], :, idx_save + 1] = pos[selected_a[start:stop], :, idx_save]
        n_iterations_done += iterations_to_do

    return pos


def mixing_uniform(
    pos0: np.ndarray, n_saves: int, n_iterations: int = None, seed_a: int = 0, seed_b: int = 1
) -> np.ndarray:
    n_particles = pos0.shape[0]
    if n_iterations is None:
        n_iterations = n_particles * 4
    iterations_per_save = n_iterations / n_saves

    pos = np.zeros((n_particles, pos0.shape[1], n_saves + 1))
    pos[:, :, 0] = pos0

    selected_a = np.random.default_rng(seed_a).integers(0, n_particles, n_iterations)
    selected_b = np.random.default_rng(seed_b).integers(0, n_particles, n_iterations)

    n_iterations_done = 0
    remainder = 0.0
    for idx_save in range(n_saves):
        remainder += iterations_per_save
        (remainder, iterations_to_do) = np.modf(remainder)
        iterations_to_do = int(iterations_to_do)

        pos[:, :, idx_save + 1] = pos[:, :, idx_save]
        for idx_sub_iteration in range(iterations_to_do):
            buff = np.array(pos[selected_a[n_iterations_done], :, idx_save + 1])
            pos[selected_a[n_iterations_done], :, idx_save + 1] = pos[selected_b[n_iterations_done], :, idx_save + 1]
            pos[selected_b[n_iterations_done], :, idx_save + 1] = buff
            n_iterations_done += 1

    return pos


def plot_2D_lattice(pos: np.ndarray, trc, size=None):
    plt.figure()
    plt.scatter(pos[trc, 0], pos[trc, 1], s=size, color="#3E758D")
    plt.scatter(pos[np.logical_not(trc), 0], pos[np.logical_not(trc), 1], s=size, color="#D73F4B")
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    plt.xlabel("x [-]")
    plt.ylabel("y [-]")
