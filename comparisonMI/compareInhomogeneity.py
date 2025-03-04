"""
Mixing Indices - Part of the open-access article:
"Mixing indices in up-scaled simulations" (Powder Technology, 2025)
DOI: https://doi.org/10.1016/j.powtec.2025.120775

Author: Balázs Füvesi
License: GPL3 - Please cite the original article if used.

This script is for the article and generates mixing data and creates figures.
"""

import comparisonMI.generatePositionData as gpd
import comparisonMI.wrappers as wp
import matplotlib.pyplot as plt
import numpy as np
import os


def generate_mixing_index_files(n_p: int, n_cell: int, mi_path: str, pos_path: str, n_saves: int):
    grid_bounds = np.array([[0, 0, 0], [1, 1, 1]])
    grid_size = np.array([n_cell, n_cell, 1])
    r_particle = 0.5 / n_p

    pos_file = f"ih_np{n_p}_pos.npz"
    mi_path_prefix = mi_path + f"np{n_p}_"
    mi_path_prefix_gb = mi_path_prefix + f"nc{n_cell}_"

    if (
        wp.gb_check_computed(mi_path_prefix_gb)
        and wp.cb_check_computed(mi_path_prefix)
        and wp.db_check_computed(mi_path_prefix)
    ):
        print(f"GB CB DB mixing index files already exist. np:{n_p} ncell:{n_cell}")
    else:
        pos = gpd.generate_pos_inhomogeneous(pos_path, pos_file, n_p, n_saves)
        trc = pos[:, 1, 0] > 0.5

        print(f"Generating mixing index files. np:{n_p} ncell:{n_cell}")
        os.makedirs(mi_path, exist_ok=True)

        wp.gb_wrapper(pos, trc, grid_bounds, grid_size, mi_path_prefix_gb)
        wp.cb_wrapper(pos, trc, r_particle, mi_path_prefix)
        wp.db_wrapper(pos, trc, r_particle, mi_path_prefix)


def create_plot_plateaumean(Ms_ref1, Ms_ref2, Ms_ih, mi_names: str):
    plateaumean_gt = np.abs(1.0 - np.mean(Ms_ref1[:, PLATEAUIDX:], axis=1))
    plateaumean_gt2 = np.abs(1.0 - np.mean(Ms_ref2[:, PLATEAUIDX:], axis=1))
    plateaumean_hi = np.abs(1.0 - np.mean(Ms_ih[:, PLATEAUIDX:], axis=1))

    bar_width = 0.3
    fig, ax = plt.subplots(figsize=(8.2, 4), dpi=100)
    x = np.arange(len(mi_names))
    plt.bar(x - bar_width / 1.5, plateaumean_gt, bar_width / 2, label="Homogeneous dense")
    plt.bar(x, plateaumean_hi, bar_width / 2, label="Inhomogeneous")
    plt.bar(x + bar_width / 1.5, plateaumean_gt2, bar_width / 2, label="Homogeneous loose")
    ax.set_xticks(x)
    ax.set_xticklabels([mi_name.upper() for mi_name in mi_names])
    ax.set_yscale("log")
    ax.set_ylabel(r"$|1-M_{\infty}$| [-]")
    plt.legend()
    plt.title("Mixing indices")

    plt.tight_layout()


def create_plot_plateaunoise(Ms_ref1, Ms_ref2, Ms_ih, mi_names: str):
    noise_gt = np.std(Ms_ref1[:, PLATEAUIDX:], axis=1)
    noise_gt2 = np.std(Ms_ref2[:, PLATEAUIDX:], axis=1)
    noise_hi = np.std(Ms_ih[:, PLATEAUIDX:], axis=1)

    bar_width = 0.3
    fig, ax = plt.subplots(figsize=(8.2, 4), dpi=100)
    x = np.arange(len(mi_names))
    plt.bar(x - bar_width / 1.5, noise_gt, bar_width / 2, label="Homogeneous dense")
    plt.bar(x, noise_hi, bar_width / 2, label="Inhomogeneous")
    plt.bar(x + bar_width / 1.5, noise_gt2, bar_width / 2, label="Homogeneous loose")
    ax.set_xticks(x)
    ax.set_xticklabels([mi_name.upper() for mi_name in mi_names])
    ax.set_yscale("log")
    plt.legend()
    ax.set_ylabel(r"$\nu_{\infty}$ [-]")
    plt.title("Mixing indices")

    plt.tight_layout()


def create_plot_all(Ms_ref1, Ms_ref2, Ms_ih, mi_names: str):
    x = np.arange(1025) / 1024
    for j, mi_name in enumerate(mi_names):

        plt.figure()
        plt.plot(x, Ms_ref1[j, :], label="Reference1")
        plt.plot(x, Ms_ih[j, :], label="Case")
        plt.plot(x, Ms_ref2[j, :], label="Reference2")
        plt.title(f"{mi_name}".upper())
        plt.ylim([-0.1, 1.1])
        plt.xlim([0.0, 1.0])
        plt.xlabel("Normalised iterations [-]")
        plt.ylabel("Mixedness [-]")
        plt.legend(loc="lower right")


if __name__ == "__main__":
    GENERATE = True
    PLOT = True
    PLATEAUIDX = 768

    buffer_path = os.path.dirname(os.path.abspath(__file__)) + "/buffer/"
    mi_buffer_path = buffer_path + "inhomogeneity/"
    pos_buffer_path = buffer_path + "positions/"

    list_n_particles = np.arange(32, 1024 + 1, 32)
    list_n_cells = np.arange(2, 64 + 1, 2)
    n_saves = 1024

    if GENERATE:
        for n_particles in list_n_particles:
            for n_cells in list_n_cells:
                generate_mixing_index_files(n_particles, n_cells, mi_buffer_path, pos_buffer_path, n_saves)

    if PLOT:
        n_particles = 512
        n_cells = 32
        n_particles_ref1 = 224
        n_cells_ref1 = 16
        n_particles_ref2 = 512
        n_cells_ref2 = 16
        frac_ref2 = 0.2

        mi_names_gb = ["rsd", "li", "me", "vbbc", "vrr"]
        mi_names_cb = ["cn", "ps", "si"]
        mi_names_db = ["ah", "ndm", "nnm", "ssm"]
        mi_names = mi_names_gb + mi_names_cb + mi_names_db

        Ms_ref1 = np.zeros((len(mi_names), n_saves + 1))
        Ms_ref2 = np.zeros((len(mi_names), n_saves + 1))
        Ms_ih = np.zeros((len(mi_names), n_saves + 1))
        for j, mi_name in enumerate(mi_names):
            mi_file_name = f"{mi_name}.npz"

            path_gb_ih = mi_buffer_path + f"np{n_particles}_nc{n_cells}_{mi_name}.npz"
            path_gb_ref1 = buffer_path + f"/numberOfCells/np{n_particles_ref1}_nc{n_cells_ref1}_{mi_name}.npz"
            path_gb_ref2 = buffer_path + f"/solidFraction/f{frac_ref2:.2f}_".replace(".", "") + f"np{n_particles_ref2}_nc{n_cells_ref2}_{mi_name}.npz"
            path_ih = mi_buffer_path + f"np{n_particles}_{mi_name}.npz"
            path_ref1 = buffer_path + f"/numberOfParticles/np{n_particles_ref1}_{mi_name}.npz"
            path_ref2 = buffer_path + f"/solidFraction/f{frac_ref2:.2f}_".replace(".", "") + f"np{n_particles_ref2}_{mi_name}.npz"

            if mi_name in mi_names_gb:
                buffer_ih = np.load(path_gb_ih, allow_pickle=False)
                buffer_ref1 = np.load(path_gb_ref1, allow_pickle=False)
                buffer_ref2 = np.load(path_gb_ref2, allow_pickle=False)
            else:
                buffer_ih = np.load(path_ih, allow_pickle=False)
                buffer_ref1 = np.load(path_ref1, allow_pickle=False)
                buffer_ref2 = np.load(path_ref2, allow_pickle=False)

            Ms_ih[j, :] = buffer_ih["M"]
            Ms_ref1[j, :] = buffer_ref1["M"]
            Ms_ref2[j, :] = buffer_ref2["M"]

        create_plot_all(Ms_ref1, Ms_ref2, Ms_ih, mi_names)

        create_plot_plateaumean(Ms_ref1, Ms_ref2, Ms_ih, mi_names)
        plt.savefig(f"IH_mean.png", dpi=600)

        create_plot_plateaunoise(Ms_ref1, Ms_ref2, Ms_ih, mi_names)
        plt.savefig(f"IH_noise.png", dpi=600)

        plt.show()
