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
import matplotlib
import numpy as np
import os


def generate_mixing_index_files(n_p: int, n_cell: int, frac: float, mi_path: str, pos_path: str, n_saves: int):
    grid_bounds = np.array([[0, 0, 0], [1, 1, 1]])
    grid_size = np.array([n_cell, n_cell, 1])
    r_particle = 0.5 / n_p

    pos_file = f"np{n_p}_pos.npz"
    mi_path_prefix = mi_path + f"f{frac:.2f}_np{n_p}_".replace(".", "")
    mi_path_prefix_gb = mi_path_prefix + f"nc{n_cell}_"
    mask_file = f"f{frac:.2f}_np{n_p}_mask.npz"

    if (
        wp.gb_check_computed(mi_path_prefix_gb)
        and wp.cb_check_computed(mi_path_prefix)
        and wp.db_check_computed(mi_path_prefix)
    ):
        print(f"GB CB DB mixing index files already exist. np:{n_p} ncell:{n_cell} f:{frac:.2f}")
    else:
        pos = gpd.generate_pos_general(pos_path, pos_file, n_p, n_saves)
        mask = gpd.generate_mask_solidfraction(pos_path, mask_file, frac, pos.shape[0])
        pos = pos[mask, :, :]
        trc = pos[:, 1, 0] > 0.5

        print(f"Generating mixing index files. np:{n_p} ncell:{n_cell} f:{frac:.2f}")
        os.makedirs(mi_path, exist_ok=True)

        wp.gb_wrapper(pos, trc, grid_bounds, grid_size, mi_path_prefix_gb)
        wp.cb_wrapper(pos, trc, r_particle, mi_path_prefix)
        wp.db_wrapper(pos, trc, r_particle, mi_path_prefix)


def create_plot_general(x, Y, mi_names, is_mean) -> tuple:
    fig, axs = plt.subplots(1, 3, figsize=(10, 4), sharey=True, dpi=100)

    parameters = np.zeros((len(mi_names), 2))

    cmap = matplotlib.colormaps["tab20"]

    ref = [0.1, 1.0]
    if is_mean:
        a1 = 1e-1
        a2 = 1e-6
        axs[0].text(0.6, a1 * 0.6 ** (-0.5), r"$10^{-1}x^{-1/2}$")
        axs[0].text(0.6, a2 * 0.6 ** (-1.0), r"$10^{-6}x^{-1}$")
    else:
        a1 = 1e-2
        a2 = 1e-5
        axs[0].text(0.6, a1 * 0.6 ** (-0.5), r"$10^{-2}x^{-1/2}$")
        axs[0].text(0.6, a2 * 0.6 ** (-1.0), r"$10^{-5}x^{-1}$")

    for ax in axs:
        ax.plot(ref, a1 * np.power(ref, -0.5), linewidth=1.0, linestyle="--", color="#595C61", label="_")
        ax.plot(ref, a2 * np.power(ref, -1.0), linewidth=1.0, linestyle="--", color="#595C61", label="_")

    for j, mi_name in enumerate(mi_names):
        if mi_name in mi_names_gb:
            axid = 0
            cmap_idx = 2 * j
        elif mi_name in mi_names_cb:
            axid = 1
            cmap_idx = 2 * (j - len(mi_names_gb))
        elif mi_name in mi_names_db:
            axid = 2
            cmap_idx = 2 * (j - (len(mi_names_gb) + len(mi_names_cb)))

        y = Y[:, j]
        axs[axid].plot(x, y, label=f"{mi_name}".upper(), color=cmap(cmap_idx))

        x0 = 0.1
        xl = np.log10(x[y > 0.0] / x0)
        yl = np.log10(y[y > 0.0])
        p = np.polyfit(xl, yl, 1)
        parameters[j, :] = np.array([10 ** p[1], p[0]])

        x_ = x
        y_ = (10 ** p[1]) * ((x_ / x0) ** p[0])
        axs[axid].plot(x_, y_, ":", label=f"_{mi_name}".upper(), color=cmap(cmap_idx + 1))

    for ax in axs:
        ax.legend(loc="lower left")
        ax.set_xlabel("Solid fraction [-]")
        ax.set_xscale("log")

    axs[0].set_title("Grid-based")
    axs[1].set_title("Contact-based")
    axs[2].set_title("Distance-based")

    return (axs, parameters)


def create_plot_plateaumean(Ms, fractions, mi_names: str):
    plateaumean = np.abs(1.0 - np.mean(Ms[:, :, PLATEAUIDX:], axis=2))

    (axs, parameters) = create_plot_general(fractions, plateaumean, mi_names, True)

    axs[0].set_ylim([1e-7, 1e0])
    axs[0].set_yscale("log")
    axs[0].set_ylabel(r"$|1-M_{\infty}$| [-]")

    plt.tight_layout()

    return parameters


def create_plot_plateaunoise(Ms, fractions, mi_names: str):
    noise = np.std(Ms[:, :, PLATEAUIDX:], axis=2)

    (axs, parameters) = create_plot_general(fractions, noise, mi_names, False)

    axs[0].set_ylim([1e-6, 5e-2])
    axs[0].set_yscale("log")
    axs[0].set_ylabel(r"$\nu_{\infty}$ [-]")

    plt.tight_layout()

    return parameters


def create_plot_all(Ms, fractions, mi_names: str):
    cmap = matplotlib.colormaps["viridis"]
    x = np.arange(1025) / 1024
    for j, mi_name in enumerate(mi_names):
        plt.figure()
        for i, fraction in enumerate(fractions):
            M = Ms[i, j, :]
            plt.plot(x, M, color=cmap(fraction), label=f"{fraction:.2}")

        plt.title(f"{mi_name}".upper())
        plt.ylim([-0.1, 1.1])
        plt.xlim([0.0, 1.0])
        plt.xlabel("Normalised iterations [-]")
        plt.ylabel("Mixedness [-]")
        plt.legend()


if __name__ == "__main__":
    GENERATE = True
    PLOT = True
    PLATEAUIDX = 768

    buffer_path = os.path.dirname(os.path.abspath(__file__)) + "/buffer/"
    mi_buffer_path = buffer_path + "solidFraction/"
    pos_buffer_path = buffer_path + "positions/"

    list_n_particles = np.arange(32, 1024 + 1, 32)
    list_n_cells = np.arange(2, 64 + 1, 2)
    list_fractions = np.linspace(0.05, 1.0, 20)
    n_saves = 1024

    if GENERATE:
        for n_p in list_n_particles:
            for n_cell in list_n_cells:
                for frac in list_fractions:
                    generate_mixing_index_files(n_p, n_cell, frac, mi_buffer_path, pos_buffer_path, n_saves)

    if PLOT:
        mi_names_gb = ["rsd", "li", "me", "vbbc", "vrr"]
        mi_names_cb = ["cnr", "ps", "si"]
        mi_names_db = ["ah", "ndm", "nnm", "ssm"]
        mi_names = mi_names_gb + mi_names_cb + mi_names_db

        list_fractions_view = np.linspace(0.1, 1.0, 19)
        n_p = 512
        n_cell = 16

        Ms = np.zeros((len(list_fractions_view), len(mi_names), n_saves + 1))
        for j, mi_name in enumerate(mi_names):
            if mi_name == "cnr":
                mi_name = "cn"

            for i, frac in enumerate(list_fractions_view):
                if mi_name in mi_names_gb:
                    mi_path = mi_buffer_path + f"f{frac:.2f}_np{n_p}_nc{n_cell}_".replace(".", "") + f"{mi_name}.npz"
                else:
                    mi_path = mi_buffer_path + f"f{frac:.2f}_np{n_p}_".replace(".", "") + f"{mi_name}.npz"

                buffer = np.load(mi_path, allow_pickle=False)
                M = buffer["M"]

                Ms[i, j, :] = M

        params_mean = create_plot_plateaumean(Ms, list_fractions_view, mi_names)
        plt.savefig(f"SF_mean.png", dpi=600)
        params_noise = create_plot_plateaunoise(Ms, list_fractions_view, mi_names)
        plt.savefig(f"SF_noise.png", dpi=600)

        # create_plot_all(Ms[0:19:2,:,:], list_fractions_view[0:19:2], mi_names)

        pma = params_mean[:, 0]
        pmb = params_mean[:, 1]
        pna = params_noise[:, 0]
        pnb = params_noise[:, 1]
        params_final = np.array([pma, pmb, pna, pnb])
        np.savetxt("SFparams.txt", params_final.T)

        plt.show()
