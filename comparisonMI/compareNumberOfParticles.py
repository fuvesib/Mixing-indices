"""
Mixing Indices - Part of the open-access article:
"Mixing indices in up-scaled simulations" (Powder Technology, 2025)
DOI: https://doi.org/10.1016/j.powtec.2025.120775

Author: Balázs Füvesi
License: GPT3 - Please cite the original article if used.

This script is for the article and generates mixing data and creates figures.
"""

import comparisonMI.generatePositionData as gpd
import comparisonMI.wrappers as wp
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os


def generate_mixing_index_files(n_p: int, mi_path: str, pos_path: str, n_saves: int):
    r_particle = 0.5 / n_p

    pos_file = f"np{n_p}_pos.npz"
    mi_path_prefix = mi_path + f"np{n_p}_"

    if wp.cb_check_computed(mi_path_prefix) and wp.db_check_computed(mi_path_prefix):
        print(f"CB DB mixing index files already exist. np:{n_p}")
    else:
        pos = gpd.generate_pos_general(pos_path, pos_file, n_p, n_saves)
        trc = pos[:, 1, 0] > 0.5

        print(f"Generating mixing index files. np:{n_p}")
        os.makedirs(mi_path, exist_ok=True)

        wp.cb_wrapper(pos, trc, r_particle, mi_path_prefix)
        wp.db_wrapper(pos, trc, r_particle, mi_path_prefix)


def create_plot_general(x, Y, mi_names, munu=False) -> tuple:
    fig, axs = plt.subplots(1, 3, figsize=(10, 4), sharey=True, dpi=100)

    parameters = np.zeros((len(mi_names), 2))

    cmap = matplotlib.colormaps["tab20"]

    ref = [1e3, 1e6]
    if munu:
        axs[0].text(1e3, 100 * 1e3 ** (-0.5), r"$10^{2}P^{-1/2}$")
    else:
        axs[0].text(1e3, 10 * 1e3 ** (-0.5), r"$10^{1}P^{-1/2}$")
    axs[0].text(1e3, 1e3 ** (-1.0), r"$P^{-1}$")
    for ax in axs:
        if munu:
            ax.plot(ref, 100 * np.power(ref, -0.5), linewidth=1.0, linestyle="--", color="#595C61", label="_")
        else:
            ax.plot(ref, 10 * np.power(ref, -0.5), linewidth=1.0, linestyle="--", color="#595C61", label="_")
        ax.plot(ref, np.power(ref, -1.0), linewidth=1.0, linestyle="--", color="#595C61", label="_")

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

        x0 = 1000
        xl = np.log10(x[y > 0.0] / x0)
        yl = np.log10(y[y > 0.0])
        p = np.polyfit(xl, yl, 1)
        parameters[j, :] = np.array([10 ** p[1], p[0]])

        x_ = x
        y_ = (10 ** p[1]) * ((x_ / x0) ** p[0])
        axs[axid].plot(x_, y_, ":", label=f"_{mi_name}".upper(), color=cmap(cmap_idx + 1))

    x_ticks = np.array([32, 64, 128, 256, 512, 1024])
    for ax in axs:
        ax.legend(loc="lower left")
        ax.set_xlabel("P [-]")
        ax.set_xscale("log", base=2)
        ax.set_xticks(x_ticks**2)

    axs[0].set_title("Grid-based")
    axs[1].set_title("Contact-based")
    axs[2].set_title("Distance-based")

    return (axs, parameters)


def create_plot_plateaumean(Ms, list_n_particles_view, mi_names: str):
    plateaumean = np.abs(1.0 - np.mean(Ms[:, :, PLATEAUIDX:], axis=2))

    (axs, parameters) = create_plot_general(list_n_particles_view**2.0, plateaumean, mi_names, True)

    axs[0].set_ylim([1e-6, 1e1])
    axs[0].set_yscale("log")
    axs[0].set_ylabel(r"$|1-M_{\infty}$| [-]")

    plt.tight_layout()

    return parameters


def create_plot_plateaunoise(Ms, list_n_particles_view, mi_names: str):
    noise = np.std(Ms[:, :, PLATEAUIDX:], axis=2)

    (axs, parameters) = create_plot_general(list_n_particles_view**2.0, noise, mi_names)

    axs[0].set_ylim([1e-6, 1e0])
    axs[0].set_yscale("log")
    axs[0].set_ylabel(r"$\nu_{\infty}$ [-]")

    plt.tight_layout()

    return parameters


def create_plot_all(Ms, n_particles_plot, mi_names: str):
    cmap = matplotlib.colormaps["viridis"]
    x = np.arange(1025) / 1024
    for j, mi_name in enumerate(mi_names):
        plt.figure()
        for i, n_particles in reversed(list(enumerate(n_particles_plot))):
            M = Ms[i, j, :]
            col = cmap((np.log2(n_particles) - min(np.log2(n_particles_plot))) / (max(np.log2(n_particles_plot)) - min(np.log2(n_particles_plot))))
            n = int(np.log2(n_particles**2))
            plt.plot(x, M, color=col, label=f"$2^{'{'}{n}{'}'}$")

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
    mi_buffer_path = buffer_path + "numberOfParticles/"
    pos_buffer_path = buffer_path + "positions/"

    list_n_particles = np.arange(32, 1024 + 1, 32)
    n_saves = 1024

    if GENERATE:
        for n_p in list_n_particles:
            generate_mixing_index_files(n_p, mi_buffer_path, pos_buffer_path, n_saves)

    if PLOT:
        mi_names_gb = ["rsd", "li", "me", "vbbc", "vrr"]
        mi_names_cb = ["cnr", "ps", "si"]
        mi_names_db = ["ah", "ndm", "nnm", "ssm"]
        mi_names = mi_names_gb + mi_names_cb + mi_names_db

        list_n_particles_view = list_n_particles
        n_cell = 16

        Ms = np.zeros((len(list_n_particles_view), len(mi_names), n_saves + 1))
        for j, mi_name in enumerate(mi_names):
            x = np.arange(1025) / 1024

            if mi_name == "cnr":
                mi_name = "cn"

            for i, n_p in enumerate(list_n_particles_view):
                if mi_name in mi_names_gb:
                    mi_path = buffer_path + f"/numberOfCells/np{n_p}_nc{n_cell}_{mi_name}.npz"
                else:
                    mi_path = mi_buffer_path + f"/np{n_p}_{mi_name}.npz"

                buffer = np.load(mi_path, allow_pickle=False)

                Ms[i, j, :] = buffer["M"]

        params_mean = create_plot_plateaumean(Ms, list_n_particles_view, mi_names)
        plt.savefig(f"NoP_mean.png", dpi=600)
        params_noise = create_plot_plateaunoise(Ms, list_n_particles_view, mi_names)
        plt.savefig(f"NoP_noise.png", dpi=600)

        # idex_toplot = [0,1,3,7,15]
        # create_plot_all(Ms[idex_toplot,:,:], list_n_particles_view[idex_toplot], mi_names)

        pma = params_mean[:, 0]
        pmb = params_mean[:, 1]
        pna = params_noise[:, 0]
        pnb = params_noise[:, 1]
        params_final = np.array([pma, pmb, pna, pnb])
        np.savetxt("NPparams.txt", params_final.T)

        plt.show()
