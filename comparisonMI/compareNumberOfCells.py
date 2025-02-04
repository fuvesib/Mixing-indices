import comparisonMI.generatePositionData as gpd
import comparisonMI.wrappers as wp
import matplotlib.pyplot as plt
import numpy as np
import os


def generate_mixing_index_files(n_p: int, n_cell: int, mi_path: str, pos_path: str, n_saves: int):
    grid_bounds = np.array([[0, 0, 0], [1, 1, 1]])
    grid_size = np.array([n_cell, n_cell, 1])

    pos_file = f"np{n_p}_pos.npz"
    mi_path_prefix = mi_path + f"np{n_p}_nc{n_cell}_"

    if wp.gb_check_computed(mi_path_prefix):
        print(f"GB mixing index files already exist. np:{n_p} ncell:{n_cell}")
    else:
        pos = gpd.generate_pos_general(pos_path, pos_file, n_p, n_saves)
        trc = pos[:, 1, 0] > 0.5

        print(f"Generating mixing index files. np:{n_p} ncell:{n_cell}")
        os.makedirs(mi_path, exist_ok=True)

        wp.gb_wrapper(pos, trc, grid_bounds, grid_size, mi_path_prefix)


def isoline_coords(list_n_particles_view, list_n_cells_view, levels=np.logspace(0, 5, 6)) -> tuple:
    x_min = np.min(list_n_cells_view**2)
    x_max = np.max(list_n_cells_view**2)
    y_min = np.min(list_n_particles_view**2)
    y_max = np.max(list_n_particles_view**2)

    x_pairs = np.zeros((2, len(levels)))
    y_pairs = np.zeros((2, len(levels)))

    for i, l in enumerate(levels):
        x_value_l = y_min / l
        x_value_r = x_max
        y_value_b = y_min
        y_value_u = x_max * l
        if x_value_l < x_min:
            x_value_l = x_min
            y_value_b = x_value_l * l
        if y_value_u > y_max:
            y_value_u = y_max
            x_value_r = y_value_u / l
        x_pairs[:, i] = [x_value_l, x_value_r]
        y_pairs[:, i] = [y_value_b, y_value_u]

    return (x_pairs, y_pairs)


def isoline_coords2(list_n_particles_view, list_n_cells_view, levels=np.logspace(3, 10, 8)) -> tuple:
    x_min = np.min(list_n_cells_view**2)
    x_max = np.max(list_n_cells_view**2)
    y_min = np.min(list_n_particles_view**2)
    y_max = np.max(list_n_particles_view**2)

    x_pairs = np.zeros((2, len(levels)))
    y_pairs = np.zeros((2, len(levels)))

    for i, l in enumerate(levels):
        x_value_l = x_min
        x_value_r = x_max
        y_value_b = (x_min * l) ** .5
        y_value_u = (x_max * l) ** .5
        if y_value_b < y_min:
            x_value_l = (y_min ** 2) / l
            y_value_b = y_min
        if y_value_u > y_max:
            x_value_r = (y_max ** 2) / l
            y_value_u = y_max
        x_pairs[:, i] = [x_value_l, x_value_r]
        y_pairs[:, i] = [y_value_b, y_value_u]

    return (x_pairs, y_pairs)


def XY_maskeddata(list_n_particles_view, list_n_cells_view, data) -> tuple:
    X, Y = np.meshgrid(np.array(list_n_cells_view) ** 2.0, np.array(list_n_particles_view) ** 2.0)
    data[Y / X <= 1.0] = np.nan
    return (X, Y, data)


def make_axis(ax):
    x_ticks = np.array([2, 4, 8, 16, 32, 64])
    y_ticks = np.array([32, 64, 128, 256, 512, 1024])

    ax.set_xscale("log", base=2)
    ax.set_xticks(x_ticks**2)

    ax.set_yscale("log", base=2)
    ax.set_yticks(y_ticks**2)
    
    ax.set_xlabel("S [-]")
    ax.set_ylabel("P [-]")


def create_contourplot_startvalue(Ms, list_n_particles_view, list_n_cells_view, mi_name: str):
    startvalue = Ms[:, :, 0]

    levels = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1]
    colors = ["#bddf26",  "#44bf70", "#21918c", "#355f8d", "#482475"]
    cbar_ticklabels = list((str(x) for x in levels))
    cbar_ticklabels[-1] = ">" + cbar_ticklabels[-1]

    startvalue_ = np.clip(np.abs(0-startvalue), min(levels), max(levels))
    (X, Y, startvalue_) = XY_maskeddata(list_n_particles_view, list_n_cells_view, startvalue_)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    plt.contourf(X, Y, startvalue_, levels=levels, colors=colors)

    make_axis(ax)

    plt.suptitle(r"$|0-M_0$|")
    plt.title(f"Index: {mi_name.upper()}", fontsize=8)
    plt.colorbar(ticks=levels).set_ticklabels(cbar_ticklabels)

    plt.tight_layout()


def create_contourplot_plateaumean(Ms, list_n_particles_view, list_n_cells_view, mi_name: str):
    plateau_mean = np.mean(Ms[:, :, PLATEAUIDX:], axis=2)

    levels = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1]
    colors = ["#bddf26",  "#44bf70", "#21918c", "#355f8d", "#482475"]
    cbar_ticklabels = list((str(x) for x in levels))
    cbar_ticklabels[-1] = ">" + cbar_ticklabels[-1]

    plateau_mean_ = np.clip(np.abs(1-plateau_mean), min(levels), max(levels))
    (X, Y, plateau_mean_) = XY_maskeddata(list_n_particles_view, list_n_cells_view, plateau_mean_)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    plt.contourf(X, Y, plateau_mean_, levels=levels, colors=colors)

    col = "#FF7F0E"
    (x_pairs, y_pairs) = isoline_coords(list_n_particles_view, list_n_cells_view)
    for i in range(x_pairs.shape[1]):
        label = r"$10^"+f"{i}"+r"$"
        if (i == 0):
            label = r"n=" + label
        plt.plot(x_pairs[:, i], y_pairs[:, i], "k-", linewidth=1, alpha=0.5)
        ax.text(x_pairs[0, i], y_pairs[0, i] * 1.2, label, size="small", rotation=31)

    make_axis(ax)

    plt.suptitle(r"$|1-M_{\infty}$|")
    plt.title(f"Index: {mi_name.upper()}", fontsize=8)

    plt.colorbar(ticks=levels).set_ticklabels(cbar_ticklabels)

    plt.tight_layout()


def create_contourplot_transientslope(Ms, list_n_particles_view, list_n_cells_view, mi_name: str):
    x = np.arange(1025) / 1024
    transientslope = np.zeros((len(list_n_particles_view), len(list_n_cells_view)))
    for i in range(len(list_n_particles_view)):
        for j in range(len(list_n_cells_view)):
            poly = np.polyfit(x[:SLOPEIDX], Ms[i, j, :SLOPEIDX], 1)
            transientslope[i, j] = poly[0]

    levels_set = False
    if mi_name == "rsd":
        levels = [5, 6, 7, 8, 9, 10]
        levels_set = True

    if mi_name == "li":
        levels = [10, 11, 12, 13, 14, 15]
        levels_set = True

    if mi_name == "me":
        levels = [13, 14, 15, 16, 17, 18]
        levels_set = True

    if mi_name == "vbbc":
        levels = [10, 11, 12, 13, 14, 15]
        levels_set = True

    if mi_name == "vrr":
        levels = [10, 11, 12, 13, 14, 15]
        levels_set = True

    if levels_set:
        colors = ["#bddf26",  "#44bf70", "#21918c", "#355f8d", "#482475"]
        cbar_ticklabels = list((str(x) for x in levels))
        cbar_ticklabels[-1] = ">" + cbar_ticklabels[-1]
        cbar_ticklabels[0] = "<" + cbar_ticklabels[0]
        transientslope = np.clip(transientslope, min(levels), max(levels))

    (X, Y, transientslope) = XY_maskeddata(list_n_particles_view, list_n_cells_view, transientslope)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    if levels_set:
        plt.contourf(X, Y, transientslope, levels=levels, colors=colors)
    else:
        plt.contourf(X, Y, transientslope)

    (x_pairs, y_pairs) = isoline_coords(list_n_particles_view, list_n_cells_view)
    for i in range(x_pairs.shape[1]):
        label = r"$10^"+f"{i}"+r"$"
        if (i == 0):
            label = r"n=" + label
        plt.plot(x_pairs[:, i], y_pairs[:, i], "k-", linewidth=1, alpha=0.5)
        ax.text(x_pairs[0, i], y_pairs[0, i] * 1.2, label, size="small", rotation=31)

    make_axis(ax)

    plt.suptitle(r"$\theta$")
    plt.title(f"Index: {mi_name.upper()}", fontsize=8)

    if levels_set:
        plt.colorbar(ticks=levels).set_ticklabels(cbar_ticklabels)
    else:
        plt.colorbar()

    plt.tight_layout()


def create_contourplot_plateaunoise(Ms, list_n_particles_view, list_n_cells_view, mi_name: str):
    noise = np.std(Ms[:, :, PLATEAUIDX:], axis=2)

    levels = np.power(10.0, np.array([-6, -5, -4, -3, -2, -1]))
    colors = ["#bddf26", "#44bf70", "#21918c", "#355f8d", "#482475"]

    cbar_ticklabels = list((str(x) for x in levels))
    cbar_ticklabels[-1] = ">" + cbar_ticklabels[-1]
    cbar_ticklabels[0] = "<" + cbar_ticklabels[0]

    noise_magintude = np.clip(noise, min(levels), max(levels))
    (X, Y, noise_magintude) = XY_maskeddata(list_n_particles_view, list_n_cells_view, noise_magintude)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    plt.contourf(X, Y, noise_magintude, levels=levels, colors=colors)

    (x_pairs, y_pairs) = isoline_coords2(list_n_particles_view, list_n_cells_view)
    for i in range(x_pairs.shape[1]):
        label = r"$10^{"+f"{i+3}"+r"}$"
        if (i == 0):
            label = r"nP=" + label
        plt.plot(x_pairs[:, i], y_pairs[:, i], "k-", linewidth=1, alpha=0.5)
        ax.text(x_pairs[0, i], y_pairs[0, i] * 1.2, label, size="small", rotation=31)

    make_axis(ax)

    plt.suptitle(r"$\nu_{\infty}$")
    plt.title(f"Index: {mi_name.upper()}", fontsize=8)
    plt.colorbar(ticks=levels).set_ticklabels(cbar_ticklabels)

    plt.tight_layout()


if __name__ == "__main__":
    GENERATE = True
    PLOT = True
    PLATEAUIDX = 768
    SLOPEIDX = 32
    buffer_path = os.path.dirname(os.path.abspath(__file__)) + "/buffer/"
    mi_buffer_path = buffer_path + "numberOfCells/"
    pos_buffer_path = buffer_path + "positions/"

    list_n_particles = np.arange(32, 1024 + 1, 32)
    list_n_cells = np.arange(2, 64 + 1, 2)
    n_saves = 1024

    if GENERATE:
        for n_particles in list_n_particles:
            for n_cells in list_n_cells:
                generate_mixing_index_files(n_particles, n_cells, mi_buffer_path, pos_buffer_path, n_saves)

    if PLOT:
        mi_names = ["rsd", "li", "me", "vbbc", "vrr"]

        x = np.arange(1025) / 1024
        for mi_name in mi_names:
            Ms = np.zeros((len(list_n_particles), len(list_n_cells), n_saves + 1))
            for i, n_particles in enumerate(list_n_particles):
                for j, n_cells in enumerate(list_n_cells):
                    mi_file_path = mi_buffer_path + f"np{n_particles}_nc{n_cells}_{mi_name}.npz"
                    buffer = np.load(mi_file_path, allow_pickle=False)
                    Ms[i, j, :] = buffer["M"]

            create_contourplot_plateaumean(Ms, list_n_particles, list_n_cells, mi_name)
            plt.savefig(f"NoC_{mi_name.upper()}_mean.png", dpi=600)

            create_contourplot_plateaunoise(Ms, list_n_particles, list_n_cells, mi_name)
            plt.savefig(f"NoC_{mi_name.upper()}_noise.png", dpi=600)

            create_contourplot_transientslope(Ms, list_n_particles, list_n_cells, mi_name)
            plt.savefig(f"NoC_{mi_name.upper()}_slope.png", dpi=600)

            create_contourplot_startvalue(Ms, list_n_particles, list_n_cells, mi_name)
            plt.savefig(f"NoC_{mi_name.upper()}_start.png", dpi=600)

        plt.show()
