import comparisonMI.generatePositionData as gpd
import comparisonMI.wrappers as wp
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os


def generate_mixing_index_files(n_p: int, n_cell: int, mi_path: str, pos_path: str, n_saves: int):
    grid_bounds = np.array([[0, 0, 0], [1, 1, 1]])
    grid_size = np.array([n_cell, n_cell, 1])
    r_particle = 0.5 / n_p

    pos_file = f"re_np{n_p}_pos.npz"
    mi_path_prefix = mi_path + f"np{n_p}_"
    mi_path_prefix_gb = mi_path_prefix + f"nc{n_cell}_"

    if (
        wp.gb_check_computed(mi_path_prefix_gb)
        and wp.cb_check_computed(mi_path_prefix)
        and wp.db_check_computed(mi_path_prefix)
    ):
        print(f"GB CB DB mixing index files already exist. np:{n_p} ncell:{n_cell}")
    else:
        pos = gpd.generate_pos_permutation(pos_path, pos_file, n_p, n_saves)
        trc = pos[:, 1, 0] > 0.5

        print(f"Generating mixing index files. np:{n_p} ncell:{n_cell}")
        os.makedirs(mi_path, exist_ok=True)

        wp.gb_wrapper(pos, trc, grid_bounds, grid_size, mi_path_prefix_gb)
        wp.cb_wrapper(pos, trc, r_particle, mi_path_prefix)
        wp.db_wrapper(pos, trc, r_particle, mi_path_prefix)


def create_plot_all(Ms, mi_names: str, loc="upper left"):
    fig, axs = plt.subplots(1, 3, figsize=(10, 4), sharey=True, dpi=100)

    cmap = matplotlib.colormaps["tab20"]
    x = 2 * np.arange(1025) / 1024

    for i, mi_name in enumerate(mi_names):
        if mi_name in mi_names_gb:
            axid = 0
            cmap_idx = 2 * i
        elif mi_name in mi_names_cb:
            axid = 1
            cmap_idx = 2 * (i - len(mi_names_gb))
        elif mi_name in mi_names_db:
            axid = 2
            cmap_idx = 2 * (i - (len(mi_names_gb) + len(mi_names_cb)))

        y = Ms[i, :]
        
        axs[axid].plot(x, y, '-', label=f"{mi_name}".upper(), color=cmap(cmap_idx))

    for ax in axs:
        ax.legend(loc=loc)
        ax.set_xlabel(r"$t/t_{\mathrm{max}} [-]$")

    axs[0].set_title("Grid-based")
    axs[1].set_title("Contact-based")
    axs[2].set_title("Distance-based")

    axs[0].set_ylabel("M [-]")
    plt.legend()

    plt.tight_layout()


if __name__ == "__main__":
    GENERATE = True
    PLOT = True
    PLATEAUIDX = 768

    buffer_path = os.path.dirname(os.path.abspath(__file__)) + "/buffer/"
    mi_buffer_path = buffer_path + "reversibility/"
    pos_buffer_path = buffer_path + "positions/"

    list_n_particles = np.arange(64, 1024 + 1, 32)
    list_n_cells = np.arange(2, 64 + 1, 2)
    n_saves = 1024

    if GENERATE:
        for n_particles in list_n_particles:
            for n_cells in list_n_cells:
                generate_mixing_index_files(n_particles, n_cells, mi_buffer_path, pos_buffer_path, n_saves)

    if PLOT:
        n_particles = 512
        n_cell = 16

        mi_names_gb = ["rsd", "li", "me", "vbbc", "vrr"]
        mi_names_cb = ["cnr", "ps", "si"]
        mi_names_db = ["ah", "ndm", "nnm", "ssm"]
        mi_names = mi_names_gb + mi_names_cb + mi_names_db

        Ms = np.zeros((len(mi_names), n_saves + 1))
        for i, mi_name in enumerate(mi_names):
            if mi_name == "cnr":
                mi_name = "cn"

            if mi_name in mi_names_gb:
                mi_path = mi_buffer_path + f"/np{n_particles}_nc{n_cell}_{mi_name}.npz"
            else:
                mi_path = mi_buffer_path + f"/np{n_particles}_{mi_name}.npz"

            buffer = np.load(mi_path, allow_pickle=False)
            Ms[i, :] = buffer["M"]

        create_plot_all(Ms, mi_names, loc="upper left")
        plt.savefig(f"Rev_MIs.png", dpi=600)

        plt.show()
