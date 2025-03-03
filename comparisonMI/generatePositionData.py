"""
Mixing Indices - Part of the open-access article:
"Mixing indices in up-scaled simulations" (Powder Technology, 2025)
DOI: https://doi.org/10.1016/j.powtec.2025.120775

Author: Balázs Füvesi
License: GPT3 - Please cite the original article if used.

This script is for the article and generates particle positions.
"""

import comparisonMI.latticeMixing as lm
import numpy as np
import os


def _load_pos(pos_path: str, pos_file: str, n_p: int):
    print(f"Loading position data. np:{n_p}")
    buffer = np.load(pos_path + pos_file, allow_pickle=False)
    return buffer["pos"]


def generate_pos_general(pos_path: str, pos_file: str, n_p: int, n_saves: int):
    if os.path.isfile(pos_path + pos_file):
        pos = _load_pos(pos_path, pos_file, n_p)
    else:
        print(f"Generating position data. np:{n_p}")
        os.makedirs(pos_path, exist_ok=True)
        pos0 = lm.create_grid_segregated((n_p, n_p, 1))
        pos = lm.mixing_uniform(pos0, n_saves=n_saves)
        np.savez_compressed(pos_path + pos_file, pos=pos)
    return pos


def generate_pos_permutation(pos_path: str, pos_file: str, n_p: int, n_saves: int):
    if os.path.isfile(pos_path + pos_file):
        pos = _load_pos(pos_path, pos_file, n_p)
    else:
        print(f"Generating position data. np:{n_p}")
        os.makedirs(pos_path, exist_ok=True)
        pos0 = lm.create_grid_segregated((n_p, n_p, 1))
        pos = lm.mixing_permutation(pos0, n_saves=n_saves)
        np.savez_compressed(pos_path + pos_file, pos=pos)
    return pos


def generate_pos_inhomogeneous(pos_path: str, pos_file: str, n_p: int, n_saves: int):
    if os.path.isfile(pos_path + pos_file):
        pos = _load_pos(pos_path, pos_file, n_p)
    else:
        print(f"Generating position data. np:{n_p}")
        os.makedirs(pos_path, exist_ok=True)
        pos0 = lm.create_grid_segregated((n_p, n_p, 1))
        rad = np.sqrt(np.power(pos0[:, 0] - 0.5, 2) + np.power(pos0[:, 1] - 0.5, 2))
        pos0 = pos0[rad >= 0.5, :]
        pos = lm.mixing_uniform(pos0, n_saves=n_saves)
        np.savez_compressed(pos_path + pos_file, pos=pos)
    return pos


def generate_mask_solidfraction(pos_path: str, mask_file: str, frac: float, pos_length):
    if os.path.isfile(pos_path + mask_file):
        print(f"Loading pos mask. f:{frac:.2f}")
        buffer = np.load(pos_path + mask_file, allow_pickle=False)
        mask = buffer["mask"]
    else:
        print(f"Generating pos mask. f:{frac:.2f}")
        os.makedirs(pos_path, exist_ok=True)
        mask = np.random.default_rng(0).random((pos_length,)) < frac
        np.savez_compressed(pos_path + mask_file, mask=mask)
    return mask
