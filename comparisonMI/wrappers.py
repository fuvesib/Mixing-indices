"""
Mixing Indices - Part of the open-access article:
"Mixing indices in up-scaled simulations" (Powder Technology, 2025)
DOI: https://doi.org/10.1016/j.powtec.2025.120775

Author: Balázs Füvesi
License: GPT3 - Please cite the original article if used.

This script is for the article and contains wrapper functions for the evaluation of the indices.
"""

import mixingIndices as mi
import numpy as np
import os


def gb_check_computed(test_dir_path) -> bool:
    if (
        os.path.isfile(test_dir_path + "dm.npz")
        and os.path.isfile(test_dir_path + "rsd.npz")
        and os.path.isfile(test_dir_path + "li.npz")
        and os.path.isfile(test_dir_path + "me.npz")
        and os.path.isfile(test_dir_path + "vbbc.npz")
        and os.path.isfile(test_dir_path + "vrr.npz")
    ):
        return True
    else:
        return False


def cb_check_computed(test_dir_path) -> bool:
    if (
        os.path.isfile(test_dir_path + "cn.npz")
        and os.path.isfile(test_dir_path + "ps.npz")
        and os.path.isfile(test_dir_path + "si.npz")
    ):
        return True
    else:
        return False


def db_check_computed(test_dir_path) -> bool:
    if (
        os.path.isfile(test_dir_path + "ah.npz")
        and os.path.isfile(test_dir_path + "ndm.npz")
        and os.path.isfile(test_dir_path + "ndms.npz")
        and os.path.isfile(test_dir_path + "nnm.npz")
        and os.path.isfile(test_dir_path + "ssm.npz")
    ):
        return True
    else:
        return False


def gb_wrapper(pos, trc, grid_bounds, grid_size, test_dir_path, force_generation=False):
    if (not force_generation) and gb_check_computed(test_dir_path):
        return

    threshold_particle_count = 0

    pgm = mi.ParticleGridMapping(grid_bounds, grid_size)

    dm = mi.DegreeOfMixedness()
    rsd = mi.RelativeStandardDeviation()
    li = mi.LaceyIndex()
    me = mi.MixingEntropy()
    vbbc = mi.VarianceAmongBimodalBinCounts()
    vrr = mi.VarianceReductionRatio()

    npbtic0 = None
    for iter in range(pos.shape[2]):
        npbtic = pgm.particle_type_to_grid(pos[:, :, iter], trc)
        npbtic = np.concatenate((npbtic[1].reshape(-1, 1), npbtic[0].reshape(-1, 1)), axis=1)
        # Delete cells with too few particles
        npbtic = np.delete(npbtic, np.argwhere(np.sum(npbtic, 1) <= threshold_particle_count), 0)
        if iter == 0:
            npbtic0 = npbtic

        dm.iterate(npbtic)
        rsd.iterate(npbtic)
        li.iterate(npbtic)
        me.iterate(npbtic)
        vbbc.iterate(npbtic)
        vrr.iterate(npbtic, npbtic0)

    try:
        np.savez_compressed(test_dir_path + "dm.npz", M=dm.get_M())
    except:
        pass
    try:
        np.savez_compressed(test_dir_path + "rsd.npz", M=rsd.get_M())
    except:
        pass
    try:
        np.savez_compressed(test_dir_path + "li.npz", M=li.get_M())
    except:
        pass
    try:
        np.savez_compressed(test_dir_path + "me.npz", M=me.get_M())
    except:
        pass
    try:
        np.savez_compressed(test_dir_path + "vbbc.npz", M=vbbc.get_M())
    except:
        pass
    try:
        np.savez_compressed(test_dir_path + "vrr.npz", M=vrr.get_M())
    except:
        pass


def cb_wrapper(pos, trc, r_particle, test_dir_path, force_generation=False):
    if (not force_generation) and cb_check_computed(test_dir_path):
        return

    pnl = mi.ParticleNeighborList()

    cn = mi.CoordinationNumber()
    ps = mi.ParticleScaleIndex(n_sample_size=5)
    si = mi.SegregationIndex()

    for iter in range(pos.shape[2]):
        npbtin = pnl.calculate_neighbor_traced(pos[:, :, iter], np.ones(trc.shape) * (r_particle * 1.001), trc)

        cn.iterate(npbtin, trc)
        ps.iterate(npbtin, trc)
        si.iterate(npbtin, trc)

    try:
        np.savez_compressed(test_dir_path + "cn.npz", M=cn.get_M())
    except:
        pass
    try:
        np.savez_compressed(test_dir_path + "ps.npz", M=ps.get_M())
    except:
        pass
    try:
        np.savez_compressed(test_dir_path + "si.npz", M=si.get_M())
    except:
        pass


def db_wrapper(pos, trc, r_particle, test_dir_path, force_generation=False):
    if (not force_generation) and db_check_computed(test_dir_path):
        return

    ah = mi.AverageHeight(axis=1)
    ndm = mi.NeighborDistanceMethod(particle_diameter=r_particle * 2.001)
    ndms = mi.NeighborDistanceMethod(particle_diameter=r_particle * 2.001)
    nnm = mi.NearestNeighborMethod()
    ssm = mi.SphereSpreadingMethod()

    for iter in range(pos.shape[2]):
        ah.iterate(pos[:, :, iter], trc)
        ndm.iterate(pos[:, :, iter])
        ndms.iterate(pos[:, :, iter], trc)
        nnm.iterate(pos[:, :, iter], trc)
        ssm.iterate(pos[:, :, iter], trc)

    try:
        np.savez_compressed(test_dir_path + "ah.npz", M=ah.get_M())
    except:
        pass
    try:
        np.savez_compressed(test_dir_path + "ndm.npz", M=ndm.get_M())
    except:
        pass
    try:
        np.savez_compressed(test_dir_path + "ndms.npz", M=ndms.get_M())
    except:
        pass
    try:
        np.savez_compressed(test_dir_path + "nnm.npz", M=nnm.get_M())
    except:
        pass
    try:
        np.savez_compressed(test_dir_path + "ssm.npz", M=ssm.get_M())
    except:
        pass
