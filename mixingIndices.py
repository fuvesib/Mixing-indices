import numpy
import warnings
from sklearn.neighbors import NearestNeighbors
import scipy.spatial.distance as sciDist
import time


class ParticleGridMapping:
    """Class for particle grid mapping."""

    def __init__(self, grid_bounds: numpy.ndarray, grid_size: numpy.ndarray, particle_max_r: float = 1e-12):
        if not (grid_bounds.shape == (2, 3)):
            raise TypeError("Incorrect gridBounds shape {shape}, should be (2,3)".format(shape=grid_bounds.shape))

        if not (grid_size.shape == (3,)):
            raise TypeError("Incorrect gridSize shape {shape}, should be (3,)".format(shape=grid_size.shape))

        self._lower_bound = grid_bounds[0, :]
        self._upper_bound = grid_bounds[1, :]
        self._grid_size = grid_size
        self._cell_length = (self._upper_bound - self._lower_bound) / self._grid_size
        self._n_cells = self._grid_size[0] * self._grid_size[1] * self._grid_size[2]
        self._particle_max_r = particle_max_r

        if self._particle_max_r * 2 > numpy.min(self._cell_length):
            warnings.warn(
                f"The largest particle diameter {particle_max_r * 2} is greater than the smallest cell length {numpy.min(self._cell_length)} of the grid used to calculate the mixing. This could cause issues!"
            )

    def particle_to_grid(self, pos: numpy.ndarray) -> dict:
        """Calculates the number of particles in each grid cell.
        Returns a dictionary with:
        nParticlesIn_cells - Number of particles in grid cells
        idxOfCell - Corresponding cell index for each particle
        nParticlesOutOfGrid - Number of particles out of grid
        """

        MixingIndex._check_shape_pos(pos)
        if not (pos.shape[1] == 3):
            raise TypeError(f"Incorrect pos shape {pos.shape}, should be (n,3)")

        particles_in_samples = numpy.zeros((self._n_cells), dtype=numpy.int32)

        pos = pos - self._lower_bound

        # index of cell in i,j,k format
        idx_of_cell_3v = numpy.array(numpy.floor(pos / self._cell_length), dtype=numpy.int32)

        # Creating a boolean array that marks the out of grid particles
        particles_out_of_lower_bound = numpy.min(idx_of_cell_3v, 1) < 0
        particles_out_of_upper_bound = numpy.any(numpy.greater_equal(idx_of_cell_3v, self._grid_size), 1)
        particles_out_of_grid = numpy.logical_or(particles_out_of_lower_bound, particles_out_of_upper_bound)

        # Creating an array that contains the cell index of particles in i format, or if they are out of grid
        idx_of_cell = (
            idx_of_cell_3v[:, 0]
            + idx_of_cell_3v[:, 1] * self._grid_size[0]
            + idx_of_cell_3v[:, 2] * self._grid_size[0] * self._grid_size[1]
        )
        idx_of_cell[particles_out_of_grid] = -1

        # Counting how many particles are in the cells by cell index occurance
        # For example if idx_of_cell = [1,0,0,-1,1,1,2,2] -> [-1,0,1,2], [1,2,3,2]
        unique_cell_idx, particle_counts_in_cells = numpy.unique(idx_of_cell, return_counts=True)

        # Discarding the out of grid particles, which are marked with -1
        # The smallest valid cell idx is 0, therefore if there is -1 it should be the 0th
        n_particles_out_of_grid = 0
        if unique_cell_idx[0] == -1:
            n_particles_out_of_grid += particle_counts_in_cells[0]
            unique_cell_idx = numpy.delete(unique_cell_idx, 0)
            particle_counts_in_cells = numpy.delete(particle_counts_in_cells, 0)

        particles_in_samples[unique_cell_idx,] = particle_counts_in_cells

        return {
            "nParticlesIn_cells": particles_in_samples,
            "idxOfCell": idx_of_cell,
            "nParticlesOutOfGrid": n_particles_out_of_grid,
        }

    def particle_to_grid_continuous(self, pos: numpy.ndarray) -> dict:
        """Calculates the number of particles in each grid cell.
        Returns a dictionary with:
        nParticlesIn_cells - Fractional number of particles in grid cells
        idxOfCell - Corresponding cell index for each particle
        nParticlesOutOfGrid - Number of particles out of grid
        """

        MixingIndex._check_shape_pos(pos)
        if not (pos.shape[1] == 3):
            raise TypeError(f"Incorrect pos shape {pos.shape}, should be (n,3)")

        particles_in_samples = numpy.zeros((self._n_cells), dtype=numpy.float64)

        pos = pos - self._lower_bound

        # index of cell in i,j,k format
        idx_of_cell_3v = numpy.array(numpy.floor(pos / self._cell_length), dtype=numpy.int32)

        (unique_cell_idx, particle_counts_in_cells, idx_of_cell_tmp) = self._helper(
            idx_of_cell_3v, numpy.array([0, 0, 0]), pos
        )

        idx_of_cell = None
        n_particles_out_of_grid = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    (unique_cell_idx, particle_counts_in_cells, idx_of_cell_tmp) = self._helper(
                        idx_of_cell_3v, numpy.array([i, j, k]), pos
                    )
                    if (i == 0) and (j == 0) and (k == 0):
                        idx_of_cell = idx_of_cell_tmp

                    # Discarding the out of grid particles, which are marked with -1
                    # The smallest valid cell idx is 0, therefore if there is -1 it should be the 0th
                    if unique_cell_idx[0] == -1:
                        if (i == 0) and (j == 0) and (k == 0):
                            n_particles_out_of_grid += numpy.sum(idx_of_cell_tmp == -1)
                        unique_cell_idx = numpy.delete(unique_cell_idx, 0)
                        particle_counts_in_cells = numpy.delete(particle_counts_in_cells, 0)

                    particles_in_samples[unique_cell_idx,] += particle_counts_in_cells

        return {
            "nParticlesIn_cells": particles_in_samples,
            "idxOfCell": idx_of_cell,
            "nParticlesOutOfGrid": n_particles_out_of_grid,
        }

    def particle_type_to_grid(self, pos: numpy.ndarray, typ: numpy.ndarray) -> dict:
        """Calculates the number of particles in each grid cell by type.
        Returns a dictionary with:
        (type number) - Number of particles by type in each grid cell
        nParticlesOutOfGrid - Number of particles out of grid regardless of type
        """

        if not (pos.shape[0] == typ.shape[0]):
            raise TypeError(f"Incorrect pos shape {pos.shape} or typ shape {typ.shape}, should be (n,3) and (n,)")
        if not (pos.ndim == 2):
            raise TypeError(f"Incorrect pos shape {pos.shape}, should be (n,3)")
        if not (typ.ndim == 1):
            raise TypeError(f"Incorrect typ shape {typ.shape}, should be (n,)")

        return_dict = dict()

        # Identifying the types
        unique_typs = numpy.unique(typ)

        # Makeing the particle to grid for each type
        n_particles_out_of_grid = 0
        for unique_typ in unique_typs:
            pos_with_unique_typ = pos[numpy.equal(typ, unique_typ), :]
            results = self.particle_to_grid(pos_with_unique_typ)
            return_dict[unique_typ] = results["nParticlesIn_cells"]
            n_particles_out_of_grid += results["nParticlesOutOfGrid"]

        return_dict["nParticlesOutOfGrid"] = n_particles_out_of_grid

        return return_dict

    def particle_type_to_grid_continuous(self, pos: numpy.ndarray, typ: numpy.ndarray) -> dict:
        """Calculates the number of particles in each grid cell by type.
        Returns a dictionary with:
        (type number) - Number of particles by type in each grid cell
        nParticlesOutOfGrid - Number of particles out of grid regardless of type
        """

        if not (pos.shape[0] == typ.shape[0]):
            raise TypeError(f"Incorrect pos shape {pos.shape} or typ shape {typ.shape}, should be (n,3) and (n,)")
        if not (pos.ndim == 2):
            raise TypeError(f"Incorrect pos shape {pos.shape}, should be (n,3)")
        if not (typ.ndim == 1):
            raise TypeError(f"Incorrect typ shape {typ.shape}, should be (n,)")

        return_dict = dict()

        # Identifying the types
        unique_typs = numpy.unique(typ)

        # Makeing the particle to grid for each type
        n_particles_out_of_grid = 0
        for uniqueTyp in unique_typs:
            pos_with_unique_typ = pos[numpy.equal(typ, uniqueTyp), :]
            results = self.particle_to_grid_continuous(pos_with_unique_typ)
            return_dict[uniqueTyp] = results["nParticlesIn_cells"]
            n_particles_out_of_grid += results["nParticlesOutOfGrid"]

        return_dict["nParticlesOutOfGrid"] = n_particles_out_of_grid

        return return_dict

    def _fi(self, d: numpy.ndarray) -> numpy.ndarray:
        w = self._particle_max_r
        lb = d > -w
        ub = d < w
        h = w - d
        y = (lb * ub) * (h * h * (3 * w - h)) / (4 * w * w * w) + (1 - lb) * 1
        return y

    def _helper(self, idx_of_cell_3v: numpy.ndarray, offset: numpy.ndarray, pos: numpy.ndarray) -> tuple:
        idx_of_cell_3v_offset = idx_of_cell_3v + offset

        # Creating a boolean array that marks the out of grid particles
        # No need to check out  of grid particles after offset because this
        particles_out_of_lower_bound = numpy.min(idx_of_cell_3v_offset, 1) < 0
        particles_out_of_upper_bound = numpy.any(numpy.greater_equal(idx_of_cell_3v_offset, self._grid_size), 1)
        particles_out_of_grid = numpy.logical_or(particles_out_of_lower_bound, particles_out_of_upper_bound)

        # Creating an array that contains the cell index of particles in i format, or if they are out of grid
        idx_of_cell = (
            idx_of_cell_3v_offset[:, 0]
            + idx_of_cell_3v_offset[:, 1] * self._grid_size[0]
            + idx_of_cell_3v_offset[:, 2] * self._grid_size[0] * self._grid_size[1]
        )
        idx_of_cell[particles_out_of_grid] = -1

        # Counting how many particles are in the cells by cell index occurance
        # For example if idx_of_cell = [1,0,0,-1,1,1,2,2] -> [-1,0,1,2], [1,2,3,2]
        unique_cell_idx, particle_counts_in_cells = numpy.unique(idx_of_cell, return_counts=True)

        # lb and ub of each particle that's why [:,i] and not just [i]
        lb = idx_of_cell_3v_offset * self._cell_length
        ub = (idx_of_cell_3v_offset + [1, 1, 1]) * self._cell_length
        fi = (
            (self._fi(pos[:, 0] - ub[:, 0]) - self._fi(pos[:, 0] - lb[:, 0]))
            * (self._fi(pos[:, 1] - ub[:, 1]) - self._fi(pos[:, 1] - lb[:, 1]))
            * (self._fi(pos[:, 2] - ub[:, 2]) - self._fi(pos[:, 2] - lb[:, 2]))
        )

        (unique_cell_idx, inv) = numpy.unique(idx_of_cell, return_inverse=True)
        particle_counts_in_cells_continuous = numpy.bincount(inv, fi.reshape(-1))

        return (unique_cell_idx, particle_counts_in_cells_continuous, idx_of_cell)


class ParticleNeighborList:
    """Class for particle neighbor list calculation."""

    def __init__(self, max_neighbor: int = 12, use_grid: bool = False, use_SKL: bool = True):
        self._max_neighbor = max_neighbor
        self._use_grid = use_grid
        self._use_SKL = use_SKL

    def _using_grid(self, pos: numpy.ndarray, rad: numpy.ndarray, trc: numpy.ndarray) -> numpy.ndarray:
        lower_bound = numpy.min(pos, axis=0)
        upper_bound = numpy.max(pos, axis=0)
        grid_size = numpy.maximum(numpy.floor((upper_bound - lower_bound) / (2.0 * numpy.max(rad))), numpy.ones((3,)))
        cell_length = numpy.divide(numpy.maximum((upper_bound - lower_bound), numpy.ones((3,))), grid_size)
        n_particles = pos.shape[0]
        ref_cell = numpy.floor((pos - lower_bound) / cell_length)
        particles_in_neighborhood = numpy.zeros((n_particles, 2))

        # for loop is half the time of numpy.apply_along_axis
        for idx_particle in range(n_particles):
            x = ref_cell[idx_particle, :]
            mask = (
                ((ref_cell[:, 0] == x[0] - 1) & (ref_cell[:, 1] == x[1] - 1) & (ref_cell[:, 2] == x[2] - 1))
                | ((ref_cell[:, 0] == x[0] - 1) & (ref_cell[:, 1] == x[1] - 1) & (ref_cell[:, 2] == x[2]))
                | ((ref_cell[:, 0] == x[0] - 1) & (ref_cell[:, 1] == x[1] - 1) & (ref_cell[:, 2] == x[2] + 1))
                | ((ref_cell[:, 0] == x[0] - 1) & (ref_cell[:, 1] == x[1]) & (ref_cell[:, 2] == x[2] - 1))
                | ((ref_cell[:, 0] == x[0] - 1) & (ref_cell[:, 1] == x[1]) & (ref_cell[:, 2] == x[2]))
                | ((ref_cell[:, 0] == x[0] - 1) & (ref_cell[:, 1] == x[1]) & (ref_cell[:, 2] == x[2] + 1))
                | ((ref_cell[:, 0] == x[0] - 1) & (ref_cell[:, 1] == x[1] + 1) & (ref_cell[:, 2] == x[2] - 1))
                | ((ref_cell[:, 0] == x[0] - 1) & (ref_cell[:, 1] == x[1] + 1) & (ref_cell[:, 2] == x[2]))
                | ((ref_cell[:, 0] == x[0] - 1) & (ref_cell[:, 1] == x[1] + 1) & (ref_cell[:, 2] == x[2] + 1))
                | ((ref_cell[:, 0] == x[0]) & (ref_cell[:, 1] == x[1] - 1) & (ref_cell[:, 2] == x[2] - 1))
                | ((ref_cell[:, 0] == x[0]) & (ref_cell[:, 1] == x[1] - 1) & (ref_cell[:, 2] == x[2]))
                | ((ref_cell[:, 0] == x[0]) & (ref_cell[:, 1] == x[1] - 1) & (ref_cell[:, 2] == x[2] + 1))
                | ((ref_cell[:, 0] == x[0]) & (ref_cell[:, 1] == x[1]) & (ref_cell[:, 2] == x[2] - 1))
                | ((ref_cell[:, 0] == x[0]) & (ref_cell[:, 1] == x[1]) & (ref_cell[:, 2] == x[2]))
                | ((ref_cell[:, 0] == x[0]) & (ref_cell[:, 1] == x[1]) & (ref_cell[:, 2] == x[2] + 1))
                | ((ref_cell[:, 0] == x[0]) & (ref_cell[:, 1] == x[1] + 1) & (ref_cell[:, 2] == x[2] - 1))
                | ((ref_cell[:, 0] == x[0]) & (ref_cell[:, 1] == x[1] + 1) & (ref_cell[:, 2] == x[2]))
                | ((ref_cell[:, 0] == x[0]) & (ref_cell[:, 1] == x[1] + 1) & (ref_cell[:, 2] == x[2] + 1))
                | ((ref_cell[:, 0] == x[0] + 1) & (ref_cell[:, 1] == x[1] - 1) & (ref_cell[:, 2] == x[2] - 1))
                | ((ref_cell[:, 0] == x[0] + 1) & (ref_cell[:, 1] == x[1] - 1) & (ref_cell[:, 2] == x[2]))
                | ((ref_cell[:, 0] == x[0] + 1) & (ref_cell[:, 1] == x[1] - 1) & (ref_cell[:, 2] == x[2] + 1))
                | ((ref_cell[:, 0] == x[0] + 1) & (ref_cell[:, 1] == x[1]) & (ref_cell[:, 2] == x[2] - 1))
                | ((ref_cell[:, 0] == x[0] + 1) & (ref_cell[:, 1] == x[1]) & (ref_cell[:, 2] == x[2]))
                | ((ref_cell[:, 0] == x[0] + 1) & (ref_cell[:, 1] == x[1]) & (ref_cell[:, 2] == x[2] + 1))
                | ((ref_cell[:, 0] == x[0] + 1) & (ref_cell[:, 1] == x[1] + 1) & (ref_cell[:, 2] == x[2] - 1))
                | ((ref_cell[:, 0] == x[0] + 1) & (ref_cell[:, 1] == x[1] + 1) & (ref_cell[:, 2] == x[2]))
                | ((ref_cell[:, 0] == x[0] + 1) & (ref_cell[:, 1] == x[1] + 1) & (ref_cell[:, 1] == x[1] + 1))
            )
            idx = numpy.arange(n_particles)[mask]
            distances = numpy.linalg.norm(numpy.subtract(pos[idx, :], pos[idx_particle, :]), axis=1)
            touching_distances = rad[idx] + rad[idx_particle]
            mask2 = (distances < touching_distances) & (distances > 0.0)
            idx2 = idx[mask2]
            particles_in_neighborhood[idx_particle, 0] = numpy.sum(trc[idx2])
            particles_in_neighborhood[idx_particle, 1] = numpy.sum(numpy.logical_not(trc)[idx2])

        return particles_in_neighborhood

    def _using_all(self, pos: numpy.ndarray, rad: numpy.ndarray, trc: numpy.ndarray) -> numpy.ndarray:
        nParticles = pos.shape[0]

        distances = sciDist.pdist(pos, "euclidean")
        distances[distances == 0] = numpy.inf
        touching_distances = rad + rad.transpose()
        mask = (distances < touching_distances).astype(int)
        particles_in_neighborhood = numpy.zeros((nParticles, 2))
        particles_in_neighborhood[:, 0] = mask.dot(trc)
        particles_in_neighborhood[:, 1] = mask.dot(numpy.logical_not(trc))

        return particles_in_neighborhood

    def _using_SKL(self, pos: numpy.ndarray, rad: numpy.ndarray, trc: numpy.ndarray) -> numpy.ndarray:
        knn = NearestNeighbors(n_neighbors=self._max_neighbor + 1, p=2, n_jobs=-1)
        knn.fit(pos)
        distances, indices = knn.kneighbors(pos, return_distance=True)
        touching_distances = rad[indices[:, 0].reshape((-1, 1))] + rad[indices[:, 1:]]
        distances = distances[:, 1:]
        indices = indices[:, 1:]
        n_particles = pos.shape[0]
        particles_in_neighborhood = numpy.zeros((n_particles, 2))
        particles_in_neighborhood[:, 0] = numpy.sum(
            numpy.multiply(trc[indices], distances < touching_distances), axis=1
        )
        particles_in_neighborhood[:, 1] = numpy.sum(
            numpy.multiply(numpy.logical_not(trc)[indices], distances < touching_distances), axis=1
        )

        return particles_in_neighborhood

    def calculate_neighbor_traced(self, pos: numpy.ndarray, rad: numpy.ndarray, trc: numpy.ndarray) -> numpy.ndarray:
        MixingIndex._check_shape_pos_and_trc(pos, trc)
        if not (rad.ndim == 1):
            raise TypeError(f"Incorrect rad shape {rad.shape}, should be (n,)")
        if not (pos.shape[0] == rad.shape[0]):
            raise TypeError(f"Incorrect pos shape {pos.shape} and rad shape {rad.shape}, should be (n,_) and (n,)")

        # SKL is the fastest
        # Just calculating all the distances is faster (probably because built in functions)
        # But with higher particle counts n_particle x n_particle matrices could run out of memory, then grid is better
        if self._use_SKL:
            particles_in_neighborhood = self._using_SKL(pos, rad, trc)
        elif self._use_grid:
            particles_in_neighborhood = self._using_grid(pos, rad, trc)
        else:
            particles_in_neighborhood = self._using_all(pos, rad, trc)

        return particles_in_neighborhood


class MixingIndex:
    """Base class for mixing indices."""

    def __init__(self):
        """Initialise the mixing index."""
        self._M = None
        self._iteration = 0
        self._runtime = 0.0

    def iterate(self):
        """Calculate the mixing index at one time step."""
        pass

    def get_M(self) -> numpy.ndarray:
        """Returns the normalised mixedness values of each iteration as a numpy.ndarray."""
        return numpy.squeeze(numpy.array(self._M))

    def get_runtime(self) -> float:
        return self._runtime

    def _check_shape_pos(pos: numpy.ndarray):
        shape = pos.shape
        if not ((pos.ndim == 2) and ((shape[1] == 3) or (shape[1] == 2) or (shape[1] == 1))):
            raise TypeError(f"Incorrect pos shape {shape}, should be (n,3) or (n,2) or (n,1)")

    def _check_shape_trc(trc: numpy.ndarray):
        if not (trc.ndim == 1):
            raise TypeError(f"Incorrect trc shape {trc.shape}, should be (n,)")

    def _check_shape_pos_and_trc(pos: numpy.ndarray, trc: numpy.ndarray):
        MixingIndex._check_shape_pos(pos)
        MixingIndex._check_shape_trc(trc)
        if not (pos.shape[0] == trc.shape[0]):
            raise TypeError(f"Incorrect pos shape {pos.shape} and trc shape {trc.shape}, should be (n,_) and (n,)")

    def _check_shape_PIS(particles_in_samples: numpy.ndarray):
        shape = particles_in_samples.shape
        if not ((particles_in_samples.ndim == 2) and (shape[1] == 2)):
            raise TypeError(
                f"Incorrect particles_in_samples shape {shape}, should be (n,2). Where [:,0] is traced, [:,1] is non-traced"
            )

    def _check_shape_PIN(particles_in_neighborhood: numpy.ndarray):
        shape = particles_in_neighborhood.shape
        if not ((particles_in_neighborhood.ndim == 2) and (shape[1] == 2)):
            raise TypeError(
                f"Incorrect particles_in_neighborhood shape {shape}, should be (n,2). Where [:,0] is traced, [:,1] is non-traced"
            )

    def _check_shape_PIN_and_trc(particles_in_neighborhood: numpy.ndarray, trc: numpy.ndarray):
        MixingIndex._check_shape_PIN(particles_in_neighborhood)
        MixingIndex._check_shape_trc(trc)
        if not (particles_in_neighborhood.shape[0] == trc.shape[0]):
            raise TypeError(
                "Incorrect particles_in_neighborhood shape {particles_in_neighborhood.shape} and trc shape {trc.shape}, should be (n,_) and (n,)"
            )


## Grid based methods


class DegreeOfMixedness(MixingIndex):

    def __init__(self):
        """Initialise the mixing index."""

        self._M = list()
        self._iteration = 0

    def _calculate(self, particles_in_samples: numpy.ndarray) -> float:
        npi = numpy.sum(particles_in_samples, 1)
        np = numpy.sum(npi)
        nai = particles_in_samples[:, 0]
        na = numpy.sum(nai)
        fai = numpy.divide(nai, npi)
        fa = na / np

        sigma_prime2 = numpy.mean(numpy.power(fai - fa, 2))
        sigma_seg2 = fa * (1 - fa)
        M = 1 - numpy.sqrt(sigma_prime2 / sigma_seg2)

        return M

    def iterate(self, particles_in_samples: numpy.ndarray) -> float:
        MixingIndex._check_shape_PIS(particles_in_samples)

        M = self._calculate(particles_in_samples)
        self._M.append(M)
        self._iteration += 1

        return M


class IntensityOfSegregation(MixingIndex):

    def __init__(self):
        """Initialise the mixing index."""

        self._M = list()
        self._iteration = 0

    def _calculate(self, particles_in_samples: numpy.ndarray) -> float:
        n_samples = particles_in_samples.shape[0]
        npi = numpy.sum(particles_in_samples, 1)
        np = numpy.sum(npi)
        nai = particles_in_samples[:, 0]
        na = numpy.sum(nai)
        fai = numpy.divide(nai, npi)
        fa = na / np

        sigma = numpy.sqrt(numpy.sum(numpy.power(fai - fa, 2.0)) / (n_samples - 1))
        sigma_seg = numpy.sqrt(fa * (1 - fa))

        M = sigma / sigma_seg

        return M

    def iterate(self, particles_in_samples: numpy.ndarray) -> float:
        MixingIndex._check_shape_PIS(particles_in_samples)

        M = self._calculate(particles_in_samples)
        self._M.append(M)
        self._iteration += 1

        return M


class LaceyIndex(MixingIndex):

    def __init__(self):
        """Initialise the mixing index."""

        self._M = list()
        self._iteration = 0

    def _calculate(self, particles_in_samples: numpy.ndarray) -> float:
        n_samples = particles_in_samples.shape[0]
        npi = numpy.sum(particles_in_samples, 1)
        np = numpy.sum(npi)
        npps = np / n_samples
        nai = particles_in_samples[:, 0]
        na = numpy.sum(nai)
        fai = numpy.divide(nai, npi)
        fa = na / np

        sigma_2 = numpy.sum(numpy.power(fai - fa, 2.0)) / (n_samples - 1)
        sigma_seg2 = fa * (1 - fa)
        sigma_mix2 = (fa * (1 - fa)) / npps

        # if only one particle per cell: npps = 1 would result division by zero
        if sigma_mix2 == sigma_seg2:
            M = 0
        else:
            M = (sigma_2 - sigma_seg2) / (sigma_mix2 - sigma_seg2)

        return M

    def iterate(self, particles_in_samples: numpy.ndarray) -> float:
        MixingIndex._check_shape_PIS(particles_in_samples)

        M = self._calculate(particles_in_samples)
        self._M.append(M)
        self._iteration += 1

        return M


class MixingEntropy(MixingIndex):

    def __init__(
        self,
        particles_in_samples_segregated: numpy.ndarray = None,
        particles_in_samples_mixed: numpy.ndarray = None,
    ):
        """Initialise the mixing index."""

        self._M = list()
        self._iteration = 0
        self._runtime = 0.0
        if particles_in_samples_segregated is not None:
            MixingIndex._check_shape_PIS(particles_in_samples_segregated)
            self._S_seg = self._calculate(particles_in_samples_segregated)
        else:
            self._S_seg = None
        if particles_in_samples_mixed is not None:
            MixingIndex._check_shape_PIS(particles_in_samples_mixed)
            self._S_mix = self._calculate(particles_in_samples_mixed)
        else:
            self._S_mix = None

    def _calculate(self, particles_in_samples: numpy.ndarray) -> float:
        # To suppress division by zero warning, those results will be filtered out later anyway
        warnings.filterwarnings("ignore")

        n_samples = particles_in_samples.shape[0]
        npi = numpy.sum(particles_in_samples, 1)
        np = numpy.sum(npi)
        nai = particles_in_samples[:, 0]
        na = numpy.sum(nai)
        nbi = particles_in_samples[:, 1]
        nb = numpy.sum(nbi)
        fai = numpy.divide(nai, npi)
        fa = na / np
        fbi = numpy.divide(nbi, npi)
        fb = nb / np

        si = numpy.multiply(fai, numpy.log(fai)) + numpy.multiply(fbi, numpy.log(fbi))
        si[numpy.logical_or((nai == 0), (nbi == 0))] = 0.0
        S = numpy.mean(numpy.multiply(si, npi))
        S_seg = 0.0 if self._S_seg is None else self._S_seg
        S_mix = (fa * numpy.log(fa) + fb * numpy.log(fb)) * np / n_samples if self._S_mix is None else self._S_mix
        M = (S - S_seg) / (S_mix - S_seg)

        warnings.filterwarnings("default")

        return M

    def iterate(self, particles_in_samples: numpy.ndarray) -> float:
        MixingIndex._check_shape_PIS(particles_in_samples)

        M = self._calculate(particles_in_samples)
        self._M.append(M)
        self._iteration += 1

        return M


class StandardDeviation(MixingIndex):

    def __init__(self):
        """Initialise the mixing index."""

        self._M = list()
        self._iteration = 0

    def _calculate(self, particles_in_samples: numpy.ndarray) -> float:
        n_samples = particles_in_samples.shape[0]
        npi = numpy.sum(particles_in_samples, 1)
        np = numpy.sum(npi)
        nai = particles_in_samples[:, 0]
        na = numpy.sum(nai)
        fai = numpy.divide(nai, npi)
        fa = na / np

        sigma = numpy.sqrt(numpy.sum(numpy.power(fai - fa, 2.0)) / (n_samples - 1))
        M = sigma

        return M

    def iterate(self, particles_in_samples: numpy.ndarray) -> float:
        MixingIndex._check_shape_PIS(particles_in_samples)

        M = self._calculate(particles_in_samples)
        self._M.append(M)
        self._iteration += 1

        return M

    def get_M(self) -> numpy.ndarray:
        return numpy.squeeze(numpy.array(self._M))


class RelativeStandardDeviation(MixingIndex):

    def __init__(self):
        """Initialise the mixing index."""

        self._M = list()
        self._iteration = 0

    def _calculate(self, particles_in_samples: numpy.ndarray) -> float:
        n_samples = particles_in_samples.shape[0]
        npi = numpy.sum(particles_in_samples, 1)
        np = numpy.sum(npi)
        nai = particles_in_samples[:, 0]
        na = numpy.sum(nai)
        fai = numpy.divide(nai, npi)
        fa = na / np

        sigma = numpy.sqrt(numpy.sum(numpy.power(fai - fa, 2.0)) / (n_samples - 1))

        M = 1.0 - sigma / fa

        return M

    def iterate(self, particles_in_samples: numpy.ndarray) -> float:
        MixingIndex._check_shape_PIS(particles_in_samples)

        M = self._calculate(particles_in_samples)
        self._M.append(M)
        self._iteration += 1

        return M


class VarianceAmongBimodalBinCounts(MixingIndex):

    def __init__(self, weighted: bool = True, weight: float = None, index: bool = False):
        """Initialise the mixing index."""

        self._M = list()
        self._M0 = None
        self._iteration = 0
        self._weighted = weighted
        self._weight = weight
        self._index = index

    def _calculate(self, particles_in_samples: numpy.ndarray) -> float:
        n_samples = particles_in_samples.shape[0]
        nai = particles_in_samples[:, 0]
        na = numpy.sum(nai)
        nbi = particles_in_samples[:, 1]
        nb = numpy.sum(nbi)

        gamma = (na / nb) if self._weight is None else self._weight

        if self._weighted:
            sigma_bi2 = numpy.mean(numpy.power(nai - gamma * nbi, 2.0))
        else:
            sigma_bi2 = numpy.mean(numpy.power((nai - nbi) - (na - nb) / n_samples, 2.0))

        if self._index:
            # sigma_biseg2 = (na**2.0 + nb**2.0) / n_samples - ((na - nb) / n_samples) ** 2.0
            sigma_biseg2 = (na / nb) * ((na - nb) / n_samples) ** 2.0
            M = 1.0 - sigma_bi2 / sigma_biseg2
            return M
        else:
            return sigma_bi2

    def iterate(self, particles_in_samples: numpy.ndarray) -> float:
        MixingIndex._check_shape_PIS(particles_in_samples)

        M = self._calculate(particles_in_samples)
        if not self._index:
            if self._iteration == 0:
                self._M0 = M
            M = 1.0 - M / self._M0
        self._M.append(M)
        self._iteration += 1

        return M


class VarianceReductionRatio(MixingIndex):

    def __init__(self, fa: float = None):
        """Initialise the mixing index."""

        self._M = list()
        self._iteration = 0

    def _calculate(self, particles_in_samples: numpy.ndarray) -> float:
        n_samples = particles_in_samples.shape[0]
        npi = numpy.sum(particles_in_samples, 1)
        np = numpy.sum(npi)
        nai = particles_in_samples[:, 0]
        na = numpy.sum(nai)
        fai = numpy.divide(nai, npi)
        fa = na / np

        sigma_2 = numpy.sum(numpy.power(fai - fa, 2.0)) / (n_samples - 1)

        return sigma_2

    def iterate(self, particles_in_samples_in: numpy.ndarray, particles_in_samples_out: numpy.ndarray) -> float:
        MixingIndex._check_shape_PIS(particles_in_samples_in)
        MixingIndex._check_shape_PIS(particles_in_samples_out)

        sigma_in2 = self._calculate(particles_in_samples_in)
        sigma_out2 = self._calculate(particles_in_samples_out)
        M = 1 - (sigma_in2 / sigma_out2)
        self._M.append(M)
        self._iteration += 1

        return M


## Contact based methods


class CoordinationNumber(MixingIndex):

    def __init__(self):
        """Initialise the mixing index."""

        self._M = list()
        self._iteration = 0

    def _calculate(self, particles_in_neighborhood: numpy.ndarray, trc: numpy.ndarray) -> float:
        # traced-traced, counted from both direction, needs to halfed
        Caa = numpy.sum(particles_in_neighborhood[trc, 0]) / 2
        # nontraced-nontraced, counted from both direction, needs to halfed
        Cbb = numpy.sum(particles_in_neighborhood[numpy.logical_not(trc), 1]) / 2
        # traced-nontraced
        Cab = numpy.sum(particles_in_neighborhood[trc, 1])
        M = Cab / (Caa + Cbb)

        return M

    def iterate(self, particles_in_neighborhood: numpy.ndarray, trc: numpy.ndarray) -> float:
        MixingIndex._check_shape_PIN_and_trc(particles_in_neighborhood, trc)

        M = self._calculate(particles_in_neighborhood, trc)
        self._M.append(M)
        self._iteration += 1

        return M


class ParticleScaleIndex(MixingIndex):
    """Particle scale index mixing index."""

    def __init__(self, n_sample_size: int = 7):
        """Initialise the mixing index."""

        self._M = list()
        self._iteration = 0
        self._n_sample_size = n_sample_size

    def _calculate(self, particles_in_neighborhood: numpy.ndarray, trc: numpy.ndarray) -> float:
        Cj = numpy.sum(particles_in_neighborhood, 1)
        pj = numpy.zeros((trc.shape))
        # traced with nontraced
        pj[trc] = particles_in_neighborhood[trc, 1] / (Cj[trc] + 1)
        # nontraced with nontraced
        pj[numpy.logical_not(trc)] = (particles_in_neighborhood[numpy.logical_not(trc), 1] + 1) / (
            Cj[numpy.logical_not(trc)] + 1
        )
        pAverage = numpy.mean(pj)
        ca = numpy.mean(trc)
        Sts = numpy.mean(numpy.power(pj - pAverage, 2))
        S0s = ca * (1 - ca)
        SRs = ca * (1 - ca) / self._n_sample_size
        M = (Sts - S0s) / (SRs - S0s)

        return M

    def iterate(self, particles_in_neighborhood: numpy.ndarray, trc: numpy.ndarray) -> float:
        MixingIndex._check_shape_PIN_and_trc(particles_in_neighborhood, trc)

        M = self._calculate(particles_in_neighborhood, trc)
        self._M.append(M)
        self._iteration += 1

        return M


class SegregationIndex(MixingIndex):
    """Segregation index mixing index."""

    def __init__(self):
        """Initialise the mixing index."""

        self._M = list()
        self._iteration = 0

    def _calculate(self, particles_in_neighborhood: numpy.ndarray, trc: numpy.ndarray) -> float:
        # traced-traced, works better without halfing
        Caa = numpy.sum(particles_in_neighborhood[trc, 0])
        # nontraced-nontraced, works better without halfing
        Cbb = numpy.sum(particles_in_neighborhood[numpy.logical_not(trc), 1])
        # traced-nontraced
        Cab = numpy.sum(particles_in_neighborhood[trc, 1])
        M = Caa / (Caa + Cab) + Cbb / (Cbb + Cab)
        M = 2 - M

        return M

    def iterate(self, particles_in_neighborhood: numpy.ndarray, trc: numpy.ndarray) -> float:
        MixingIndex._check_shape_PIN_and_trc(particles_in_neighborhood, trc)

        M = self._calculate(particles_in_neighborhood, trc)
        self._M.append(M)
        self._iteration += 1

        return M


## Distance based methods


class AverageHeight(MixingIndex):
    """Average height mixing index."""

    def __init__(self, normalisation: bool = True, axis: tuple = 2):
        """Initialise the mixing index."""
        self._M = list()
        self._iteration = 0
        self._M0 = 0
        self._normalisation = normalisation
        self._axis = axis

    def _calculate(self, pos: numpy.ndarray, trc: numpy.ndarray) -> float:
        COM_traced = numpy.mean(pos[trc, self._axis])
        COM_all = numpy.mean(pos[:, self._axis])
        M = COM_traced / COM_all

        if not self._normalisation:
            return M

        if self._iteration == 0:
            self._M0 = M
        # M = 2.0 * (M - 0.5)
        M = (M - self._M0) / (1.0 - self._M0)

        return M

    def iterate(self, pos: numpy.ndarray, trc: numpy.ndarray) -> float:
        MixingIndex._check_shape_pos_and_trc(pos, trc)

        M = self._calculate(pos, trc)
        self._M.append(M)
        self._iteration += 1

        return M


class NeighborDistanceMethod(MixingIndex):
    """Neighbor distance method mixing index."""

    def __init__(self, particle_diameter: float = 0.0):
        """Initialise the mixing index."""
        self._M = list()
        self._iteration = 0
        self._d_particle = particle_diameter
        self._idx_neighbour = None
        self._idx_random = None

    def _calculate(self, pos: numpy.ndarray, trc: numpy.ndarray = None) -> float:
        if self._iteration == 0:
            # After a few thousand particle SKL is faster
            # n_neighbors=2: since we fit and feed back again the particles 1 would give back the particle itself
            # p=2: Minkowski metric euclidean_distance (l2) for p=2
            # n_jobs=-1: means using all processors
            knn = NearestNeighbors(n_neighbors=2, p=2, n_jobs=-1)
            knn.fit(pos)
            my_random = numpy.random.default_rng(0)
            if trc is None:
                indices = knn.kneighbors(pos, return_distance=False)
                self._idx_neighbour = indices[:, 1]
                self._idx_random = my_random.permutation(pos.shape[0])
            else:
                posa = pos[trc, :]
                indices = knn.kneighbors(posa, return_distance=False)
                self._idx_neighbour = indices[:, 1]
                self._idx_random = my_random.permutation(posa.shape[0])

        if trc is None:
            mn = numpy.mean(numpy.linalg.norm(pos[:, :] - pos[self._idx_neighbour, :], axis=1) - self._d_particle)
            mr = numpy.mean(numpy.linalg.norm(pos[:, :] - pos[self._idx_random, :], axis=1) - self._d_particle)
        else:
            posa = pos[trc, :]
            mn = numpy.mean(numpy.linalg.norm(posa[:, :] - pos[self._idx_neighbour, :], axis=1) - self._d_particle)
            mr = numpy.mean(numpy.linalg.norm(posa[:, :] - pos[self._idx_random, :], axis=1) - self._d_particle)
        M = mn / mr

        return M

    def iterate(self, pos: numpy.ndarray, trc: numpy.ndarray = None) -> float:
        MixingIndex._check_shape_pos(pos)

        if trc is None:
            M = self._calculate(pos)
        else:
            M = self._calculate(pos, trc)
        self._M.append(M)
        self._iteration += 1

        return M


class NearestNeighborMethod(MixingIndex):

    def __init__(self, closest_n: int = 12):
        """Initialise the mixing index."""
        self._M = list()
        self._iteration = 0
        self._closest_n = closest_n

    def _calculate(self, pos: numpy.ndarray, trc: numpy.ndarray) -> float:
        # After a few thousand particle SKL is faster
        knn = NearestNeighbors(n_neighbors=self._closest_n + 1, p=2, n_jobs=-1)
        knn.fit(pos)
        indices = knn.kneighbors(pos, return_distance=False)
        idx_closest = indices[:, 1 : self._closest_n + 1]
        trc_repeated = numpy.broadcast_to(trc, (self._closest_n, trc.size)).transpose()
        n_same = numpy.sum(trc[idx_closest] != trc_repeated, axis=1)
        M = numpy.mean(2 * n_same / self._closest_n)

        return M

    def iterate(self, pos: numpy.ndarray, trc: numpy.ndarray) -> float:
        MixingIndex._check_shape_pos_and_trc(pos, trc)

        M = self._calculate(pos, trc)
        self._M.append(M)
        self._iteration += 1

        return M


class SiiriaMethod(MixingIndex):
    """Siiria method mixing index."""

    def __init__(self, R: float = 0.87, g: float = 10.0 / numpy.sqrt(2.0), particle_max_r: int = 100000):
        """Initialise the mixing index."""
        self._M = list()
        self._iteration = 0
        self._runtime = 0.0
        self._r0 = None
        self._pos0 = None
        self._R = R
        self._g = g
        # Avoiding calculating index above high particle counts
        self._particle_max_r = particle_max_r

    def _calculate(self, pos: numpy.ndarray) -> float:
        # Calculating index with n particles requires n^2 memory
        # To avoid running out of memory switching to a slower but less memory heavy option
        if pos.shape[0] > self._particle_max_r:
            if self._iteration == 0:
                self._pos0 = pos

            M = 0.0
            for idx in range(pos.shape[0]):
                r0j = sciDist.cdist(self._pos0[idx, :].reshape((1, 3)), self._pos0, "euclidean")
                rj = sciDist.cdist(pos[idx, :].reshape((1, 3)), pos, "euclidean")
                M += numpy.sum(
                    numpy.power(self._R, self._g * numpy.abs(r0j))
                    * (1 - numpy.power(self._R, self._g * numpy.abs(r0j - rj)))
                )
            M /= pos.shape[0] * pos.shape[0]
        else:
            r = sciDist.pdist(pos, "euclidean")

            if self._iteration == 0:
                self._r0 = r

            M = numpy.mean(
                numpy.power(self._R, self._g * numpy.abs(self._r0))
                * (1 - numpy.power(self._R, self._g * numpy.abs(self._r0 - r)))
            )

        return M

    def iterate(self, pos: numpy.ndarray) -> float:
        time_start = time.time()
        MixingIndex._check_shape_pos(pos)

        M = self._calculate(pos)
        self._M.append(M)
        self._iteration += 1

        self._runtime += time.time() - time_start
        return M

    def get_M(self) -> numpy.ndarray:
        return numpy.squeeze(numpy.array(self._M))


class SphereSpreadingMethod(MixingIndex):

    def __init__(self):
        """Initialise the mixing index."""
        self._M = list()
        self._iteration = 0
        self._ma0 = 0

    def _calculate(self, pos: numpy.ndarray, trc: numpy.ndarray) -> float:
        posa = pos[trc, :]
        coma = numpy.mean(posa, 0)
        distance = numpy.linalg.norm(posa - coma, axis=1)
        R = numpy.mean(distance)

        return R

    def iterate(self, pos: numpy.ndarray, trc: numpy.ndarray) -> float:
        MixingIndex._check_shape_pos_and_trc(pos, trc)

        m_a = self._calculate(pos, trc)
        if self._iteration == 0:
            self._ma0 = m_a
        m = self._calculate(pos, numpy.full(trc.shape, True))
        M = (m_a - self._ma0) / (m - self._ma0)
        self._M.append(M)
        self._iteration += 1

        return M


## Continuoum based methods


class NewIndex(MixingIndex):
    """New index."""

    def __init__(self):
        """Initialise the mixing index."""

        self._M = list()
        self._iteration = 0
        self._runtime = 0.0

    def _calculate(self, cai: numpy.ndarray, ca: float, Nsample: int) -> float:
        sigmas = numpy.sum(numpy.divide(numpy.power(cai - ca, 2), Nsample - 1))
        sigma0s = ca * (1 - ca)
        if (sigma0s) == 0.0:
            return -1.0
        M = (sigmas - sigma0s) / (-sigma0s)

        return M

    def iterate(self, cai: numpy.ndarray, ca: float, Nsample: int) -> float:
        time_start = time.time()
        M = self._calculate(cai, ca, Nsample)
        self._M.append(M)
        self._iteration += 1

        self._runtime += time.time() - time_start
        return M

    def get_M(self) -> numpy.ndarray:
        return numpy.squeeze(numpy.array(self._M))
