'''Python implementation of Non-Cartesian GRAPPA.'''

import numpy as np
from scipy.spatial import cKDTree # pylint: disable=E0611

def ncgrappa(kx, ky, k, calib, kernel_size, coil_axis=-1):
    '''Non-Cartesian GRAPPA.

    Parameters
    ----------
    kx, ky : array_like
        k-space coordinates of kspace data, k.  kx and ky are 1D
        arrays.
    k : array_like
        Complex kspace data corresponding the measurements at
        locations kx, ky.  k has two dimensions: data and coil.  The
        coil dimension will be assumed to be last unless coil_axis=0.
        Unsampled points should be exactly 0.
    calib : array_like
        Cartesian calibration data, usually the fully sampled center
        of kspace.
    kernel_size : float
        Radius of kernel.
    coil_axis : int, optional
        Dimension of calib holding coil data.

    Notes
    -----
    Implements to the algorithm described in [1]_.

    References
    ----------
    .. [1] Luo, Tianrui, et al. "A GRAPPA algorithm for arbitrary
           2D/3D non‐Cartesian sampling trajectories with rapid
           calibration." Magnetic resonance in medicine 82.3 (2019):
           1101-1112.
    '''

    # Assume k has coil at end unless user says it's upfront.  We
    # want to assume that coils are in the back of calib and k:
    calib = np.moveaxis(calib, coil_axis, -1)
    if coil_axis == 0:
        k = np.moveaxis(k, coil_axis, -1)

    # Find all unsampled points
    idx = np.argwhere(np.abs(k[:, 0]) == 0)
    print(idx)

    # Identify all the constellations for calibration
    kxy = np.concatenate((kx[None, :], ky[None, :]), axis=0)
    kdtree = cKDTree(kxy)
    print(kdtree)

    # For each un‐sampled k‐space point, query the kd‐tree with the
    # prescribed distance (i.e., GRAPPA kernel size)
    # constellations = kdtree.query_ball_point(
    #     kxy[idx, ...], r=kernel_size)
