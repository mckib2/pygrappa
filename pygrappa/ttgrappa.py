'''Python implementation of through-time GRAPPA.'''

from time import time

import numpy as np
from scipy.spatial import cKDTree # pylint: disable=E0611

def ttgrappa(
        kx, ky, kspace, cx, cy, calib, kernel_size=25, coil_axis=-1,
        time_axis=-2, lamda=0.01):
    '''Through-time GRAPPA.

    Parameters
    ----------
    kx, ky: array_like
        k-space coordinates of kspace data, kspace.  kx and ky are 1D
        arrays.
    kspace : array_like
        Complex kspace data corresponding to the measurements at
        locations kx, ky.  kspace has two dimensions: data and coil.
        Unsampled points should be exactly 0.
    cx, cy: array_like
        k-space coordinates of calibration kspace data.  cx and cy
        are 1D arrays.
    calib : array_like
        Complex kspace data corresponding to the measurements at
        locations cx, cy.  calib has three dimensions: data, time,
        and coil.
    kernel_size : int, optional
        Number of points to use as sources for kernel training.  This
        many nearest neighbors to the targets will be chosen.
    coil_axis : int, optional
        Dimension of kspace and calib holding coil data.
    time_axis : int, optional
        Dimension of calib holding time data.
    lamda : float, optional
        Tikhonov regularization for the kernel calibration.

    Returns
    -------
    res : array_like
        The reconstructed measurements with the same size as kspace.

    Notes
    -----
    Implements the through-time GRAPPA algorithm for non-Cartesian
    reconstruction as described in [1]_.

    This implementation uses a kd-tree for kernel selection similar
    to [2]_.  This simplifies searches for kernel geometries and
    helps make this implementation trajectory agnostic.

    References
    ----------
    .. [1] Seiberlich, Nicole, et al. "Improved radial GRAPPA
           calibration for real‐time free‐breathing cardiac imaging."
           Magnetic resonance in medicine 65.2 (2011): 492-505.
    .. [2] Luo, Tianrui, et al. "A GRAPPA algorithm for arbitrary
           2D/3D non‐Cartesian sampling trajectories with rapid
           calibration." Magnetic resonance in medicine 82.3 (2019):
           1101-1112.
    '''

    # Move da coils to da back and time_axis to the middle
    kspace = np.moveaxis(kspace, coil_axis, -1)
    calib = np.moveaxis(calib, (coil_axis, time_axis), (-1, -2))
    _sx, nt, _nc = calib.shape[:]

    # Find target locations
    mask = np.abs(kspace[..., 0]) > 0
    sampled = np.argwhere(mask).squeeze()
    holes = np.argwhere(~mask).squeeze()

    # Find nearest neighbors using a kd-tree
    kxy = np.concatenate((kx[:, None], ky[:, None]), axis=-1)
    cxy = np.concatenate((cx[:, None], cy[:, None]), axis=-1)
    t0 = time()
    kdtree = cKDTree(cxy)
    _, idx = kdtree.query(kxy[holes, :], k=kernel_size+1)
    idx = idx[..., 1:].squeeze() # first will always be the target
    print('Took %g seconds to find neighbors' % (time() - t0))

    # Get targets, sources, and weights.  Make sure to collapse the
    # through-time dimension!
    t0 = time()
    S = calib[idx, ...].reshape((idx.shape[0]*nt, -1))
    T = calib[holes, ...].reshape((holes.shape[0]*nt, -1))

    ShS = S.conj().T @ S
    ShT = S.conj().T @ T
    lamda0 = lamda*np.linalg.norm(ShS)/ShS.shape[0]
    W = np.linalg.solve(ShS + lamda0*np.eye(ShS.shape[0]), ShT)
    print('Took %g seconds to train weights' % (time() - t0))

    # Apply the weights to get reconstructed kspace
    t0 = time()
    S = kspace[idx, :].reshape((idx.shape[0], -1))
    res = np.zeros(kspace.shape, dtype=kspace.dtype)
    res[holes, :] = S @ W
    res[sampled, :] = kspace[sampled, :]
    print('Took %g seconds to apply weights' % (time()- t0))

    return np.moveaxis(res, -1, coil_axis)
