'''Python implementation of the PARS algorithm.'''

import numpy as np
from scipy.spatial import cKDTree # pylint: disable=E0611
from scipy.interpolate import griddata

def pars(kx, ky, k, sens, kernel_size=25, lamda=0.01):
    '''Parallel MRI with adaptive radius in k‐space.

    Parameters
    ----------

    Returns
    -------

    Notes
    -----

    References
    ----------
    .. [1]
    '''

    kxy = np.concatenate((kx[:, None], ky[:, None]), axis=-1)

    # We want to resample onto a Cartesian grid
    sx, sy, nc = sens.shape[:]
    tx, ty = np.meshgrid(
        np.linspace(np.min(kx), np.max(kx), sx),
        np.linspace(np.min(ky), np.max(ky), sy))
    tx, ty = tx.flatten(), ty.flatten()
    txy = np.concatenate((tx[:, None], ty[:, None]), axis=-1)

    kdtree = cKDTree(kxy)
    _, idx = kdtree.query(txy, k=kernel_size+1)
    idx = idx[..., 1:].squeeze()

    # Need to figure out the encoding matrix formalism so we can
    # map to coil sensitivities in the image domain...

    # # S = k[idx, :].reshape((idx.shape[0], -1))
    # S = griddata((tx, ty), sens.reshape((-1, nc)), (kx, ky))
    # S = S[idx, :].reshape((idx.shape[0], -1))
    # T = sens.reshape((idx.shape[0], -1))
    #
    # ShS = S.conj().T @ S
    # ShT = S.conj().T @ T
    # lamda0 = lamda*np.linalg.norm(ShS)/ShS.shape[0]
    # W = np.linalg.solve(ShS + lamda0*np.eye(ShS.shape[0]), ShT)
    #
    # # Now do the regridding
    # S = k[idx, :].reshape((idx.shape[0], -1))
    # return (S @ W).reshape((sx, sy, nc))
