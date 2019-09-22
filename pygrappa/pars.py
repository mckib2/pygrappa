'''Python implementation of the PARS algorithm.'''

import numpy as np
from scipy.spatial import cKDTree # pylint: disable=E0611
from tqdm import tqdm

def pars(
        kx, ky, k, sens, tx=None, ty=None, kernel_size=25,
        coil_axis=-1):
    '''Parallel MRI with adaptive radius in k‐space.

    Parameters
    ----------
    kx, ky : array_like
        Sample points in kspace corresponding to measurements k.
        kx, kx are 1D arrays.
    k : array_like
        Complex kspace coil measurements corresponding to points
        (kx, ky).
    sens : array_like
        Coil sensitivity maps with shape of desired reconstruction.
    tx, ty : array_like
        Sample points in kspace defining the grid of ifft2(sens).
        If None, then tx, ty will be generated from a meshgrid with
        endpoints [min(kx), max(kx), min(ky), max(ky)].
    kernel_size : int, optional
        Number of nearest neighbors to use when interpolating kspace.
    coil_axis : int, optional
        Dimension holding coil data.

    Returns
    -------
    res : array_like
        Reconstructed kspace on a Cartesian grid with the same shape
        as sens.

    Notes
    -----

    References
    ----------
    .. [1] Yeh, Ernest N., et al. "3Parallel magnetic resonance
           imaging with adaptive radius in k‐space (PARS):
           Constrained image reconstruction using k‐space locality in
           radiofrequency coil encoded data." Magnetic Resonance in
           Medicine: An Official Journal of the International Society
           for Magnetic Resonance in Medicine 53.6 (2005): 1383-1392.
    '''

    # Move coil axis to the back
    k = np.moveaxis(k, coil_axis, -1)
    sens = np.moveaxis(sens, coil_axis, -1)

    #  Make (kx, ky) easier to do algebra with
    kxy = np.concatenate((kx[:, None], ky[:, None]), axis=-1)

    # We want to resample onto a Cartesian grid
    sx, sy, nc = sens.shape[:]
    if tx is None or ty is None:
        tx, ty = np.meshgrid(
            np.linspace(np.min(kx), np.max(kx), sx),
            np.linspace(np.min(ky), np.max(ky), sy))
        tx, ty = tx.flatten(), ty.flatten()
    txy = np.concatenate((tx[:, None], ty[:, None]), axis=-1)

    # Make a kd-tree and find all point around targets
    kdtree = cKDTree(kxy)
    _, idx = kdtree.query(txy, k=kernel_size)

    # Encoding matrix is much too large to invert, so we'll go
    # kernel by kernel to grid/reconstruct kspace
    sens = np.reshape(sens, (-1, nc))
    res = np.zeros(sens.shape, dtype=sens.dtype)
    for ii, idx0 in tqdm(
            enumerate(idx), total=idx.shape[0], leave=False,
            desc='PARS'):

        # Scale kspace coordinates to be within [-.5, .5]
        dk = np.reshape(kxy[idx0, :]/(sx*2), (-1, 2))
        r = txy[ii, :]/(sx*2)

        # Create local encoding matrix and train weights
        E = np.exp(2*np.pi*1j*(-dk @ r))
        E = E[:, None]*sens[ii, :]
        W = sens[ii, :] @ E.conj().T @ np.linalg.pinv(E @ E.conj().T)

        # Grid the sample:
        res[ii, :] = W @ k[idx0, :]
    return res.reshape((sx, sy, nc))
