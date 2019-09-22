'''Python implementation of the PARS algorithm.'''

from time import time

import numpy as np
from scipy.spatial import cKDTree # pylint: disable=E0611
from scipy.ndimage import zoom
from tqdm import tqdm

def pars(
        kx, ky, k, sens, tx=None, ty=None, kernel_size=25,
        kernel_radius=None, coil_axis=-1):
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
    kernel_radius : float, optional
        Raidus in kspace (units same as (kx, ky)) to select neighbors
        when training kernels.
    coil_axis : int, optional
        Dimension holding coil data.

    Returns
    -------
    res : array_like
        Reconstructed image space on a Cartesian grid with the same
        shape as sens.

    Notes
    -----
    Implements the algorithm described in [1]_.

    Using kernel_radius seems to perform better than kernel_size.

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
    kxy = np.concatenate((kx[:, None], ky[:, None]), axis=-1)

    # Oversample the sensitivity maps by a factor of 2
    t0 = time()
    sensr = zoom(sens.real, (2, 2, 1), order=1)
    sensi = zoom(sens.imag, (2, 2, 1), order=1)
    sens = sensr + 1j*sensi
    print('Took %g seconds to upsample sens' % (time() - t0))

    # We want to resample onto a Cartesian grid
    sx, sy, nc = sens.shape[:]
    if tx is None or ty is None:
        tx, ty = np.meshgrid(
            np.linspace(np.min(kx), np.max(kx), sx),
            np.linspace(np.min(ky), np.max(ky), sy))
        tx, ty = tx.flatten(), ty.flatten()
    txy = np.concatenate((tx[:, None], ty[:, None]), axis=-1)

    # Make a kd-tree and find all point around targets
    t0 = time()
    kdtree = cKDTree(kxy)
    if kernel_radius is None:
        _, idx = kdtree.query(txy, k=kernel_size)
    else:
        idx = kdtree.query_ball_point(txy, r=kernel_radius)
    print('Took %g seconds to find nearest neighbors' % (time() - t0))

    # Scale kspace coordinates to be within [-.5, .5]
    kxy0 = np.concatenate(
        (kx[:, None]/np.max(kx), ky[:, None]/np.max(ky)), axis=-1)/2
    txy0 = np.concatenate(
        (tx[:, None]/np.max(tx), ty[:, None]/np.max(ty)), axis=-1)/2

    # Encoding matrix is much too large to invert, so we'll go
    # kernel by kernel to grid/reconstruct kspace
    sens = np.reshape(sens, (-1, nc))
    res = np.zeros(sens.shape, dtype=sens.dtype)
    t0 = time()
    for ii, idx0 in tqdm(
            enumerate(idx), total=idx.shape[0], leave=False,
            desc='PARS'):

        dk = kxy0[idx0, :]
        r = txy0[ii, :]

        # Create local encoding matrix and train weights
        E = np.exp(1j*(-dk @ r))
        E = E[:, None]*sens[ii, :]
        W = sens[ii, :] @ E.conj().T @ np.linalg.pinv(E @ E.conj().T)

        # Grid the sample:
        res[ii, :] = W @ k[idx0, :]

    print('Took %g seconds to regrid' % (time() - t0))

    # Return image at correct resolution with coil axis in right place
    ax = (0, 1)
    sx4 = int(sx/4)
    return np.moveaxis(np.fft.fftshift(np.fft.ifft2(
        np.fft.ifftshift(np.reshape(res, (sx, sy, nc), 'F'), axes=ax),
        axes=ax), axes=ax)[sx4:-sx4, sx4:-sx4, :], -1, coil_axis)
