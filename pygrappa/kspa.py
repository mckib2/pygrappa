'''Python implementation of the kSPA algorithm.'''

from time import time

import numpy as np
from scipy.spatial import cKDTree  # pylint: disable=E0611
from scipy.sparse import lil_matrix
from scipy.interpolate import griddata


def kspa(
        kx, ky, k, sens, coil_axis=-1, sens_coil_axis=-1):
    '''Recon for arbitrary trajectories using k‐space sparse matrices.

    Parameters
    ----------

    Returns
    -------

    Notes
    -----

    References
    ----------
    .. [1] Liu, Chunlei, Roland Bammer, and Michael E. Moseley.
           "Parallel imaging reconstruction for arbitrary
           trajectories using k‐space sparse matrices (kSPA)."
           Magnetic Resonance in Medicine: An Official Journal of the
           International Society for Magnetic Resonance in Medicine
           58.6 (2007): 1171-1181.
    '''

    # Move coils to the back
    sens = np.moveaxis(sens, sens_coil_axis, -1)
    k = np.moveaxis(k, coil_axis, -1)
    sx, sy, nc = sens.shape[:]

    # Create a k-d tree
    kxy = np.concatenate((kx[:, None], ky[:, None]), axis=-1)
    kdtree = cKDTree(kxy)

    # Find spectrum of coil sensitivities interpolated at acquired
    # points (kx, ky)
    ax = (0, 1)
    ksens = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(
        sens, axes=ax), axes=ax), axes=ax)/np.sqrt(sx*sy)

    # Find kxy that satisfy || k_mu - k_rho ||_2 <= ws
    tx, ty = np.meshgrid(
        np.linspace(np.min(kx), np.max(kx), sx),
        np.linspace(np.min(ky), np.max(ky), sy))
    tx, ty = tx.flatten(), ty.flatten()
    txy = np.concatenate((tx[:, None], ty[:, None]), axis=-1)

    # Build G
    nk, nc = k.shape[:]
    G = lil_matrix((nk*nc, tx.size), dtype=k.dtype)

    t0 = time()
    Ginterp = {}
    idx = {}
    sx2, sy2 = int(sx/2), int(sy/2)
    for ii in range(tx.size):
        for jj in range(nc):

            if jj not in idx:
                # Choose ws cutoff frequency to be when ksens
                # decreases to around %0.36 of its peak value
                ll = np.abs(ksens[sx2, sy2:, jj])
                p = np.max(ll)*0.0036
                ws = np.argmin(np.abs(ll - p))
                idx[jj] = kdtree.query_ball_point(txy, r=ws)

            if jj not in Ginterp:
                Ginterp[jj] = griddata(
                    (tx, ty), ksens[..., jj].flatten(), (kx, ky),
                    method='cubic')

            # Stick interpolated values in for this coil
            idx0 = np.array(idx[jj][ii]) + jj*nk
            G[idx0, ii] = Ginterp[jj][idx[jj][ii]]

    print('Took %g seconds to build G' % (time() - t0))

    # import matplotlib.pyplot as plt
    # plt.imshow(np.abs(G).todense())
    # plt.show()

    d = k.flatten('F')
    m = np.linalg.pinv(G.todense()) @ d
    return np.reshape(m, (sx, sy))
