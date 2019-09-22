'''Python implementation of the kSPA algorithm.'''

from time import time

import numpy as np
from scipy.interpolate import griddata
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from tqdm import trange

def kspa(kx, ky, k, sens, ws=8, coil_axis=-1):
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

    # Coils to da back
    k = np.moveaxis(k, coil_axis, -1)
    sens = np.moveaxis(sens, coil_axis, -1)

    # d = G m
    # (ns*nc, 1) = (ns*nc, N) (N, 1)

    # d is a column vector of coil data, column stacking -> Fortran
    # ordering to match how we will construct G
    d = k.flatten('F')

    # G is the convolution matrix of coil sensitivities
    # Would upsampling help here like it did in PARS?
    # from scipy.ndimage import zoom
    # sensr = zoom(sens.real, (1.5, 1.5, 1))
    # sensi = zoom(sens.imag, (1.5, 1.5, 1))
    # sens = sensr + 1j*sensi
    sx, sy, nc = sens.shape[:]
    sens = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(
        sens, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))

    # Coil sensitivities assumed to be sampled on a Cartesian grid
    tx, ty = np.meshgrid(
        np.linspace(np.min(kx), np.max(kx), sx),
        np.linspace(np.min(ky), np.max(ky), sy))
    tx, ty = tx.flatten(), ty.flatten()

    dx = kx[:, None] - tx[None, :]
    dy = ky[:, None] - ty[None, :]

    # # Purposefully set samples outside radius to np.nan, which will
    # # be converted to zeros
    idx = np.sqrt(dx**2 + dy**2) > ws
    dx[idx] = np.inf
    dy[idx] = np.inf
    # print('ws applied!')
    # idx = np.sqrt(dx**2 + dy**2) <= ws
    # print(len(idx))

    t0 = time()
    G = lil_matrix((d.size, tx.size), dtype=k.dtype)
    for ii in trange(nc, leave=False):
        G[ii*kx.size:(ii+1)*kx.size, :] = np.nan_to_num(griddata(
            (tx, ty), sens[..., ii].flatten(), (dx, dy)))
    print(d.size*tx.size, G.getnnz())
    G = G.tocsc()
    print('Took %g seconds to build G' % (time() - t0))

    # G = np.zeros((d.size, tx.size), dtype=k.dtype)
    # print('G allocated: %s' % str(G.shape))
    # for ii in trange(nc):
    #     G[ii*kx.size:(ii+1)*kx.size, :] = griddata(
    #         (tx, ty), sens[..., ii].flatten(), (dx, dy))
    # G = np.nan_to_num(G)
    # print('Done constructing G!')

    # # Make sure G is sparse
    # import matplotlib.pyplot as plt
    # plt.imshow(np.abs(G.todense()))
    # plt.show()

    t0 = time()
    GhG = G.getH().dot(G)
    Ghd = G.getH().dot(d)
    m = spsolve(GhG, Ghd)
    print('Took %g seconds for spsolve' % (time() - t0))
    return np.reshape(m, (sx, sy), 'F')


    #
    # # d = G m
    # # Gh d = GhG m
    # # (GhG)^-1 Gh d = m
    # GhG = G.conj().T @ G
    # Ghd = G.conj().T @ d
    #
    # # Paper suggests Fermi window, use something simple for now
    # fil = np.hamming(sx)[:, None]*np.hamming(sy)[None, :]
    # # import matplotlib.pyplot as plt
    # # plt.imshow(fil)
    # # plt.show()
    #
    # # Fortran ordering to match flattening of d, notice there is no
    # # longer a coil axis!  Coils are combined at end of kSPA!
    # return np.reshape(
    #     np.linalg.solve(GhG, Ghd), (sx, sy), order='F')*fil**2
