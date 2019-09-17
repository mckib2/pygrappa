# distutils: language = c++
# cython: language_level=3

import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def idft2d(kx, ky, k, M, N):
    '''Naive implementation of the 2D IDFT.

    Parameters
    ----------
    k : array_like
        kspace coefficients corresponding to coordinates (kx, ky).
    kx, ky : array_like
        Spatial frequency coordinates at which to evaluate the IDFT.
        kx and ky are 1D arrays.
    M, N : int
        Size of reconstructed image.
    '''

    # Normalize spatial frequencies
    kx = kx/(2*np.max(np.abs(kx)))
    ky = ky/(2*np.max(np.abs(ky)))
    kx = kx + 1
    ky = ky + 1

    F = np.zeros((M, N), dtype='complex')
    yy, xx = np.meshgrid(
        np.linspace(-M/2, M/2, M),
        np.linspace(-N/2, N/2, N))

    for ii in range(k.size):
        exp = np.exp(1j*2*np.pi*(kx[ii]*xx + ky[ii]*yy))
        F += exp*k[ii]

    # I think we need some density compensation...  Voronoi weights?

    return F
