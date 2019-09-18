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

    # Normalize spatial frequencies to be in [0, 1]
    # kx0 = kx/(2*np.max(np.abs(kx)))
    # ky0 = ky/(2*np.max(np.abs(ky)))
    # kx0 = kx + 1
    # ky0 = ky + 1

    F = np.zeros((M, N), dtype='complex')
    yy, xx = np.meshgrid(
        np.linspace(-np.pi, np.pi, M),
        np.linspace(-np.pi, np.pi, N))

    for ii in range(k.size):
        exp = np.exp(1j*(kx[ii]*xx + ky[ii]*yy))
        F += exp*k[ii]
    return F/kx.size
