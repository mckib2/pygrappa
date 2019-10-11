# distutils: language = c++
# cython: language_level=3

import numpy as np
cimport numpy as np
from libcpp.map cimport map
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def grog_gridding(
        tx, ty, kx, ky, k, idx, res, inside, Dx, Dy, int precision):
    '''Do GROG.

    Notes
    -----
    `res` is modified in-place.
    '''

    cdef:
        double[::1] tx_memview = tx
        double[::1] ty_memview = ty
        double[::1] kx_memview = kx
        double[::1] ky_memview = ky

        complex[:, ::1] k_memview = k

        unsigned int ii, jj, N, M, idx0, in0
        double pfac

    pfac = 10.0**precision
    N = tx.size
    for ii in range(N):
        M = len(idx[ii])
        in0 = inside[ii]
        for jj in range(M):
            idx0 = idx[ii][jj]
            Gxf = Dx[round((tx_memview[ii] - kx_memview[idx0])*pfac)/pfac]
            Gyf = Dy[round((ty_memview[ii] - ky_memview[idx0])*pfac)/pfac]

            res[in0, :] += Gxf @ Gyf @ k_memview[idx0, :]

        # Finish the averaging (dividing step)
        if M:
            res[in0, :] /= M
