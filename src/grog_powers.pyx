# distutils: language = c++
# cython: language_level=3

import numpy as np
cimport numpy as np
from libcpp.unordered_set cimport unordered_set
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def grog_powers(
        tx, ty, kx, ky, idx,
        int precision
    ):
    '''Find unique fractional matrix powers.'''

    cdef:
        double[::1] tx_memview = tx
        double[::1] ty_memview = ty
        double[::1] kx_memview = kx
        double[::1] ky_memview = ky

        unordered_set[double] dx, dy

        unsigned int ii, N, M
        double pval

    pval = 10.0**precision
    N = tx.size
    for ii in range(N):
        M = len(idx[ii])
        for jj in range(M):
            dx.insert(round(
                (tx_memview[ii] - kx_memview[idx[ii][jj]])*pval)/pval)
            dy.insert(round(
                (ty_memview[ii] - ky_memview[idx[ii][jj]])*pval)/pval)

    return(dx, dy)
