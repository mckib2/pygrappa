# distutils: language = c++
# cython: language_level=3

cimport numpy as np
from libcpp.unordered_set cimport unordered_set
from libcpp.vector cimport vector
cimport cython

cdef extern from "math.h":
    double round(double d)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def grog_powers(
        const double[::1] tx,
        const double[::1] ty,
        const double[::1] kx,
        const double[::1] ky,
        const vector[vector[int]] idx,
        const unsigned int precision
    ):
    '''Find unique fractional matrix powers.'''

    cdef:
        unordered_set[double] dx, dy
        unsigned int ii, N, M, idx0
        double pval

    pval = 10.0**precision
    N = len(tx)
    for ii in range(N):
        M = idx[ii].size()
        for jj in range(M):
            idx0 = idx[ii][jj]
            dx.insert(round((tx[ii] - kx[idx0])*pval)/pval)
            dy.insert(round((ty[ii] - ky[idx0])*pval)/pval)

    return dx, dy
