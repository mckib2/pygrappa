# distutils: language = c++
# cython: language_level=3

cimport numpy as np
from libcpp.unordered_set cimport unordered_set
from libcpp.vector cimport vector
cimport cython

cdef extern from "_grog_powers_template.h":
    vector[unordered_set[double]] _grog_powers_double(
            const double[] tx,
            const double[] ty,
            const double[] kx,
            const double[] ky,
            vector[vector[int]] idx,
            const int precision)

    vector[unordered_set[float]] _grog_powers_float(
            const float[] tx,
            const float[] ty,
            const float[] kx,
            const float[] ky,
            vector[vector[int]] idx,
            const int precision)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def grog_powers_float(
        const float[::1] tx,
        const float[::1] ty,
        const float[::1] kx,
        const float[::1] ky,
        vector[vector[int]] idx,
        const int precision):
    return _grog_powers_float(
        &tx[0], &ty[0], &kx[0], &ky[0], idx, precision)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def grog_powers_double(
        const double[::1] tx,
        const double[::1] ty,
        const double[::1] kx,
        const double[::1] ky,
        vector[vector[int]] idx,
        const int precision):
    return _grog_powers_double(
        &tx[0], &ty[0], &kx[0], &ky[0], idx, precision)
