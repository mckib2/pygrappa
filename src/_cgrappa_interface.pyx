# distutils: language = c++
# cython: language_level=3

cimport numpy as np
import numpy as np

cdef extern from "_cgrappa.hpp" nogil:
    int _cgrappa_complex64(
        size_t,
        size_t*,
        size_t*,
        float complex*,
        float complex*,
        size_t*,
        float complex*)

def cgrappa_complex64(
        kspace,
        calib,
        kernel_size):
    '''Forward arguments to C++ implementation.'''

    cdef size_t ndim = kspace.ndim
    cdef size_t[::1] kspace_dims = np.array(kspace.shape).astype(np.uintp)
    cdef size_t[::1] calib_dims = np.array(calib.shape).astype(np.uintp)
    cdef float complex[::1] recon_memview = kspace.copy().flatten().astype(np.complex64)
    cdef float complex[::1] kspace_memview = kspace.flatten().astype(np.complex64)
    cdef float complex[::1] calib_memview = calib.flatten().astype(np.complex64)
    cdef size_t[::1] kernel_size_memview = np.array(kernel_size).astype(np.uintp)

    cdef int res = _cgrappa_complex64(
        ndim,
        &kspace_dims[0],
        &calib_dims[0],
        &kspace_memview[0],
        &calib_memview[0],
        &kernel_size_memview[0],
        &recon_memview[0])

    return np.array(recon_memview).reshape(kspace.shape)
