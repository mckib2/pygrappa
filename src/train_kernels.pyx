# distutils: language = c++
# cython: language_level=3

cimport cython

cimport numpy as np
import numpy as np

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def train_kernels(
        np.ndarray kspace,
        const Py_ssize_t nc,
        const np.complex[:, :, ::1] A,
        dict P,
        const size_t[::1] kernel_size,
        const size_t[::1] pads,
        const double lamda):

    # Train and apply kernels
    cdef Py_ssize_t ksize = nc
    cdef Py_ssize_t ii
    for ii in range(kernel_size.shape[0]):
        ksize *= kernel_size[ii]
    cdef complex[:, :, ::1] Ws = np.empty((len(P), ksize, nc), dtype=kspace.dtype)
    cdef complex[:, ::1] S = np.empty((A.shape[0], ksize), dtype=kspace.dtype, order='C')
    cdef Py_ssize_t ctr = np.ravel_multi_index([pd for pd in pads], dims=kernel_size)
    cdef complex[:, ::1] T = np.empty((A.shape[0], nc), dtype=kspace.dtype, order='C')
    cdef np.ndarray ShS = np.empty((ksize, ksize), dtype=kspace.dtype, order='C')
    cdef np.ndarray ShT = np.empty((ksize, nc), dtype=kspace.dtype, order='C')
    cdef int np0
    cdef double lamda0
    cdef Py_ssize_t jj, kk, aa, bb, cc
    cdef Py_ssize_t[::1] idx = np.empty(ksize, dtype=np.intp)
    cdef int M = A.shape[0]
    for ii, key in enumerate(P):
        np0 = 0
        for jj in range(len(key)):
            idx[np0] = jj
            np0 += key[jj]

        # gather sources
        for aa in range(M):
            for bb in range(np0):
                for cc in range(nc):
                    S[aa, bb*nc + cc] = A[aa, idx[bb], cc]

        for jj in range(nc):
            T[:, jj] = A[:, ctr, jj]

        # construct square matrices
        np0 *= nc  # consider all coil elements
        # TODO: use BLAS/LAPACK functions here
        ShS[:np0, :np0] = np.dot(np.conj(S[:M, :np0]).T, S[:M, :np0])
        ShT[:np0, :] = np.dot(np.conj(S[:M, :np0]).T, T)

        # tik reg
        if lamda:
            lamda0 = lamda*np.linalg.norm(ShS[:np0, :np0])/np0
            for jj in range(np0):
                ShS[jj, jj] += lamda0

        # Solve the LS problem
        ShT[:np0, :] = np.linalg.solve(ShS[:np0, :np0], ShT[:np0, :])

        for jj in range(np0):
            for kk in range(nc):
                Ws[ii, jj, kk] = ShT[jj, kk]

    return np.array(Ws)
