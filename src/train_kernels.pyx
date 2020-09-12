# distutils: language = c++
# cython: language_level=3

from libcpp cimport bool
cimport numpy as np
import numpy as np

cimport cython
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def train_kernels(
        np.ndarray kspace,
        Py_ssize_t nc,
        np.complex[:, :, ::1] A,
        dict P,
        size_t[::1] kernel_size,
        size_t[::1] pads,
        double lamda):

    # Train and apply kernels
    cdef Py_ssize_t ksize = nc
    cdef Py_ssize_t ii
    for ii in range(kernel_size.shape[0]):
        ksize *= kernel_size[ii]

    cdef complex[:, :, ::1] Ws = np.empty((len(P), ksize, nc), dtype=kspace.dtype)

    # Find the max size that sources workspace can be
    cdef Py_ssize_t max_holes = np.max([v.shape[1] for v in P.values()])
    cdef Py_ssize_t max_patches = np.max((max_holes, A.shape[0]))
    cdef complex[:, ::1] S = np.empty((max_patches, ksize), dtype=kspace.dtype)
    cdef Py_ssize_t ctr = np.ravel_multi_index([pd for pd in pads], dims=kernel_size)
    cdef complex[:, ::1] T = np.empty((A.shape[0], nc), dtype=kspace.dtype)
    # cdef complex[:, ::1] ShS = np.empty((ksize, ksize), dtype=kspace.dtype)
    # cdef complex[:, ::1] ShT = np.empty((ksize, nc), dtype=kspace.dtype)
    cdef bool[::1] p0
    cdef size_t np0
    cdef double lamda0
    cdef Py_ssize_t jj, kk, aa, bb, cc
    cdef Py_ssize_t[::1] idx = np.empty(ksize, dtype=np.intp)
    for ii, key in enumerate(P):
        p0 = np.fromstring(key, dtype=np.bool)

        np0 = 0
        for jj in range(len(p0)):
            idx[np0] = jj
            np0 += p0[jj]

        # gather sources
        for aa in range(A.shape[0]):
            for bb in range(np0):
                for cc in range(nc):
                    S[aa, bb*nc + cc] = A[aa, idx[bb], cc]

        # gather targets
        for jj in range(nc):
            T[:, jj] = A[:, ctr, jj]

        # construct square matrices
        np0 *= nc  # consider all coil elements
        ShS = np.dot(np.conj(S[:A.shape[0], :np0]).T, S[:A.shape[0], :np0])
        ShT = np.dot(np.conj(S[:A.shape[0], :np0]).T, T)

        # Solve the LS problem
        lamda0 = lamda*np.linalg.norm(ShS)/np0
        res = np.linalg.solve(
            ShS[:np0, :np0] + lamda0*np.eye(np0), ShT[:np0, :])

        for jj in range(np0):
            for kk in range(nc):
                Ws[ii, jj, kk] = res[jj, kk]

    return np.array(Ws)
