# distutils: language = c++
import numpy as np
from skimage.util import pad, view_as_windows
from libcpp.map cimport map
from libcpp.vector cimport vector
from cython.operator cimport dereference, postincrement

ctypedef vector[unsigned int] vector_uint

cdef extern from "grappa_in_c.h":
    map[unsigned long long int, vector_uint] grappa_in_c(
        complex*, int*, unsigned int, unsigned int,
        complex*, unsigned int, unsigned int,
        unsigned int, unsigned int, unsigned int)

def cgrappa(kspace, calib, kernel_size, lamda=.01, coil_axis=-1):

    # Put coil axis in the back
    print('start move axis')
    kspace = np.moveaxis(kspace, coil_axis, -1)
    calib = np.moveaxis(calib, coil_axis, -1)
    print('end move axis')

    # Make sure we have arrays in C contiguous order
    print('start make contiguous')
    if not kspace.flags['C_CONTIGUOUS']:
        kspace = np.ascontiguousarray(kspace)
    if not calib.flags['C_CONTIGUOUS']:
        calib = np.ascontiguousarray(calib)
    print('end make contiguous')

    # Get size of arrays
    kx, ky, nc = kspace.shape[:]
    cx, cy, nc = calib.shape[:]
    ksx, ksy = kernel_size[:]
    ksx2, ksy2 = int(ksx/2), int(ksy/2)
    adjx = np.mod(ksx, 2)
    adjy = np.mod(ksy, 2)

    print('start mask')
    mask = (np.abs(kspace[:, :, 0]) > 0).astype(np.int32)
    print(mask)
    print('end mask')

    # Pad the arrays
    print('start pad')
    kspace = pad(
        kspace, ((ksx2, ksx2), (ksy2, ksy2), (0, 0)), mode='constant')
    calib = pad(
        calib, ((ksx2, ksx2), (ksy2, ksy2), (0, 0)), mode='constant')
    print('end pad')

    # Define complex memory views, ::1 ensures contiguous
    cdef:
        double complex[:, :, ::1] kspace_memview = kspace
        double complex[:, :, ::1] calib_memview = calib
        int[:, ::1] mask_memview = mask

    # Pass in arguments to C function, arrays pass pointer to start
    # of arrays, i.e., [x=0, y=0, coil=0].
    print('start c')
    cdef map[unsigned long long int, vector_uint] res
    res = grappa_in_c(
        &kspace_memview[0, 0, 0],
        &mask_memview[0, 0],
        kx, ky,
        &calib_memview[0, 0, 0], cx, cy,
        nc, ksx, ksy)
    print('end c')

    # Get all overlapping patches of ACS
    print('start A')
    A = view_as_windows(
        calib, (ksx, ksy, nc)).reshape((-1, ksx, ksy, nc))
    print('end A')

    # Initialize recon array
    recon = np.zeros(kspace.shape, dtype=kspace.dtype)

    cdef map[unsigned long long int, vector_uint].iterator it = res.begin()
    while(it != res.end()):

        P = format(dereference(it).first, 'b').zfill(ksx*ksy)
        P = (np.fromstring(P, np.int8) - ord('0')).astype(bool)
        # P = np.flip(P) # I think we should flip it?  Don't know...
        P = P.reshape((ksx, ksy))
        P = np.tile(P[..., None], (1, 1, nc))


        S = A[:, P]
        T = A[:, ksx2, ksy2, :]
        ShS = S.conj().T @ S
        ShT = S.conj().T @ T
        lamda0 = lamda*np.linalg.norm(ShS)/ShS.shape[0]
        W = np.linalg.solve(
            ShS + lamda0*np.eye(ShS.shape[0]), ShT).T

        idx = dereference(it).second
        x, y = np.unravel_index(idx, (kx, ky))
        x += ksx2 # Remember zero padding
        y += ksy2 # Remember zero padding
        for xx, yy in zip(x, y):
            # Collect sources for this hole and apply weights
            S = kspace[xx-ksx2:xx+ksx2+adjx, yy-ksy2:yy+ksy2+adjy, :]
            S = S[P]
            recon[xx, yy, :] = (W @ S[:, None]).squeeze()

        postincrement(it)

    return np.moveaxis(
        (recon[:] + kspace)[ksx2:-ksx2, ksy2:-ksy2, :], -1, coil_axis)
