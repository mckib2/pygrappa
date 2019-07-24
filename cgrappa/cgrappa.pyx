# distutils: language = c++
import numpy as np
from skimage.util import pad

cdef extern from "grappa_in_c.h":
    void grappa_in_c(
        complex*, int*, unsigned int, unsigned int,
        complex*, unsigned int, unsigned int,
        unsigned int, unsigned int, unsigned int)

def cgrappa(kspace, calib, kernel_size, coil_axis=-1):

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

    print('start mask')
    mask = (np.abs(kspace[:, :, 0]) > 0).astype(np.int32)
    print(mask)
    print('end mask')

    # # Pad the arrays
    # print('start pad')
    # kspace = pad(
    #     kspace, ((ksx2, ksx2), (ksy2, ksy2), (0, 0)), mode='constant')
    # calib = pad(
    #     calib, ((ksx2, ksx2), (ksy2, ksy2), (0, 0)), mode='constant')
    # print('end pad')

    # Define complex memory views, ::1 ensures contiguous
    cdef:
        double complex[:, :, ::1] kspace_memview = kspace
        double complex[:, :, ::1] calib_memview = calib
        int[:, ::1] mask_memview = mask

    # Pass in arguments to C function, arrays pass pointer to start
    # of arrays, i.e., [x=0, y=0, coil=0].
    print('start c')
    grappa_in_c(
        &kspace_memview[0, 0, 0],
        &mask_memview[0, 0],
        kx, ky,
        &calib_memview[0, 0, 0], cx, cy,
        nc, ksx, ksy)
    print('end c')

# def multiply_by_10(arr): # 'arr' is a one-dimensional numpy array
#
#     if not arr.flags['C_CONTIGUOUS']:
#         # Makes a contiguous copy of the numpy array
#         arr = np.ascontiguousarray(arr)
#
#     cdef double[::1] arr_memview = arr
#
#     multiply_by_10_in_C(&arr_memview[0], arr_memview.shape[0])
#
#     return arr
