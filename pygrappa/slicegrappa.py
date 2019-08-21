'''Python implementation of the Slice-GRAPPA algorithm.'''

import numpy as np
from skimage.util import view_as_windows

def slicegrappa(
        kspace, calib, kernel_size=(5, 5), coil_axis=-2,
        time_axis=-1, slice_axis=-1):
    '''Slice-GRAPPA.

    Parameters
    ----------
    kspace : array_like
        The sum of k-space coil measurements for multiple slices.
    calib : array_like
        Single slice measurements for each slice present in kspace.
        Should be the same dimensions.
    kernel_size : tuple, optional
        Size of the GRAPPA kernel: (kx, ky).

    References
    ----------
    .. [1] Setsompop, Kawin, et al. "Blipped‐controlled aliasing in
           parallel imaging for simultaneous multislice echo planar
           imaging with reduced g‐factor penalty." Magnetic resonance
           in medicine 67.5 (2012): 1210-1224.
    '''

    # Put the axes where we expect them
    kspace = np.moveaxis(kspace, (coil_axis, time_axis), (-2, -1))
    calib = np.moveaxis(calib, (coil_axis, slice_axis), (-2, -1))
    # nx, ny, nc, nt = kspace.shape[:]
    kx, ky = kernel_size[:]
    _cx, _cy, nc, cs = calib.shape[:]

    # Get all overlapping patches of single slice calibration data
    A = view_as_windows(
        calib, (kx, ky, nc, cs)).reshape((-1, kx, ky, nc, cs))

    print(A.shape)
