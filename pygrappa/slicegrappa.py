'''Python implementation of the Slice-GRAPPA algorithm.'''

import numpy as np
from skimage.util import view_as_windows, pad

def slicegrappa(
        kspace, calib, kernel_size=(5, 5), coil_axis=-2,
        time_axis=-1, slice_axis=-1, lamda=0.01):
    '''Slice-GRAPPA.

    Parameters
    ----------
    kspace : array_like
        Time frames of sum of k-space coil measurements for
        multiple slices.
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
    nx, ny, nc, _nt = kspace.shape[:]
    kx, ky = kernel_size[:]
    kx2, ky2 = int(kx/2), int(ky/2)
    _cx, _cy, nc, cs = calib.shape[:]

    # Pad kspace data
    kspace = pad( # pylint: disable=E1102
        kspace, ((kx2, kx2), (ky2, ky2), (0, 0), (0, 0)),
        mode='constant')
    calib = pad( # pylint: disable=E1102
        calib, ((kx2, kx2), (ky2, ky2), (0, 0), (0, 0)),
        mode='constant')

    # Get all overlapping patches of single slice calibration data
    A = view_as_windows(
        calib, (kx, ky, nc, cs)).reshape((-1, kx, ky, nc, cs))

    # Source data from SMS simulated calibration data
    calib_sms = np.fft.fft2(np.sum(np.fft.ifft2(
        calib, axes=(0, 1)), -1), axes=(0, 1))
    S = view_as_windows(
        calib_sms, (kx, ky, nc)).reshape((-1, kx*ky*nc))

    # Train a kernel for each target slice
    res = np.zeros((nx, ny, nc, cs), dtype=kspace.dtype)
    for sl in range(cs):

        # Train GRAPPA kernel for the current slice
        T = A[:, kx2, ky2, :, sl] # Target slice
        ShS = S.conj().T @ S
        ShT = S.conj().T @ T
        lamda0 = lamda*np.linalg.norm(ShS)/ShS.shape[0]
        W = np.linalg.solve(
            ShS + lamda0*np.eye(ShS.shape[0]), ShT)

        # Check to make sure it does what we think it's doing
        res[..., sl] = (S @ W).reshape((nx, ny, nc))
        import matplotlib.pyplot as plt
        res0 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(
            res[..., sl], axes=(0, 1)), axes=(0, 1)), axes=(0, 1))
        plt.imshow(np.abs(res0[..., 0]))
        plt.show()
