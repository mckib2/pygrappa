'''Python implementation of the Slice-GRAPPA algorithm.'''

import numpy as np
from skimage.util import view_as_windows, pad
from tqdm import trange

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

    Returns
    -------
    res : array_like
        Reconstructed slices for each time frame.  res has fixed
        shape: (nx, ny, num_coils, num_time_frames, num_slices).

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
    nx, ny, nc, nt = kspace.shape[:]
    kx, ky = kernel_size[:]
    kx2, ky2 = int(kx/2), int(ky/2)
    # adjx, adjy = np.mod(kx, 2), np.mod(ky, 2)
    _cx, _cy, nc, cs = calib.shape[:]

    # Pad kspace data
    kspace = pad( # pylint: disable=E1102
        kspace, ((kx2, kx2), (ky2, ky2), (0, 0), (0, 0)),
        mode='constant')
    calib = pad( # pylint: disable=E1102
        calib, ((kx2, kx2), (ky2, ky2), (0, 0), (0, 0)),
        mode='constant')

    # Source data from SMS simulated calibration data
    S = view_as_windows(
        np.sum(calib, axis=-1), (kx, ky, nc)).reshape((-1, kx*ky*nc))

    # Train a kernel for each target slice
    W = np.zeros((cs, S.shape[-1], nc), dtype=calib.dtype)
    for sl in range(cs):

        # Train GRAPPA kernel for the current slice
        T = calib[kx2:-kx2, ky2:-ky2, :, sl].reshape((-1, nc))
        ShS = S.conj().T @ S
        ShT = S.conj().T @ T
        lamda0 = lamda*np.linalg.norm(ShS)/ShS.shape[0]
        W[sl, ...] = np.linalg.solve(
            ShS + lamda0*np.eye(ShS.shape[0]), ShT)

        # # Check to make sure it does what we think it's doing
        # res[..., sl] = (S @ W[sl, ...]).reshape((nx, ny, nc))
        # import matplotlib.pyplot as plt
        # res0 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(
        #     res[..., sl], axes=(0, 1)), axes=(0, 1)), axes=(0, 1))
        # plt.imshow(np.sqrt(np.sum(np.abs(res0)**2, axis=-1)))
        # plt.show()

    # Now pull apart slices for each time frame
    res = np.zeros((nx, ny, nc, nt, cs), dtype=kspace.dtype)
    S = view_as_windows(
        kspace, (kx, ky, nc, nt)).reshape((-1, kx*ky*nc, nt))
    for tt in trange(nt, leave=False, desc='Slice-GRAPPA'):
        for sl in range(cs):
            res[..., tt, sl] = (
                S[..., tt] @ W[sl, ...]).reshape((nx, ny, nc))

    # Return results in fixed order: (nx, ny, nc, nt, cs)
    return res
