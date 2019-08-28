'''Python implementation of the Slice-GRAPPA algorithm.'''

import numpy as np
from skimage.util import view_as_windows, pad
from tqdm import trange

def slicegrappa(
        kspace, calib, kernel_size=(5, 5), prior='sim', coil_axis=-2,
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
    prior : { 'sim', 'kspace' }, optional
        How to construct GRAPPA sources.  GRAPPA weights are found by
        solving the least squares problem T = S W, where T are the
        targets (calib), S are the sources, and W are the weights.
        The possible options are:

            - 'sim': simulate SMS acquisition from calibration data,
              i.e., sources S = sum(calib, axis=slice_axis).  This
              presupposes that the spatial locations of the slices in
              the calibration data are the same as in the overlapped
              kspace data.  This is similar to how the k-t BLAST
              Wiener filter is constructed (see equation 1 in [2]_).
            - 'kspace': uses the first time frame of the overlapped
              data as sources, i.e., S = kspace[1st time frame].

    coil_axis : int, optional
        Dimension that holds the coil data.
    time_axis : int, optional
        Dimension of kspace that holds the time data.
    slice_axis : int, optional
        Dimension of calib that holds the slice information.
    lamda : float, optional
        Tikhonov regularization for the kernel calibration.

    Returns
    -------
    res : array_like
        Reconstructed slices for each time frame.  res will always
        return the data in fixed order or shape:
        (nx, ny, num_coils, num_time_frames, num_slices).

    References
    ----------
    .. [1] Setsompop, Kawin, et al. "Blipped‐controlled aliasing in
           parallel imaging for simultaneous multislice echo planar
           imaging with reduced g‐factor penalty." Magnetic resonance
           in medicine 67.5 (2012): 1210-1224.
    .. [2] Sigfridsson, Andreas, et al. "Improving temporal fidelity
           in k-t BLAST MRI reconstruction." International Conference
           on Medical Image Computing and Computer-Assisted
           Intervention. Springer, Berlin, Heidelberg, 2007.
    '''

    # Make sure we know how to construct the sources:
    if prior not in ['sim', 'kspace']:
        raise NotImplementedError("Unknown 'prior' value: %s" % prior)

    # Put the axes where we expect them
    kspace = np.moveaxis(kspace, (coil_axis, time_axis), (-2, -1))
    calib = np.moveaxis(calib, (coil_axis, slice_axis), (-2, -1))
    nx, ny, nc, nt = kspace.shape[:]
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

    # Figure out how to construct the sources:
    if prior == 'sim':
        # Source data from SMS simulated calibration data.  This is
        # constructing the "prior" like k-t BLAST does, using the
        # calibration data to form the aliased/overlapped images.
        # This requires the single-band images to be in the same
        # spatial locations as the SMS data.
        S = view_as_windows(np.sum(
            calib, axis=-1), (kx, ky, nc)).reshape((-1, kx*ky*nc))
    elif prior == 'kspace':
        # Source data from the first time frame of the kspace data.
        S = view_as_windows(np.ascontiguousarray(
            kspace[..., 0]), (kx, ky, nc)).reshape((-1, kx*ky*nc))

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
