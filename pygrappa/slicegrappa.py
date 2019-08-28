'''Python implementation of the Slice-GRAPPA algorithm.'''

import numpy as np
from skimage.util import view_as_windows, pad
from tqdm import trange

def slicegrappa(
        kspace, calib, kernel_size=(5, 5), prior='sim', coil_axis=-2,
        time_axis=-1, slice_axis=-1, lamda=0.01, split=False):
    '''(Split)-Slice-GRAPPA for SMS reconstruction.

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

        This option is not used for Split-Slice-GRAPPA.

    coil_axis : int, optional
        Dimension that holds the coil data.
    time_axis : int, optional
        Dimension of kspace that holds the time data.
    slice_axis : int, optional
        Dimension of calib that holds the slice information.
    lamda : float, optional
        Tikhonov regularization for the kernel calibration.
    split : bool, optional
        Uses Split-Slice-GRAPPA kernel training method.

    Returns
    -------
    res : array_like
        Reconstructed slices for each time frame.  res will always
        return the data in fixed order or shape:
        (nx, ny, num_coils, num_time_frames, num_slices).

    Raises
    ------
    NotImplementedError
        When "prior" is an invalid option.

    Notes
    -----
    This function implements both the Slice-GRAPPA algorithm as
    described in [1]_ and the Split-Slice-GRAPPA algorithm as first
    described in [3]_.

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
    .. [3] Cauley, Stephen F., et al. "Interslice leakage artifact
           reduction technique for simultaneous multislice
           acquisitions." Magnetic resonance in medicine 72.1 (2014):
           93-102.
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

    # Figure out how to construct the sources (only relevant for
    # Slice-GRAPPA, not Split-Slice-GRAPPA):
    if not split:
        if prior == 'sim':
            # Source data from SMS simulated calibration data.  This
            # is constructing the "prior" like k-t BLAST does, using
            # the calibration data to form the aliased/overlapped
            # images.  This requires the single-band images to be in
            # the same spatial locations as the SMS data.
            S = view_as_windows(np.sum(
                calib, axis=-1), (kx, ky, nc)).reshape((-1, kx*ky*nc))
        elif prior == 'kspace':
            # Source data from the first time frame of the
            # kspace data.
            S = view_as_windows(np.ascontiguousarray(
                kspace[..., 0]), (kx, ky, nc)).reshape((-1, kx*ky*nc))

    # Train a kernel for each target slice -- use Split-Slice-GRAPPA
    # if the user asked for it, else use Slice-GRAPPA
    W = np.zeros((cs, kx*ky*nc, nc), dtype=calib.dtype)
    for sl in range(cs):

        # Train GRAPPA kernel for the current slice
        T = calib[kx2:-kx2, ky2:-ky2, :, sl].reshape((-1, nc))
        if not split:
            # Regular old Slice-GRAPPA
            ShS = S.conj().T @ S
            ShT = S.conj().T @ T
            lamda0 = lamda*np.linalg.norm(ShS)/ShS.shape[0]
            W[sl, ...] = np.linalg.solve(
                ShS + lamda0*np.eye(ShS.shape[0]), ShT)
        else:
            # Split-Slice-GRAPPA for all your slice leakage needs!
            # Equation (7) from ref. [3]:
            MhM = np.zeros((kx*ky*nc,)*2, dtype=calib.dtype)

            # This might be inefficient as we're getting patches
            # for every single slice for every slice kernel, but it
            # does run pretty fast and it's pretty memory intensive
            # to get all patches for all slices outside of the loop.
            # Maybe use temporary files?
            for jj in range(cs):
                calib0 = view_as_windows(np.ascontiguousarray(
                    calib[..., jj]), (kx, ky, nc)).reshape(
                        (-1, kx*ky*nc))
                MhM += calib0.conj().T @ calib0

                # Find and save the target calibration slice, Mz:
                if jj == sl:
                    Mz = calib0

            MhT = Mz.conj().T @ T
            lamda0 = lamda*np.linalg.norm(MhM)/MhM.shape[0]
            W[sl, ...] = np.linalg.solve(
                MhM + lamda0*np.eye(MhM.shape[0]), MhT)

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
