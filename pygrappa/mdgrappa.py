'''Python implementation of multidimensional GRAPPA.'''

from collections import defaultdict

import numpy as np
from skimage.util import view_as_windows


def mdgrappa(
        kspace,
        calib=None,
        kernel_size=None,
        coil_axis=-1,
        lamda=0.01,
        nnz=None):
    '''GeneRalized Autocalibrating Partially Parallel Acquisitions.

    Parameters
    ----------
    kspace : N-D array
        Measured undersampled complex k-space data. N-1 dimensions
        hold spatial frequency axes (kx, ky, kz, etc.).  1 dimension
        holds coil images (`coil_axis`).  The missing entries should
        have exactly 0.
    calib : N-D array or None, optional
        Fully sampled calibration data.  If `None`, calibration data
        will be extracted from the largest possible hypercube with
        origin at the center of k-space.
    kernel_size : tuple or None, optional
        The size of the N-1 dimensional GRAPPA kernels: (kx, ky, ...).
        Default: (5,)*(kspace.ndim-1)
    coil_axis : int, optional
        Dimension holding coil images.
    lamda : float, optional
        Tikhonov regularization constant for kernel calibration.
    nnz : int or None, optional
        Number of nonzero elements in a multidimensional patch
        required to train/apply a kernel.
        Default: `sqrt(prod(kernel_size))`.

    Returns
    -------
    res : array_like
        k-space data where missing entries have been filled in.

    Notes
    -----
    Based on the GRAPPA algorithm described in [1]_.

    All axes (except coil axis) are used for GRAPPA reconstruction.

    References
    ----------
    .. [1] Griswold, Mark A., et al. "Generalized autocalibrating
           partially parallel acquisitions (GRAPPA)." Magnetic
           Resonance in Medicine: An Official Journal of the
           International Society for Magnetic Resonance in Medicine
           47.6 (2002): 1202-1210.
    '''

    # coils to the back
    kspace = np.moveaxis(kspace, coil_axis, -1)
    nc = kspace.shape[-1]

    # Make sure we have a kernel_size
    if kernel_size is None:
        kernel_size = (5,)*(kspace.ndim-1)
    assert len(kernel_size) == kspace.ndim-1, (
        'kernel_size must have %d entries' % (kspace.ndim-1))

    # Only consider sampling patterns that have at least nnz samples
    if nnz is None:
        nnz = int(np.sqrt(np.prod(kernel_size)))

    # User can supply calibration region separately or we can find it
    if calib is not None:
        calib = np.moveaxis(calib, coil_axis, -1)
    else:
        # Find the calibration region and split it out from kspace
        raise NotImplementedError("Auto ACS extraction not implemented!")
        # calib = find_acs(kspace)

    # Pad the arrays
    pads = [int(k/2) for k in kernel_size]
    adjs = [np.mod(k, 2) for k in kernel_size]
    kspace = np.pad(kspace, [(pd, pd) for pd in pads] + [(0, 0)], mode='constant')
    calib = np.pad(calib, [(pd, pd) for pd in pads] + [(0, 0)], mode='constant')

    # Find all the unique sampling patterns
    mask = np.abs(kspace[..., 0]) > 0
    P = defaultdict(list)
    for idx in np.argwhere(~mask[tuple([slice(pd, -pd) for pd in pads])]):
        p0 = mask[tuple([slice(ii, ii+2*pd+adj) for ii, pd, adj in zip(idx, pads, adjs)])].flatten()
        if np.sum(p0) >= nnz:  # only counts if it has enough samples
            P[tuple(p0.astype(int))].append(idx)

    # We need all overlapping patches from calibration data
    A = view_as_windows(
        calib, tuple(kernel_size) + (nc,)).reshape((-1, np.prod(kernel_size), nc,))

    # Train and apply kernels
    ctr = np.ravel_multi_index([pd for pd in pads], dims=kernel_size)
    recon = np.zeros(kspace.shape, dtype=kspace.dtype)
    for key, holes in P.items():

        # Get sampling pattern from key
        p0 = np.array(p0, dtype=bool)

        # Train kernels
        S = A[:, p0, :].reshape(A.shape[0], -1)
        T = A[:, ctr, :]
        ShS = S.conj().T @ S
        ShT = S.conj().T @ T
        lamda0 = lamda*np.linalg.norm(ShS)/ShS.shape[0]
        W = np.linalg.solve(
            ShS + lamda0*np.eye(ShS.shape[0]), ShT)

        # Doesn't seem to be a big difference in speed?
        # Try gathering all sources and doing single matrix multiply
        # S = np.empty((len(holes), W.shape[0]), dtype=kspace.dtype)
        # targets = np.empty((kspace.ndim-1, len(holes)), dtype=int)
        # for jj, idx in enumerate(holes):
        #    S[jj, :] = kspace[tuple([slice(ii, ii+2*pd+adj) for ii, pd, adj in zip(idx, pads, adjs)] +
        #               [slice(None)])].reshape((-1, nc))[p0, :].flatten()
        #     targets[:, jj] = [ii + pd for ii, pd in zip(idx, pads)]
        # recon = np.reshape(recon, (-1, nc))
        # targets = np.ravel_multi_index(targets, dims=kspace.shape[:-1])
        # recon[targets, :] = S @ W
        # recon = np.reshape(recon, kspace.shape)

        # Apply kernel to fill each hole
        for idx in holes:
            S = kspace[tuple([slice(ii, ii+2*pd+adj) for ii, pd, adj in zip(idx, pads, adjs)] +
                             [slice(None)])].reshape((-1, nc))[p0, :].flatten()
            recon[tuple([ii + pd for ii, pd in zip(idx, pads)] + [slice(None)])] = S @ W

    # Add back in the measured voxels, put axis back where it goes
    recon[mask] += kspace[mask]
    return np.moveaxis(
        recon[tuple([slice(pd, -pd) for pd in pads] + [slice(None)])], -1, coil_axis)


if __name__ == '__main__':
    pass
