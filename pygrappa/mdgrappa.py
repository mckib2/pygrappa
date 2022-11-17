'''Python implementation of multidimensional GRAPPA.'''

from collections import defaultdict
from time import time
import logging

import numpy as np
from skimage.util import view_as_windows

from pygrappa.train_kernels import train_kernels
from .find_acs import find_acs


def mdgrappa(
        kspace,
        calib=None,
        kernel_size=None,
        coil_axis=-1,
        lamda=0.01,
        weights=None,
        ret_weights=False):
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
    weights : dict, optional
        Maps sampling patterns to trained kernels.
    ret_weights : bool, optional
        Return the trained weights as a dictionary mapping sampling
        patterns to kernels. Default is ``False``.

    Returns
    -------
    res : array_like
        k-space data where missing entries have been filled in.
    weights : dict, optional
        Returned if ``ret_weights=True``.

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

    # User can supply calibration region separately or we can find it
    if calib is not None:
        calib = np.moveaxis(calib, coil_axis, -1)
    else:
        # Find the calibration region and split it out from kspace
        calib = find_acs(kspace, coil_axis=-1)

    # Pad the arrays
    pads = [int(k/2) for k in kernel_size]
    adjs = [np.mod(k, 2) for k in kernel_size]
    kspace = np.pad(
        kspace, [(pd, pd) for pd in pads] + [(0, 0)], mode='constant')
    calib = np.pad(
        calib, [(pd, pd) for pd in pads] + [(0, 0)], mode='constant')
    mask = np.abs(kspace[..., 0]) > 0

    t0 = time()
    padmask = ~mask
    for ii in range(mask.ndim):
        padmask[tuple([slice(0, pd) if ii == jj else slice(None) for jj, pd in enumerate(pads)])] = False
        padmask[tuple([slice(-pd, None) if ii == jj else slice(None) for jj, pd in enumerate(pads)])] = False
    P = defaultdict(list)
    idxs = np.moveaxis(np.indices(mask.shape), 0, -1)[padmask]
    for ii, idx in enumerate(idxs):
        p0 = mask[tuple([slice(ii-pd, ii+pd+adj) for ii, pd, adj in zip(idx, pads, adjs)])].flatten()
        P[p0.tobytes()].append(tuple(idx))
    P = {k: np.array(v).T for k, v in P.items()}
    logging.info('Took %g seconds to find geometries and holes', (time() - t0))

    # We need all overlapping patches from calibration data
    A = view_as_windows(
        calib,
        tuple(kernel_size) + (nc,)).reshape(
            (-1, np.prod(kernel_size), nc,))

    # Set everything up to train and apply weights
    ksize = np.prod(kernel_size)*nc
    S = np.empty((np.max([P[k].shape[1] for k in P] if P else [0]), ksize), dtype=kspace.dtype)
    recon = np.zeros((np.prod(kspace.shape[:-1]), nc), dtype=kspace.dtype)

    def _apply_weights(holes, p0, np0, Ws0):
        # Collect all the sources
        for jj, _idx in enumerate(holes.T):
            S[jj, :np0] = kspace[tuple([slice(kk-pd, kk+pd+adj)
                                        for kk, pd, adj in zip(_idx, pads, adjs)])].reshape((-1, nc))[p0, :].flatten()
        # Apply kernel to all sources to generate all targets at once
        recon[np.ravel_multi_index(holes, mask.shape)] = np.einsum(
            'fi,ij->fj', S[:holes.shape[1], :np0], Ws0)

    if not weights:
        # train weights
        t0 = time()
        Ws = train_kernels(kspace.astype(np.complex128), nc, A.astype(np.complex128), P,
                           np.array(kernel_size, dtype=np.uintp),
                           np.array(pads, dtype=np.uintp), lamda)
        logging.info('Took %g seconds to train weights', (time() - t0))

        # Fill holes for each geometry
        t0 = time()
        for ii, (key, holes) in enumerate(P.items()):
            p0 = np.frombuffer(key, dtype=bool)
            np0 = np.sum(p0)*nc
            _apply_weights(holes, p0, np0, Ws[ii, :np0, :])
        logging.info('Took %g seconds to apply weights', (time() - t0))
    else:
        # Unpack weights and fill holes for each geometry
        t0 = time()
        for ii, (key, holes) in enumerate(P.items()):
            p0 = np.frombuffer(key, dtype=bool)
            np0 = weights[key].shape[0]
            _apply_weights(holes, p0, np0, weights[key])
        logging.info('Took %g seconds to unpack and apply weights', (time() - t0))

    # Add back in the measured voxels, put axis back where it goes
    recon = np.reshape(recon, kspace.shape)
    recon[mask] += kspace[mask]
    recon = np.moveaxis(
        recon[tuple([slice(pd, -pd) for pd in pads] + [slice(None)])],
        -1, coil_axis)

    if ret_weights:
        if weights:
            return (recon, weights)
        return (recon,
                {k: Ws[ii, :np.sum(np.frombuffer(k, dtype=bool))*nc, :]
                 for ii, k in enumerate(P)})
    return recon


if __name__ == '__main__':
    pass
