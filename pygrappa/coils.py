'''Coil estimation strategies.'''

import numpy as np
from scipy.linalg import eigh
from skimage.filters import threshold_li


def walsh(imspace, mask=None, coil_axis=-1):
    '''Stochastic matched filter coil combine.

    Parameters
    ----------
    mask : array_like
        A mask indicating which pixels of the coil sensitivity mask
        should be computed. If ``None``, this will be computed by
        applying a threshold to the sum-of-squares coil combination.
        Must be the same shape as a single coil.
    coil_axis : int
        Dimension that has coils.

    Notes
    -----
    Adapted from [1]_.  Based on the paper [2]_.

    References
    ----------
    .. [1] https://github.com/ismrmrd/ismrmrd-python-tools/
           blob/master/ismrmrdtools/coils.py
    .. [2] Walsh, David O., Arthur F. Gmitro, and Michael W.
           Marcellin. "Adaptive reconstruction of phased array MR
           imagery." Magnetic Resonance in Medicine: An Official
           Journal of the International Society for Magnetic
           Resonance in Medicine 43.5 (2000): 682-690.
    '''
    imspace = np.moveaxis(imspace, coil_axis, -1)
    ncoils = imspace.shape[-1]
    ns = np.prod(imspace.shape[:-1])

    if mask is None:
        sos = np.sqrt(np.sum(np.abs(imspace)**2, axis=-1))
        thresh = threshold_li(sos)
        mask = (sos > thresh).flatten()
    else:
        mask = mask.flatten()
    assert mask.size == ns, 'mask must be the same size as a coil!'

    # Compute the sample auto-covariances pointwise, will be
    # Hermitian symmetric, only need lower triangular matrix
    Rs = np.empty((ncoils, ncoils, ns), dtype=imspace.dtype)
    for p in range(ncoils):
        for q in range(p):
            Rs[q, p, :] = (np.conj(
                imspace[..., p])*imspace[..., q]).flatten()

    # TODO:
    # # Smooth the covariance
    # for p in range(ncoils):
    #     for q in range(ncoils):
    #         Rs[p, q] = smooth(Rs[p, q, ...], smoothing)

    # At each point in the image, find the dominant eigenvector
    # and corresponding eigenvalue of the signal covariance
    # matrix using the power method
    csm = np.zeros((ns, ncoils), dtype=imspace.dtype)
    for ii in np.nonzero(mask)[0]:
        R = Rs[..., ii]
        v = eigh(R, lower=False,
                 eigvals=(ncoils-1, ncoils-1))[1].squeeze()
        csm[ii, :] = v/np.linalg.norm(v)

    return np.moveaxis(np.reshape(csm, imspace.shape), -1, coil_axis)
