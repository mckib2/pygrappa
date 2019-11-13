'''Python implementation of the iGRAPPA algorithm.'''

import numpy as np
from tqdm import trange
from skimage.metrics import mean_squared_error as compare_mse # pylint: disable=E0611,E0401

from pygrappa import cgrappa

def igrappa(
        kspace, calib, kernel_size=(5, 5), k=0.3, coil_axis=-1,
        lamda=0.01, ref=None, niter=10, silent=True):
    '''Iterative GRAPPA.

    Parameters
    ----------
    kspace : array_like
        2D multi-coil k-space data to reconstruct from.  Make sure
        that the missing entries have exact zeros in them.
    calib : array_like
        Calibration data (fully sampled k-space).
    kernel_size : tuple, optional
        Size of the 2D GRAPPA kernel (kx, ky).
    k : float, optional
        Regularization parameter for iterative reconstruction.  Must
        be in the interval (0, 1).
    coil_axis : int, optional
        Dimension holding coil data.  The other two dimensions should
        be image size: (sx, sy).
    lamda : float, optional
        Tikhonov regularization for the kernel calibration.
    ref : array_like or None, optional
        Reference k-space data.  This is the true data that we are
        attempting to reconstruct.  If provided, MSE at each
        iteration will be returned.  If None, only reconstructed
        kspace is returned.
    niter : int, optional
        Number of iterations.
    silent : bool, optional
        Suppress messages to user.

    Returns
    -------
    res : array_like
        k-space data where missing entries have been filled in.
    mse : array_like, optional
        MSE at each iteration.  Returned if ref not None.

    Raises
    ------
    AssertionError
        If regularization parameter k is not in the interval (0, 1).

    Notes
    -----
    More or less implements the iterative algorithm described in [1].

    References
    ----------
    .. [1] Zhao, Tiejun, and Xiaoping Hu. "Iterative GRAPPA (iGRAPPA)
           for improved parallel imaging reconstruction." Magnetic
           Resonance in Medicine: An Official Journal of the
           International Society for Magnetic Resonance in Medicine
           59.4 (2008): 903-907.
    '''

    # Make sure k has a reasonable value
    assert 0 < k < 1, 'Parameter k should be in (0, 1)!'

    # Collect arguments to pass to cgrappa:
    grappa_args = {
        'kernel_size': kernel_size,
        'coil_axis': -1,
        'lamda': lamda,
        'silent': silent
    }

    # Put the coil dimension at the end
    kspace = np.moveaxis(kspace, coil_axis, -1)
    calib = np.moveaxis(calib, coil_axis, -1)
    kx, ky, _nc = kspace.shape[:]
    cx, cy, _nc = calib.shape[:]
    kx2, ky2 = int(kx/2), int(ky/2)
    cx2, cy2 = int(cx/2), int(cy/2)

    # Initial conditions
    kIm, W = cgrappa(kspace, calib, ret_weights=True, **grappa_args)
    ax = (0, 1)
    Im = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(
        kIm, axes=ax), axes=ax), axes=ax)
    Fp = 1e6 # some large number to begin with

    # If user has provided reference, let's track the MSE
    if ref is not None:
        mse = np.zeros(niter)
        aref = np.abs(ref)

    # Fixed number of iterations
    for ii in trange(niter, leave=False, desc='iGRAPPA'):

        # Update calibration region -- now includes all estimated
        # lines plus unchanged calibration region
        calib0 = kIm.copy()
        calib0[kx2-cx2:kx2+cx2, ky2-cy2:ky2+cy2, :] = calib.copy()

        kTm, Wn = cgrappa(
            kspace, calib0, ret_weights=True, **grappa_args)
        Tm = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(
            kTm, axes=ax), axes=ax), axes=ax)

        # Estimate relative image intensity change
        l1_Tm = np.linalg.norm(Tm.flatten(), ord=1)
        l1_Im = np.linalg.norm(Im.flatten(), ord=1)
        Tp = np.abs(l1_Tm - l1_Im)/l1_Im

        # Update weights
        p = Tp/(k*Fp)
        if p < 1:
            # Take this reconstruction and new weights
            Im = Tm
            kIm = kTm
            W = Wn
        else:
            # Modify weights to get new reconstruction
            p = 1/p
            W = [(1 - p)*Wn0 + p*W0 for Wn0, W0 in zip(Wn, W)]

            # Need to be able to supply grappa with weights to use!
            kIm = cgrappa(kspace, calib0, Wsupp=W, **grappa_args)
            Im = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(
                kIm, axes=ax), axes=ax), axes=ax)

        # Update Fp
        Fp = Tp

        # Track MSE
        if ref is not None:
            mse[ii] = compare_mse(aref, np.abs(kIm))

    # Return the reconstructed kspace and MSE if ref kspace provided,
    # otherwise, just return reconstruction
    if ref is not None:
        return(kIm, mse)
    return kIm
