'''Python implementation of iterative and CG-SENSE.'''

from time import time
import logging

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg


def _fft(x0, axes=None):
    '''Utility Forward FFT function.
    '''
    if axes is None:
        axes = np.arange(x0.ndim-1)
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(
        x0, axes=axes), axes=axes), axes=axes)


def _ifft(x0, axes=None):
    '''Utility Inverse FFT function.
    '''
    if axes is None:
        axes = np.arange(x0.ndim-1)
    return np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(
        x0, axes=axes), axes=axes), axes=axes)


def cgsense(kspace, sens, coil_axis=-1):
    '''Conjugate Gradient SENSE for arbitrary Cartesian acquisitions.

    Parameters
    ----------
    kspace : array_like
        Undersampled kspace data with exactly 0 in place of missing
        samples.
    sens : array_like or callable.
        Coil sensitivity maps or a function that generates them with
        the following function signature:

            sens(kspace: np.ndarray, coil_axis: int) -> np.ndarray

    coil_axis : int, optional
        Dimension of kspace and sens holding the coil data.

    Returns
    -------
    res : array_like
        Single coil unaliased estimate (imspace).

    Notes
    -----
    Implements a Cartesian version of the iterative algorithm
    described in [1]_.  It can handle arbitrary undersampling of
    Cartesian acquisitions and arbitrarily-dimensional
    datasets.  All dimensions except ``coil_axis`` will be used
    for reconstruction.

    This implementation uses the scipy.sparse.linalg.cg() conjugate
    gradient algorithm to solve A^H A x = A^H b.

    References
    ----------
    .. [1] Pruessmann, Klaas P., et al. "Advances in sensitivity
           encoding with arbitrary k‐space trajectories." Magnetic
           Resonance in Medicine: An Official Journal of the
           International Society for Magnetic Resonance in Medicine
           46.4 (2001): 638-651.
    '''
    # Make sure coils are in the back
    kspace = np.moveaxis(kspace, coil_axis, -1)

    # Generate the coil sensitivities
    if callable(sens):
        imspace = _ifft(kspace)
        sens = sens(imspace, coil_axis=coil_axis)

    sens = np.moveaxis(sens, coil_axis, -1)
    tipe = kspace.dtype

    # Get the sampling mask:
    dims = kspace.shape[:-1]
    mask = np.abs(kspace[..., 0]) > 0

    # We are solving Ax = b where A takes the unaliased single coil
    # image x to the undersampled kspace data, b.  Since A is usually
    # not square we'd need to use lsqr/lsmr which can take a while
    # and won't give great results.  So we can make a sqaure encoding
    # matrix like this:
    #     Ax = b
    #     A^H A x = A^H b
    #     E = A^H A is square!
    # So now we can solve using scipy's cg() method which luckily
    # accepts complex inputs!  We will need to represent our data
    # and encoding matrices as vectors and matrices:
    #     A : (sx*sy*nc, sx*sy)
    #     x : (sx*sy,)
    #     b : (sx*sy*nc,)
    # => E : (sx*sy, sx*sy)

    def _AH(x0):
        '''kspace -> imspace'''
        x0 = np.reshape(x0, kspace.shape)
        res = np.sum(sens.conj()*_ifft(x0), axis=-1)
        return np.reshape(res, (-1,))

    def _A(x0):
        '''imspace -> kspace'''
        res = np.reshape(x0, dims)
        res = _fft(res[..., None]*sens)*mask[..., None]
        return np.reshape(res, (-1,))

    # Make LinearOperator, A^H b, and use CG to solve
    def E(x0):
        return _AH(_A(x0))
    AHA = LinearOperator(
        (np.prod(dims), np.prod(dims)),
        matvec=E, rmatvec=E)
    b = _AH(np.reshape(kspace, (-1,)))

    t0 = time()
    x, _info = cg(AHA, b, atol=0)
    logging.info('CG-SENSE took %g sec', (time() - t0))

    return np.reshape(x, dims).astype(tipe)


if __name__ == '__main__':
    pass
