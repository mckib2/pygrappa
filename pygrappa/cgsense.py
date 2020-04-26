'''Python implementation of iterative and CG-SENSE.'''

from time import time

import numpy as np
from scipy.sparse.linalg import LinearOperator, lsmr, cg


def fft2(x0, axes=(0, 1)):
    '''Utility Forward FFT function.

    :meta private:
    '''
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(
        x0, axes=axes), axes=axes), axes=axes)


def ifft2(x0, axes=(0, 1)):
    '''Utility Inverse FFT function.

    :meta private:
    '''
    return np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(
        x0, axes=axes), axes=axes), axes=axes)


def _isense(kspace, sens, show=False):
    '''Iterative SENSE using nonsquare matrix and LSMR solver.'''

    sx, sy, nc = kspace.shape[:]
    mask = np.abs(kspace[..., 0]) > 0

    # We need to implement an E and E.H operator
    def _EH(x0):
        x0 = x0[:sx*sy*nc] + 1j*x0[sx*sy*nc:]
        x0 = np.reshape(x0, (sx, sy, nc))
        res = np.sum(sens.conj()*ifft2(x0), axis=-1)
        res = np.reshape(res, (-1,))
        return np.concatenate((res.real, res.imag))

    def _E(x0):
        res = x0[:sx*sy] + 1j*x0[sx*sy:]
        res = np.reshape(res, (sx, sy))
        res = fft2(res[..., None]*sens)*mask[..., None]
        res = np.reshape(res, (-1,))
        return np.concatenate((res.real, res.imag))

    # Ax = b
    # A : (2*sx*sy*nc, 2*sx*sy)
    # x : (2*sx*sy)
    # b : (2*sx*sy*nc)
    # Twice the size since complex, do simple real/imag stacking
    A = LinearOperator((2*sx*sy*nc, 2*sx*sy), matvec=_E, rmatvec=_EH)
    b = np.reshape(kspace, (-1,))
    b = np.concatenate((b.real, b.imag))
    x = lsmr(A, b, show=show)[0]
    x = x[:sx*sy] + 1j*x[sx*sy:]

    return np.reshape(x, (sx, sy))


def _isense2(kspace, sens, show=False):
    '''Try LSMR with square matrix.'''

    sx, sy, nc = kspace.shape[:]
    mask = np.abs(kspace[..., 0]) > 0

    def _AH(x0):
        x0 = x0[:sx*sy*nc] + 1j*x0[sx*sy*nc:]
        x0 = np.reshape(x0, (sx, sy, nc))
        res = np.sum(sens.conj()*ifft2(x0), axis=-1)
        res = np.reshape(res, (-1,))
        return np.concatenate((res.real, res.imag))

    def _A(x0):
        res = x0[:sx*sy] + 1j*x0[sx*sy:]
        res = np.reshape(res, (sx, sy))
        res = fft2(res[..., None]*sens)*mask[..., None]
        res = np.reshape(res, (-1,))
        return np.concatenate((res.real, res.imag))

    def E(x0):
        return _AH(_A(x0))
    AHA = LinearOperator((sx*sy, sx*sy), matvec=E, rmatvec=E)
    b = np.reshape(kspace, (-1,))
    b = np.concatenate((b.real, b.imag))
    b = _AH(b)
    x = lsmr(AHA, b, show=show)[0]
    x = x[:sx*sy] + 1j*x[sx*sy:]

    return np.reshape(x, (sx, sy))


def cgsense(kspace, sens, coil_axis=-1):
    '''Conjugate Gradient SENSE for arbitrary Cartesian acquisitions.

    Parameters
    ----------
    kspace : array_like
        Undersampled kspace data with exactly 0 in place of missing
        samples.
    sens : array_like
        Coil sensitivity maps.
    coil_axis : int, optional
        Dimension of kspace and sens holding the coil data.

    Returns
    -------
    res : array_like
        Single coil unaliased estimate.

    Notes
    -----
    Implements a Cartesian version of the iterative algorithm
    described in [1]_.  It can handle arbitrary undersampling of
    Cartesian acquisitions.

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
    sens = np.moveaxis(sens, coil_axis, -1)

    # Get the sampling mask:
    sx, sy, nc = kspace.shape[:]
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
        x0 = np.reshape(x0, (sx, sy, nc))
        res = np.sum(sens.conj()*ifft2(x0), axis=-1)
        return np.reshape(res, (-1,))

    def _A(x0):
        '''imspace -> kspace'''
        res = np.reshape(x0, (sx, sy))
        res = fft2(res[..., None]*sens)*mask[..., None]
        return np.reshape(res, (-1,))

    # Make LinearOperator, A^H b, and use CG to solve
    def E(x0):
        return _AH(_A(x0))
    AHA = LinearOperator((sx*sy, sx*sy), matvec=E, rmatvec=E)
    b = _AH(np.reshape(kspace, (-1,)))

    t0 = time()
    x, _info = cg(AHA, b)
    print('CG-SENSE took %g sec' % (time() - t0))

    return np.reshape(x, (sx, sy))


if __name__ == '__main__':
    pass
