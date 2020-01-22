'''Python implementation of SENSE.'''

from time import time

import numpy as np

def sense1d(im, sens, Rx=1, Ry=1, coil_axis=-1, imspace=True):
    '''Sensitivity Encoding for Fast MRI (SENSE) along one dimension.

    Parameters
    ----------
    im : array_like
        Array of the aliased 2D multicoil coil image. If
        imspace=False, im is the undersampled k-space data.
    sens : array_like
        Complex coil sensitivity maps with the same dimensions as im.
    Rx, Ry : ints, optional
        Acceleration factor in x and y.  One of Rx, Ry must be 1.  If
        both are 1, then this is Roemer's optimal coil combination.
    coil_axis : int, optional
        Dimension holding coil data.
    imspace : bool, optional
        If im is image space or k-space data.

    Returns
    -------
    res : array_like
        Unwrapped single coil reconstruction.

    Notes
    -----
    Implements the algorithm first described in [1]_.  This
    implementation is based on the MATLAB tutorial found in [2]_.

    This implementation handles only regular undersampling along a
    single dimension.  Arbitrary undersampling is not supported by
    this function.

    Odd Rx, Ry seem to behave strangely, i.e. not as well as even
    factors.  Right now I'm padding im and sens by 1 and removing at
    end.

    References
    ----------
    .. [1] Pruessmann, Klaas P., et al. "SENSE: sensitivity encoding
           for fast MRI." Magnetic Resonance in Medicine: An Official
           Journal of the International Society for Magnetic
           Resonance in Medicine 42.5 (1999): 952-962.
    .. [2] https://users.fmrib.ox.ac.uk/~mchiew/docs/
           SENSE_tutorial.html
    '''

    # We can only handle unwrapping one dimension:
    assert Rx == 1 or Ry == 1, 'One of Rx, Ry must be 1!'

    # Coils to da back
    im = np.moveaxis(im, coil_axis, -1)
    sens = np.moveaxis(sens, coil_axis, -1)

    # Assume the first dimension has the unwrapping, so move the
    # axis we want to operate on to the front
    flip_xy = False
    if Ry > 1:
        flip_xy = True
        im = np.moveaxis(im, 0, 1)
        sens = np.moveaxis(sens, 0, 1)
        Rx, Ry = Ry, Rx

    # Put kspace into image space if needed
    if not imspace:
        im = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(
            im, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))

    # If undersampling factor is odd, pad the image
    if Rx % 2 > 0:
        im = np.pad(im, ((0, 1), (0, 0), (0, 0)))
        sens = np.pad(sens, ((0, 1), (0, 0), (0, 0)))

    nx, ny, _nc = im.shape[:]
    res = np.zeros((nx, ny), dtype=im.dtype)

    # loop over the top 1/R of the image, use einsum to get all the
    # inner loops where the subproblems are extracted and solved
    # in the least squares sense
    t0 = time()
    for x in range(int(nx/Rx)):
        x_idx = np.arange(x, nx, step=int(nx/Rx))
        S = sens[x_idx, ...].transpose((1, 2, 0))

        # Might be more efficient way then explicit pinv along axis?
        res[x_idx, :] = np.einsum(
            'ijk,ik->ij', np.linalg.pinv(S), im[x, ...]).T
    print('Took %g sec for unwrapping' % (time() - t0))

    # Remove pad if Rx is odd
    if Rx % 2 > 0:
        res = res[:-1, ...]

    # Put all the axes back where the user had them
    if flip_xy:
        res = np.moveaxis(res, 1, 0)
    return np.moveaxis(res, -1, coil_axis)

if __name__ == '__main__':
    pass
