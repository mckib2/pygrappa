'''Python implementation of VC-GRAPPA.'''

import numpy as np

from pygrappa import cgrappa as grappa


def vcgrappa(kspace, calib, *args, coil_axis=-1, **kwargs):
    '''Virtual Coil GRAPPA.

    See pygrappa.grappa() for argument list.

    Notes
    -----
    Implements modifications to GRAPPA as described in [1]_.  The
    only change I can see is stacking the conjugate coils in the
    coil dimension.  For best results, make sure there is a suitably
    chosen background phase variation as described in the paper.

    This function is a wrapper of pygrappa.cgrappa().  The existing
    coils are conjugated, added to the coil dimension, and passed
    through along with all other arguments.

    References
    ----------
    .. [1] Blaimer, Martin, et al. "Virtual coil concept for improved
           parallel MRI employing conjugate symmetric signals."
           Magnetic Resonance in Medicine: An Official Journal of the
           International Society for Magnetic Resonance in Medicine
           61.1 (2009): 93-102.
    '''

    # Move coil axis to end
    kspace = np.moveaxis(kspace, coil_axis, -1)
    calib = np.moveaxis(calib, coil_axis, -1)
    ax = (0, 1)

    # remember the type we started out with, np.fft will change
    # to complex128 regardless of what we started with
    tipe = kspace.dtype

    # We will return twice the number of coils we started with
    nc = kspace.shape[-1]

    # In and out of kspace to get conjugate coils
    vc_kspace = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(
        kspace, axes=ax), axes=ax), axes=ax)
    vc_kspace = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(
        np.conj(vc_kspace), axes=ax), axes=ax), axes=ax)

    # Same deal for calib...
    vc_calib = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(
        calib, axes=ax), axes=ax), axes=ax)
    vc_calib = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(
        np.conj(vc_calib), axes=ax), axes=ax), axes=ax)

    # Put all our ducks in a row...
    kspace = np.concatenate((kspace, vc_kspace), axis=-1)
    calib = np.concatenate((calib, vc_calib), axis=-1)

    # Pass through to GRAPPA
    return grappa(
        kspace, calib, coil_axis=-1, nc_desired=2*nc,
        *args, **kwargs).astype(tipe)


if __name__ == '__main__':
    pass
