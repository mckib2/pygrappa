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
    coil dimension.

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

    # Create conjugate virtual coils for kspace
    ax = (0, 1)
    vc_kspace = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(
        kspace, axes=ax), axes=ax), axes=ax)
    vc_kspace = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(
        np.conj(vc_kspace), axes=ax), axes=ax), axes=ax)
    # Make sure zeros stay the same through FFTs:
    vc_kspace = vc_kspace*(np.abs(kspace) > 0)
    kspace = np.concatenate((kspace, vc_kspace), axis=-1)

    # Create conjugate virtual coils for calib
    vc_calib = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(
        calib, axes=ax), axes=ax), axes=ax)
    vc_calib = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(
        np.conj(vc_calib), axes=ax), axes=ax), axes=ax)
    calib = np.concatenate((calib, vc_calib), axis=-1)

    # Pass through to GRAPPA
    return grappa(kspace, calib, coil_axis=-1, *args, **kwargs)

if __name__ == '__main__':
    pass
