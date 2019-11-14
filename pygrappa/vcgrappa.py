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

    # We only need to return the number of coils we are provided
    nc = kspace.shape[-1]

    # Create conjugate virtual coils for kspace using properties
    # of Fourier transform
    vc_kspace = np.rot90(np.conj(kspace), 2)
    vc_calib = np.rot90(np.conj(calib), 2)
    kspace = np.concatenate((kspace, vc_kspace), axis=-1)
    calib = np.concatenate((calib, vc_calib), axis=-1)

    # Pass through to GRAPPA
    return grappa(
        kspace, calib, coil_axis=-1, nc_desired=nc, *args, **kwargs)

if __name__ == '__main__':
    pass
