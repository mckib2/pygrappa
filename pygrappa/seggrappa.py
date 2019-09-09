'''Python implementation of the Segmented GRAPPA algorithm.'''

import numpy as np

from pygrappa import cgrappa

def seggrappa(kspace, calibs, *args, **kwargs):
    '''Segmented GRAPPA.

    See pygrappa.grappa() for full list of arguments.

    Parameters
    ----------
    calibs : list of array_like
        List of calibration regions.

    Notes
    -----
    A generalized implementation of the method described in [1]_.
    Multiple ACS regions can be supplied to function.  GRAPPA is run
    for each ACS region and then averaged to produce the final
    reconstruction.

    References
    ----------
    .. [1] Park, Jaeseok, et al. "Artifact and noise suppression in
           GRAPPA imaging using improved k‐space coil calibration and
           variable density sampling." Magnetic Resonance in
           Medicine: An Official Journal of the International Society
           for Magnetic Resonance in Medicine 53.1 (2005): 186-193.
    '''

    # Do the reconstruction for each of the calibration regions
    recons = [cgrappa(kspace, c, *args, **kwargs) for c in calibs]

    # Average all the reconstructions
    return np.mean(recons, axis=0)
