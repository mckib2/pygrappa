'''Python implementation of the iGRAPPA algorithm.'''

import numpy as np

from pygrappa import cgrappa

def igrappa(kspace, calib, k=0.6, coil_axis=-1):
    '''Iterative GRAPPA.

    Notes
    -----
    Implements the two-stage algorithm described in [1].

    References
    ----------
    .. [1] Zhao, Tiejun, and Xiaoping Hu. "Iterative GRAPPA (iGRAPPA)
           for improved parallel imaging reconstruction." Magnetic
           Resonance in Medicine: An Official Journal of the
           International Society for Magnetic Resonance in Medicine
           59.4 (2008): 903-907.
    '''

    # Make sure 
    assert 0 < k < 1, 'k should be in (0, 1)!'

    # Put the coil dimension at the end
    kspace = np.moveaxis(kspace, coil_axis, -1)
    calib = np.moveaxis(calib, coil_axis, -1)

    # Stage 1: "Using calibration data to obtain the initial weights,
    # followed by iterative GRAPPA reconstruction with the
    # restriction that only continuously measured lines in the
    # middle of k‐space are used on the left side of Eq. [1]."

    # Initial conditions
    Im, W = cgrappa(kspace, calib, coil_axis=-1, ret_weights=True)
    Fp = 1e6

    niter = 4
    for _ii in range(niter):

        Tm, Wn = cgrappa(
            kspace, calib, coil_axis=-1, ret_weights=True)

        # Estimate relative image intensity change
        Tp = np.sum((np.abs(Tm) - np.abs(Im)).flatten())
        Tp /= np.sum(np.abs(Im).flatten())

        # Update weights
        p = Tp/(k*Fp)
        if p < 1:
            # Take this reconstruction and new weights
            Im = Tm
            W = Wn
        else:
            # Modify weights to get new reconstruction
            p = 1/p
            W = (1 - p)*Wn + p*W

            # Need to be able to supply grappa with weights to use!
            # Im = grappa_with_supplied_weights(W)

        # Update Fp
        Fp = Tp

        assert False
