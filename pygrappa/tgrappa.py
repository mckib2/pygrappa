'''TGRAPPA implementation.'''

import numpy as np

def tgrappa(kspace, calib_size, coil_axis=-1, time_axis=-2):
    '''Temporal GRAPPA.

    Parameters
    ----------
    kspace : array_like
        2+1D multi-coil k-space data to reconstruct from (total of
        4 dimensions).  Missing entries should have exact zeros in
        them.
    calib_size : tuple
        Desired shape of the in-plane calibration regions: (cx, cy).
    coil_axis : int, optional
        Dimension holding coil data.
    time_axis : int, optional
        Dimension holding time data.

    Notes
    -----
    Implementation of the method presented in [1]_.

    The idea is to form ACS regions using data from adjacent time
    frames.  For example, in the case of 1D undersampling using
    undersampling factor R, at least R time frames must be merged to
    form a completely sampled ACS.  Then we can simply supply the
    undersampled data and the synthesized ACS to GRAPPA.  Thus the
    heavy lifting of this function will be in determining the ACS
    calibration region at each time frame.

    References
    ----------
    .. [1] Breuer, Felix A., et al. "Dynamic autocalibrated parallel
           imaging using temporal GRAPPA (TGRAPPA)." Magnetic
           Resonance in Medicine: An Official Journal of the
           International Society for Magnetic Resonance in Medicine
           53.4 (2005): 981-985.
    '''

    # Move coil and time axes to a place we can find them
    kspace = np.moveaxis(kspace, (coil_axis, time_axis), (-1, -2))

    # First we need to find out if the calibration regions are even
    # feasible, that is, if data can be found in each time frame to
    # synthesize an ACS without any holes
