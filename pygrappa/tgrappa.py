'''TGRAPPA implementation.'''

import numpy as np

def tgrappa(kspace, kernel_size=(5, 5), coil_axis=-2, time_axis=-1):
    '''Temporal GRAPPA.

    Parameters
    ----------
    kspace : array_like
        2+1D multi-coil k-space data to reconstruct from (total of
        4 dimensions).  Missing entries should have exact zeros in
        them.
    kernel_size : tuple, optional
        Desired shape of the in-plane calibration regions: (kx, ky).
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
    kspace = np.moveaxis(kspace, (coil_axis, time_axis), (-2, -1))
    sx, sy, _sc, _st = kspace.shape[:]
    kx, ky = kernel_size[:]
    sx2, sy2 = int(sx/2), int(sy/2)
    kx2, ky2 = int(kx/2), int(ky/2)
    adjx, adjy = int(np.mod(kx, 2)), int(np.mod(ky, 2))

    # First we need to find out if the calibration regions are even
    # feasible, that is, if data can be found in each time frame to
    # synthesize an ACS without any holes


    # Find the first feasible kernel -- Strategy: consume time frames
    # until all kernel elements have been filled.  We won't assume
    # that overlaps will not happen frame to frame, so we will
    # appropriately average each kernel position.
    got_kernel = False
    calib = []
    tt = 0 # time frame index
    filled = np.zeros((kx, ky), dtype=bool)
    while not got_kernel:

        # Might need to consider odd/even kernel sizes
        calib.append(kspace[
            sx2-kx2:sx2+kx2+adjx, sy2-ky2:sy2+ky2+adjy, :, tt].copy())
        filled = np.logical_or(filled, np.abs(calib[tt][..., 0]) > 0)
        if np.all(filled):
            got_kernel = True
        tt += 1

    # Now average over all time frames to get a single ACS
    calib =  np.array(calib) # time axis is in front
    print(calib.shape)

    return 0
