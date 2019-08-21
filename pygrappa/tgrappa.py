'''TGRAPPA implementation.'''

import numpy as np
from tqdm import tqdm

# from pygrappa import grappa
from pygrappa import cgrappa as grappa # need for speed!

def tgrappa(
        kspace, calib_size=(20, 20), kernel_size=(5, 5),
        coil_axis=-2, time_axis=-1):
    '''Temporal GRAPPA.

    Parameters
    ----------
    kspace : array_like
        2+1D multi-coil k-space data to reconstruct from (total of
        4 dimensions).  Missing entries should have exact zeros in
        them.
    calib_size : array_like, optional
        Size of calibration region at the center of kspace.
    kernel_size : tuple, optional
        Desired shape of the in-plane calibration regions: (kx, ky).
    coil_axis : int, optional
        Dimension holding coil data.
    time_axis : int, optional
        Dimension holding time data.

    Returns
    -------
    res : array_like
        Reconstructed k-space data.

    Raises
    ------
    ValueError
        When no complete ACS region can be found.

    Notes
    -----
    Implementation of the method proposed in [1]_.

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
    sx, sy, _sc, st = kspace.shape[:]
    cx, cy = calib_size[:]
    sx2, sy2 = int(sx/2), int(sy/2)
    cx2, cy2 = int(cx/2), int(cy/2)
    adjx, adjy = int(np.mod(cx, 2)), int(np.mod(cy, 2))

    # Make sure that it's even possible to create complete ACS, we'll
    # have problems in the loop if we don't catch it here!
    if not np.all(np.sum(np.abs(kspace[
            sx2-cx2:sx2+cx2+adjx,
            sy2-cy2:sy2+cy2+adjy, ...]), axis=-1)):
        raise ValueError('Full ACS region cannot be found!')

    # To avoid running GRAPPA more than once on one time frame,
    # we'll keep track of which frames have been reconstruced:
    completed_tframes = np.zeros(st, dtype=bool)

    # Initialize the progress bar
    pbar = tqdm(total=st, leave=False, desc='TGRAPPA')

    # Iterate through all time frames, construct ACS regions, and
    # run GRAPPA on each time slice
    res = np.zeros(kspace.shape, dtype=kspace.dtype)
    tt = 0 # time frame index
    done = False # True when all time frames have been consumed
    from_end = False # start at the end and go backwards
    while not done:

        # Find next feasible kernel -- Strategy: consume time frames
        # until all kernel elements have been filled.  We won't
        # assume that overlaps will not happen frame to frame, so we
        # will appropriately average each kernel position by keeping
        # track of how many samples are in a position with 'counts'
        got_kernel = False
        calib = []
        counts = np.zeros((cx, cy), dtype=int)
        tframes = [] # time frames over which the ACS is valid
        while not got_kernel:
            if not completed_tframes[tt]:
                tframes.append(tt)
            calib.append(kspace[
                sx2-cx2:sx2+cx2+adjx,
                sy2-cy2:sy2+cy2+adjy, :, tt].copy())
            counts += np.abs(calib[-1][..., 0]) > 0
            if np.all(counts > 0):
                got_kernel = True

            # Go to next time frame except maybe for the last ACS
            if not from_end:
                tt += 1 # Consume the next time frame
            else:
                tt -= 1 # Consume previous time frame

            # If we need more time frames than we have, then we need
            # to start from the end and come forward.  This can only
            # happen on the last iteration of the outer loop
            if not got_kernel and tt == st:
                # Start at the end
                tt = st-1

                # Reset ACS, counts, and tframes
                calib = []
                counts = np.zeros((cx, cy), dtype=int)
                tframes = []

                # Let the loop know we want to reverse directions
                from_end = True

        # Now average over all time frames to get a single ACS
        calib = np.sum(calib, axis=0)/counts[..., None]

        # This ACS region is valid over all time frames used to
        # create it.  Run GRAPPA on each valid time frame with calib:
        for t0 in tframes:
            res[..., t0] = grappa(
                kspace[..., t0], calib, kernel_size)
            completed_tframes[t0] = True
            pbar.update(1)

        # Stopping condition: end when all time frames are consumed
        if np.all(completed_tframes):
            done = True

    # Close out the progress bar
    pbar.close()

    # Move axes back to where the user had them
    return np.moveaxis(res, (-1, -2), (time_axis, coil_axis))
