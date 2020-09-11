'''Automated location of a rectangular ACS.'''

from time import time
import logging

import numpy as np


def find_acs(kspace, coil_axis=-1):
    '''Find the largest centered hyper-rectangle possible.

    Parameters
    ----------
    kspace : array_like
        Measured undersampled complex k-space data. N-1 dimensions
        hold spatial frequency axes (kx, ky, kz, etc.).  1 dimension
        holds coil images (`coil_axis`).  The missing entries should
        have exactly 0.
    coil_axis : int, optional
        Dimension holding coil images.

    Returns
    -------
    calib : array_like
        Fully sampled calibration data extracted from the largest
        possible hypercube with origin at the center of k-space.

    Notes
    -----
    This algorithm is not especially elegant, but works just fine
    with the assumption that the ACS region will be significantly
    smaller than the entirety of the data.  It grows a hyper-
    rectangle from the center and checks to see if there are any
    new holes in the region each time it expands.
    '''

    kspace = np.moveaxis(kspace, coil_axis, -1)
    mask = np.abs(kspace[..., 0]) > 0

    # Start by finding the largest hypercube
    ctrs = [d // 2 for d in mask.shape]  # assume ACS is at center
    slices = [[c, c+1] for c in ctrs]  # start with 1 voxel region
    t0 = time()
    while (all(l > 0 and r < mask.shape[ii] for
               ii, (l, r) in enumerate(slices)) and  # bounds check
           np.all(mask[tuple([slice(l-1, r+1) for
                              l, r in slices])])):  # hole check
        # expand isotropically until we can't no more
        slices = [[l0-1, r0+1] for l0, r0 in slices]
    logging.info('Took %g sec to find hyper-cube', (time() - t0))

    # FOR DEBUG:
    # region = np.zeros(mask.shape, dtype=bool)
    # region[tuple([slice(l, r) for l, r in slices])] = True
    # import matplotlib.pyplot as plt
    # plt.imshow(region[..., 20])
    # plt.show()

    # Stretch left/right in each dimension
    t0 = time()
    for dim in range(mask.ndim):
        # left: only check left condition on the current dimension
        while (slices[dim][0] > 0 and
               np.all(mask[tuple([slice(l-(dim == k), r) for
                                  k, (l, r) in enumerate(slices)])])):
            slices[dim][0] -= 1
        # right: only check right condition on the current dimension
        while (slices[dim][1] < mask.shape[dim] and
               np.all(mask[tuple([slice(l, r+(dim == k)) for
                                  k, (l, r) in enumerate(slices)])])):
            slices[dim][1] += 1
    logging.info('Took %g sec to find hyper-rect', (time() - t0))

    return np.moveaxis(
        kspace[tuple([slice(l0, r0) for l0, r0 in slices] +
                     [slice(None)])].copy(),  # extra dim for coils
        -1, coil_axis)
