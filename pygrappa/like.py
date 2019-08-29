'''Python implementation of the LIKE algorithm.'''

# import numpy as np

from pygrappa import cgrappa

def like(kspace, calib, coil_axis=-1):
    '''Linear Interpolation in K-spacE (LIKE).

    Parameters
    ----------

    Returns
    -------

    Notes
    -----
    Implements the algorithm described in [1]_ for auto-calibrating
    parallel image reconstruction with extremely small ACS regions,
    e.g., 1 or 2 lines.

    References
    ----------
    .. [1] Huang, F., et al. "Linear interpolation in k-space."
           Proceedings of the 12th Annual Meeting of the ISMRM,
           Kyoto. 2004.
    '''

    # # We want the coil axis at the end
    # kspace = np.moveaxis(kspace, coil_axis, -1)
    # calib = np.moveaxis(calib, coil_axis, -1)

    # Step 1: GRAPPA
    # col = grappa(kspace, )
    calib0 = kspace.copy()
    sx, _sy, _nc = kspace.shape[:]
    sx2 = int(sx/2)
    calib0[sx2-1:sx2+2, ...] = calib
    row = cgrappa(
        kspace, calib0, kernel_size=(2, 7), coil_axis=coil_axis)
    col = cgrappa(
        kspace, calib0, kernel_size=(4, 4), coil_axis=coil_axis)

    # # Step 2: Consider all acquired to be ACS lines
    # return cgrappa(kspace, res, kernel_size=kernel_size)

    return (row + col)/2
