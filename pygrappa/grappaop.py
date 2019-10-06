'''Python implementation of the GRAPPA operator formalism.'''

import numpy as np

def grappaop(calib, coil_axis=-1, lamda=0.01):
    '''GRAPPA operator for Cartesian calibration datasets.

    Parameters
    ----------
    calib : array_like
        Calibration region data.  Usually a small portion from the
        center of kspace.
    coil_axis : int, optional
        Dimension holding coil data.
    lamda : float, optional
        Tikhonov regularization parameter.  Set to 0 for no
        regularization.

    Returns
    -------
    Gx, Gy : array_like
        GRAPPA operators for both the x and y directions.

    Notes
    -----
    Produces the unit operator described in [1]_.

    This seems to only work well when coil sensitivities are very
    well separated/distinct.  If coil sensitivities are similar,
    operators perform poorly.

    References
    ----------
    .. [1] Griswold, Mark A., et al. "Parallel magnetic resonance
           imaging using the GRAPPA operator formalism." Magnetic
           resonance in medicine 54.6 (2005): 1553-1556.
    '''

    # Coil axis in the back
    calib = np.moveaxis(calib, coil_axis, -1)
    _cx, _cy, nc = calib.shape[:]

    # We need sources (last source has no target!)
    Sx = np.reshape(calib[:-1, ...], (-1, nc))
    Sy = np.reshape(calib[:, :-1, :], (-1, nc))

    # And we need targets for an operator along each axis (first
    # target has no associated source!)
    Tx = np.reshape(calib[1:, ...], (-1, nc))
    Ty = np.reshape(calib[:, 1:, :], (-1, nc))

    # Train the operators:
    Sxh = Sx.conj().T
    lamda0 = lamda*np.linalg.norm(Sxh)/Sxh.shape[0]
    Gx = np.linalg.solve(
        Sxh @ Sx + lamda0*np.eye(Sxh.shape[0]), Sxh @ Tx)

    Syh = Sy.conj().T
    lamda0 = lamda*np.linalg.norm(Syh)/Syh.shape[0]
    Gy = np.linalg.solve(
        Syh @ Sy + lamda0*np.eye(Syh.shape[0]), Syh @ Ty)
    return(Gx, Gy)
