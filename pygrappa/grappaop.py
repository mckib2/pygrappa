'''Python implementation of the GRAPPA operator formalism.'''

import numpy as np

def grappaop(calib, coil_axis=-1):
    '''GRAPPA operator.

    Parameters
    ----------

    Returns
    -------

    Notes
    -----
    Produces the infinitesimal operator described in [1]_.

    References
    ----------
    .. [1] Griswold, Mark A., et al. "Parallel magnetic resonance
           imaging using the GRAPPA operator formalism." Magnetic
           resonance in medicine 54.6 (2005): 1553-1556.
    '''

    # Coil axis in the back
    calib = np.moveaxis(calib, coil_axis, -1)
    _cx, _cy, nc = calib.shape[:]

    # consider a single dimension of calibration data:
    #     T = S G
    #     S^H T = S^H S G
    #     (S^H S)^-1 S^H T = G

    # We need sources and targets
    dim = 'C' # use 'F' for other direction
    S = np.reshape(calib, (-1, nc), order=dim)

    # Could be (0, 1, 0)?
    T = np.roll(calib, (1, 0, 0)).reshape((-1, nc), order=dim)

    ShS = S.conj().T @ S
    ShT = S.conj().T @ T
    G = np.linalg.solve(ShS, ShT)

    return G
