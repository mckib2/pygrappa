'''Python implementation of Non-Cartesian GRAPPA.'''

import numpy as np

def ncgrappa(kx, ky, k, calib, coil_axis=-1):
    '''Non-Cartesian GRAPPA.

    Parameters
    ----------
    kx, ky : array_like
        k-space coordinates of kspace data, k.  kx and ky are 1D
        arrays.
    k : array_like
        Complex kspace data corresponding the measurements at
        locations kx, ky.  k has two dimensions: data and coil.  The
        coil dimension will be assumed to be last unless coil_axis=0.
    calib : array_like
        Cartesian calibration data, usually the fully sampled center
        of kspace.
    coil_axis : int, optional
        Dimension of calib holding coil data.

    Notes
    -----
    Implements to the algorithm described in [1]_.

    References
    ----------
    .. [1] Luo, Tianrui, et al. "A GRAPPA algorithm for arbitrary
           2D/3D non‐Cartesian sampling trajectories with rapid
           calibration." Magnetic resonance in medicine 82.3 (2019):
           1101-1112.
    '''

    # Assume k has coil at end unless user says it's upfront.  We
    # want to assume that coils are in the back of calib and k:
    calib = np.moveaxis(calib, coil_axis, -1)
    if coil_axis == 0:
        k = np.moveaxis(k, coil_axis, -1)

    # Identify all the constellations for calibration
