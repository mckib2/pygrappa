'''Python implementation of Non-Linear GRAPPA.'''

from functools import partial

import numpy as np
from sklearn.metrics import pairwise # where the kernels live
from pygrappa import cgrappa

def nlgrappa(
        kspace, calib, kernel_size=(5, 5), ml_kernel='polynomial',
        ml_kernel_args=None, coil_axis=-1):
    '''NL-GRAPPA.

    Parameters
    ----------
    kspace : array_like
    calib : array_like
    kernel_size : tuple of int, optional
    ml_kernel : {
            'linear', 'polynomial', 'sigmoid', 'rbf',
            'laplacian', 'chi2'}, optional
        Kernel functions from scikit-learn metrics.pairwise module.
    ml_kernel_args : dict or None, optional
        Arguments to pass to kernel functions.
    coil_axis : int, optional
        Axis holding the coil data.

    Returns
    -------

    Notes
    -----
    Implements the algorithm described in [1]_.

    References
    ----------
    .. [1] Chang, Yuchou, Dong Liang, and Leslie Ying. "Nonlinear
           GRAPPA: A kernel approach to parallel MRI reconstruction."
           Magnetic resonance in medicine 68.3 (2012): 730-740.
    '''

    # Coils to the back
    kspace = np.moveaxis(kspace, coil_axis, -1)
    calib = np.moveaxis(calib, coil_axis, -1)
    _kx, _ky, _nc = kspace.shape[:]

    # Get the correct kernel:
    _phi = {
        'linear': pairwise.linear_kernel,
        'polynomial': pairwise.polynomial_kernel,
        'sigmoid': pairwise.sigmoid_kernel,
        'rbf': pairwise.rbf_kernel,
        'laplacian': pairwise.laplacian_kernel,
        'chi2': pairwise.chi2_kernel,
    }[ml_kernel]

    # Get default args if none were passed in
    if ml_kernel_args is None:
        ml_kernel_args = {
            'gamma': 1,
            'coef0': 1,
            'degree': 2
        }

    # Pass arguments to kernel function
    phi = partial(_phi, **ml_kernel_args)

    # Get the extra "virtual" channels using kernel function, phi

    # Pass onto cgrappa for the heavy lifting
    return cgrappa(
        kspace, calib, kernel_size=kernel_size, coil_axis=-1)
