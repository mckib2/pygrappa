'''Python implementation of Non-Linear GRAPPA.'''

from functools import partial

import numpy as np
from pygrappa import cgrappa
from pygrappa.kernels import polynomial_kernel

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
        Kernel functions modeled on scikit-learn metrics.pairwise
        module but which can handle complex-valued inputs.
    ml_kernel_args : dict or None, optional
        Arguments to pass to kernel functions.
    coil_axis : int, optional
        Axis holding the coil data.

    Returns
    -------
    res : array_like
        Reconstructed k-space.

    Notes
    -----
    Implements the algorithm described in [1]_.

    Bias term is removed from polynomial kernel as it adds a PSF-like
    overlay onto the reconstruction.

    Currently only `polynomial` method is implemented.

    References
    ----------
    .. [1] Chang, Yuchou, Dong Liang, and Leslie Ying. "Nonlinear
           GRAPPA: A kernel approach to parallel MRI reconstruction."
           Magnetic resonance in medicine 68.3 (2012): 730-740.
    '''

    # Coils to the back
    kspace = np.moveaxis(kspace, coil_axis, -1)
    calib = np.moveaxis(calib, coil_axis, -1)
    _kx, _ky, nc = kspace.shape[:]

    # Get the correct kernel:
    _phi = {
        # 'linear': linear_kernel,
        'polynomial': polynomial_kernel,
        # 'sigmoid': sigmoid_kernel,
        # 'rbf': rbf_kernel,
        # 'laplacian': laplacian_kernel,
        # 'chi2': chi2_kernel,
    }[ml_kernel]

    # Get default args if none were passed in
    if ml_kernel_args is None:
        ml_kernel_args = {
            'cross_term_neighbors': 1,
        }

    # Pass arguments to kernel function
    phi = partial(_phi, **ml_kernel_args)

    # Get the extra "virtual" channels using kernel function, phi
    vkspace = phi(kspace)
    vcalib = phi(calib)

    # Pass onto cgrappa for the heavy lifting
    return np.moveaxis(
        cgrappa(
            vkspace, vcalib, kernel_size=kernel_size, coil_axis=-1,
            nc_desired=nc, lamda=0),
        -1, coil_axis)
