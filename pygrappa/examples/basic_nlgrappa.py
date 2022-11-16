'''Basic usage of NL-GRAPPA.'''

import numpy as np
import matplotlib.pyplot as plt
try:
    from skimage.metrics import normalized_root_mse as compare_nrmse  # pylint: disable=E0611,E0401
except ImportError:
    from skimage.measure import compare_nrmse
from phantominator import shepp_logan

from pygrappa import nlgrappa, cgrappa
from pygrappa.utils import gaussian_csm

if __name__ == '__main__':

    N, nc = 256, 16
    ph = shepp_logan(N)[..., None]*gaussian_csm(N, N, nc)

    # Put into kspace
    ax = (0, 1)
    kspace = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(
        ph, axes=ax), axes=ax), axes=ax)

    # 20x20 calibration region
    ctr = int(N/2)
    pad = 20
    calib = kspace[ctr-pad:ctr+pad, ctr-pad:ctr+pad, :].copy()

    # Undersample: R=3
    kspace3x1 = kspace.copy()
    kspace3x1[1::3, ...] = 0
    kspace3x1[2::3, ...] = 0

    # Reconstruct using both GRAPPA and VC-GRAPPA
    res_grappa = cgrappa(kspace3x1.copy(), calib)
    res_nlgrappa = nlgrappa(
        kspace3x1.copy(), calib, ml_kernel='polynomial',
        ml_kernel_args={'cross_term_neighbors': 0})

    # Bring back to image space
    imspace_nlgrappa = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(
        res_nlgrappa, axes=ax), axes=ax), axes=ax)
    imspace_grappa = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(
        res_grappa, axes=ax), axes=ax), axes=ax)

    # Coil combine (sum-of-squares)
    cc_nlgrappa = np.sqrt(
        np.sum(np.abs(imspace_nlgrappa)**2, axis=-1))
    cc_grappa = np.sqrt(np.sum(np.abs(imspace_grappa)**2, axis=-1))
    ph = shepp_logan(N)

    cc_nlgrappa /= np.max(cc_nlgrappa.flatten())
    cc_grappa /= np.max(cc_grappa.flatten())
    ph /= np.max(cc_grappa.flatten())

    # Take a look
    plt.subplot(1, 2, 1)
    plt.imshow(cc_nlgrappa, cmap='gray')
    plt.title('NL-GRAPPA')
    plt.xlabel('NRMSE: %g' % compare_nrmse(ph, cc_nlgrappa))

    plt.subplot(1, 2, 2)
    plt.imshow(cc_grappa, cmap='gray')
    plt.title('GRAPPA')
    plt.xlabel('NRMSE: %g' % compare_nrmse(ph, cc_grappa))
    plt.show()
