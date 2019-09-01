'''Demonstrate usage of iGRAPPA.'''

import numpy as np
import matplotlib.pyplot as plt
from phantominator import shepp_logan
from skimage.measure import compare_nrmse

from pygrappa import igrappa, cgrappa
from utils import gaussian_csm

if __name__ == '__main__':

    # Simple phantom
    N = 128
    ncoil = 5
    csm = gaussian_csm(N, N, ncoil)
    ph = shepp_logan(N)[..., None]*csm

    # Add a little noise to spice things up
    std = .001
    n = (np.random.normal(0, std, ph.shape) +
         1j*np.random.normal(0, std, ph.shape))
    ph += n

    # Throw into k-space
    ax = (0, 1)
    kspace = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(
        ph, axes=ax), axes=ax), axes=ax)

    # 20x20 ACS region
    pad = 10
    ctr = int(N/2)
    calib = kspace[ctr-pad:ctr+pad, ctr-pad:ctr+pad, :].copy()

    # R=2x2
    kspace[::2, 1::2, :] = 0
    kspace[1::2, ::2, :] = 0

    # Reconstruct using both GRAPPA and iGRAPPA
    res_grappa = cgrappa(kspace, calib)
    res_igrappa = igrappa(kspace, calib)

    # Bring back to image space
    imspace_igrappa = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(
        res_igrappa, axes=ax), axes=ax), axes=ax)
    imspace_grappa = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(
        res_grappa, axes=ax), axes=ax), axes=ax)

    # Coil combine (sum-of-squares)
    cc_igrappa = np.sqrt(np.sum(np.abs(imspace_igrappa)**2, axis=-1))
    cc_grappa = np.sqrt(np.sum(np.abs(imspace_grappa)**2, axis=-1))
    ph = shepp_logan(N)

    # Take a look
    plt.subplot(1, 2, 1)
    plt.imshow(cc_igrappa, cmap='gray')
    plt.title('iGRAPPA')
    plt.xlabel('NRMSE: %g' % compare_nrmse(ph, cc_igrappa))

    plt.subplot(1, 2, 2)
    plt.imshow(cc_grappa, cmap='gray')
    plt.title('GRAPPA')
    plt.xlabel('NRMSE: %g' % compare_nrmse(ph, cc_grappa))
    plt.show()
