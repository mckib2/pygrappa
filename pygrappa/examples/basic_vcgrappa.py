'''Demonstrate usage of VC-GRAPPA.'''

import numpy as np
import matplotlib.pyplot as plt
from phantominator import shepp_logan
from skimage.metrics import normalized_root_mse as compare_nrmse # pylint: disable=E0611,E0401

from pygrappa import vcgrappa, grappa
from utils import gaussian_csm

if __name__ == '__main__':

    # Simple phantom
    N = 128
    ncoil = 8
    _, phi = np.meshgrid( # background phase variation
        np.linspace(-np.pi, np.pi, N),
        np.linspace(-np.pi, np.pi, N))
    phi = np.exp(1j*phi)
    csm = gaussian_csm(N, N, ncoil)
    ph = shepp_logan(N)*phi
    ph = ph[..., None]*csm

    # Throw into k-space
    ax = (0, 1)
    kspace = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(
        ph, axes=ax), axes=ax), axes=ax)

    # 24 ACS lines
    pad = 12
    ctr = int(N/2)
    calib = kspace[ctr-pad:ctr+pad, ...].copy()

    # R=4
    kspace[1::4, ...] = 0
    kspace[2::4, ...] = 0
    kspace[3::4, ...] = 0

    # Reconstruct using both GRAPPA and VC-GRAPPA
    res_grappa = grappa(kspace, calib)
    res_vcgrappa = vcgrappa(kspace, calib)

    # Bring back to image space
    imspace_vcgrappa = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(
        res_vcgrappa, axes=ax), axes=ax), axes=ax)
    imspace_grappa = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(
        res_grappa, axes=ax), axes=ax), axes=ax)

    # Coil combine (sum-of-squares)
    cc_vcgrappa = np.sqrt(
        np.sum(np.abs(imspace_vcgrappa)**2, axis=-1))
    cc_grappa = np.sqrt(np.sum(np.abs(imspace_grappa)**2, axis=-1))
    ph = shepp_logan(N)

    # Normalize
    cc_vcgrappa /= np.max(cc_vcgrappa.flatten())
    cc_grappa /= np.max(cc_grappa.flatten())
    ph /= np.max(ph.flatten())

    # Take a look
    nx, ny = 2, 2
    plt.subplot(nx, ny, 1)
    plt.imshow(cc_vcgrappa, cmap='gray')
    plt.title('VC-GRAPPA')
    plt.xlabel('NRMSE: %g' % compare_nrmse(ph, cc_vcgrappa))

    plt.subplot(nx, ny, 2)
    plt.imshow(cc_grappa, cmap='gray')
    plt.title('GRAPPA')
    plt.xlabel('NRMSE: %g' % compare_nrmse(ph, cc_grappa))

    # Check residuals
    cc_vcgrappa_resid = ph - cc_vcgrappa
    cc_grappa_resid = ph - cc_grappa
    fac = np.max(np.concatenate(
        (cc_vcgrappa_resid, cc_grappa_resid)).flatten())
    plt_args = {
        'vmin': 0,
        'vmax': fac,
        'cmap': 'gray'
    }

    plt.subplot(nx, ny, 3)
    plt.imshow(np.abs(cc_vcgrappa_resid), **plt_args)
    plt.ylabel('Residuals (x%d)' % int(1/fac + .5))

    plt.subplot(nx, ny, 4)
    plt.imshow(np.abs(cc_grappa_resid), **plt_args)

    plt.show()
