'''Demonstrate how to implement Segmented GRAPPA.

Notes
-----
A modified implementation of the method described in [1]_.

Instead of separating the reconstructions of upper and lower kspaces
by upper and lower ACS regions, I found it more convienent to do the
entire reconstruction twice (one for each ACS region) and then
average the results.

References
----------
.. [1] Park, Jaeseok, et al. "Artifact and noise suppression in
       GRAPPA imaging using improved k‐space coil calibration and
       variable density sampling." Magnetic Resonance in
       Medicine: An Official Journal of the International Society
       for Magnetic Resonance in Medicine 53.1 (2005): 186-193.
'''

import numpy as np
import matplotlib.pyplot as plt
from phantominator import shepp_logan
from skimage.measure import compare_nrmse

from pygrappa import cgrappa
from utils import gaussian_csm

if __name__ == '__main__':

    # Simple phantom
    N, ncoil = 128, 5
    ph = shepp_logan(N)[..., None]*gaussian_csm(N, N, ncoil)
    ax = (0, 1)
    kspace = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(
        ph, axes=ax), axes=ax), axes=ax)

    # Two different calibration regions not including the center
    offset = 10
    pad = 5
    ctr = int(N/2)
    calib_upper = kspace[ctr-pad+offset:ctr+pad+offset, ...].copy()
    calib_lower = kspace[ctr-pad-offset:ctr+pad-offset, ...].copy()

    # A single calibration region at the center for comparison
    pad_single = 2*pad
    calib = kspace[ctr-pad_single:ctr+pad_single, ...].copy()

    # Undersample kspace
    kspace[:, ::2, :] = 0

    # Reconstruct segmented
    res_upper = cgrappa(kspace, calib_upper)
    res_lower = cgrappa(kspace, calib_lower)
    res_seg = (res_upper + res_lower)/2

    # Reconstruct using single calibration region at the center
    res_grappa = cgrappa(kspace, calib)

    # Into image space
    imspace_seg = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(
        res_seg, axes=ax), axes=ax), axes=ax)
    imspace_grappa = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(
        res_grappa, axes=ax), axes=ax), axes=ax)

    # Coil combine (sum-of-squares)
    cc_seg = np.sqrt(
        np.sum(np.abs(imspace_seg)**2, axis=-1))
    cc_grappa = np.sqrt(np.sum(np.abs(imspace_grappa)**2, axis=-1))
    ph = shepp_logan(N)

    # Normalize
    cc_seg /= np.max(cc_seg.flatten())
    cc_grappa /= np.max(cc_grappa.flatten())
    ph /= np.max(ph.flatten())

    # Take a look
    tx, ty = (0, 10)
    text_args = {'color': 'white'}
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(cc_seg, cmap='gray')
    plt.title('Segmented GRAPPA')
    plt.text(
        tx, ty, 'MSE: %.2f' % compare_nrmse(ph, cc_seg), text_args)

    plt.subplot(2, 2, 2)
    plt.imshow(cc_grappa, cmap='gray')
    plt.title('GRAPPA')
    plt.text(
        tx, ty, 'MSE: %.4f' % compare_nrmse(ph, cc_grappa), text_args)

    plt.subplot(2, 2, 3)
    calib_region = np.zeros((N, N), dtype=bool)
    calib_region[ctr-pad+offset:ctr+pad+offset, ...] = True
    calib_region[ctr-pad-offset:ctr+pad-offset, ...] = True
    plt.imshow(calib_region)
    plt.title('Segmented GRAPPA ACS regions')

    plt.subplot(2, 2, 4)
    calib_region = np.zeros((N, N), dtype=bool)
    calib_region[ctr-pad_single:ctr+pad_single, ...] = True
    plt.imshow(calib_region)
    plt.title('GRAPPA ACS region')

    plt.show()
