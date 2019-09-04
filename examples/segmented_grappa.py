'''Demonstrate how to implement Segmented GRAPPA.

Notes
-----
There must be something I don't understand about Segmented GRAPPA
because I can't seem to get it to work.  I'll have to look at it
again at a later date...

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
    offset = 5
    pad = 5
    ctr = int(N/2)
    calib_lower = kspace[ctr-pad-offset:ctr+pad-offset, ...].copy()
    calib_upper = kspace[ctr-pad+offset:ctr+pad+offset, ...].copy()

    # # Normal calibration region
    # pad = 10
    # ctr = int(N/2)
    # calib = kspace[ctr-pad:ctr+pad, ...].copy()

    # Undersample kspace
    ctr_pad = 5
    kspace_ctr = kspace[ctr-ctr_pad:ctr+ctr_pad, ...].copy()
    kspace[::2, ...] = 0
    N4 = int(N/4)
    kspace[0:N4:4, ...] = 0
    kspace[1:N4:4, ...] = 0
    kspace[2:N4:4, ...] = 0
    kspace[-N4-0::4, ...] = 0
    kspace[-N4-1::4, ...] = 0
    kspace[-N4-2::4, ...] = 0
    kspace[ctr-ctr_pad:ctr+ctr_pad, ...] = kspace_ctr

    plt.imshow(np.log(np.abs(kspace[..., 0])))
    plt.show()

    # Do GRAPPA on both regions separately
    res_lower = cgrappa(kspace, calib_lower)
    res_upper = cgrappa(kspace, calib_upper)
    res = np.concatenate(
        (res_lower[:ctr, ...], res_upper[ctr:, ...]), axis=0)

    # Plug back in calib
    res[ctr-pad-offset:ctr+pad-offset, ...] = calib_lower
    res[ctr-pad+offset:ctr+pad+offset, ...] = calib_upper

    plt.imshow(np.abs(res[..., 0]))
    plt.show()

    # Take a look
    res = np.abs(np.sqrt(N**2)*np.fft.fftshift(np.fft.ifft2(
        np.fft.ifftshift(res, axes=ax), axes=ax), axes=ax))
    res0 = np.zeros((2*N, 2*N))
    kk = 0
    for idx in np.ndindex((2, 2)):
        ii, jj = idx[:]
        res0[ii*N:(ii+1)*N, jj*N:(jj+1)*N] = res[..., kk]
        kk += 1
    plt.imshow(res0, cmap='gray')
    plt.show()
