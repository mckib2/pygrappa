'''Basic demo of Slice-GRAPPA.'''

import numpy as np
from phantominator import shepp_logan

from pygrappa import slicegrappa
from utils import gaussian_csm

if __name__ == '__main__':

    # Get 2 slices of 3D Shepp-Logan phantom
    N = 128
    ph = shepp_logan((N, N, 2), zlims=(-.3, 0))

    # Apply some coil sensitivities
    ncoil = 4
    csm = gaussian_csm(N, N, ncoil)
    ph = ph[..., None, :]*csm[..., None]

    # Flip one of the slices so we can see it better
    ph[..., -1] = np.rot90(ph[..., -1])

    # import matplotlib.pyplot as plt
    # for ii in range(ph.shape[-1]):
    #     plt.subplot(ph.shape[-1], 1, ii+1)
    #     plt.imshow(np.abs(ph[..., 0, ii]), cmap='gray')
    # plt.show()

    # Put into kspace
    kspace = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(
        ph, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))

    # Calibration data is individual slices
    calib = kspace.copy()

    # Simulate SMS by simply adding slices together
    kspace_sms = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(
        np.sum(ph, axis=-1), axes=(0, 1)), axes=(0, 1)), axes=(0, 1))

    # We technically only have 1 time frame
    kspace_sms = kspace_sms[..., None]

    # Separate the slices using Slice-GRAPPA
    slicegrappa(kspace_sms, calib)
