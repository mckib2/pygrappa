'''Basic ESPIRiT example using Shepp-Logan phantom.'''

import numpy as np
import matplotlib.pyplot as plt
from phantominator import shepp_logan
from utils import gaussian_csm

from pygrappa import espirit


if __name__ == '__main__':

    # Generate fake sensitivity maps: mps
    # generate 4 coil phantom
    N, ncoil = 128, 6
    imspace = shepp_logan(N)[..., None]*gaussian_csm(N, N, ncoil)

    ax = (0, 1)
    kspace = 1/np.sqrt(N**2)*np.fft.fftshift(np.fft.fft2(
        np.fft.ifftshift(imspace, axes=ax), axes=ax), axes=ax)

    # crop window from the center of k-space for calibration
    pd = 12
    ctr = int(N/2)
    calib = kspace[ctr-pd:ctr+pd, ctr-pd:ctr+pd, :].copy()

    # calibrate a kernel
    kernel_size = (6, 7)

    # undersample by a factor of 2 in both kx and ky
    kspace[::2, 1::2, :] = 0
    kspace[1::2, ::2, :] = 0

    # reconstruct:
    res = espirit(
        kspace, calib, kernel_size, coil_axis=-1)
    print(res.shape)

    idx = 1
    for ii in range(res.shape[-1]):
        plt.subplot(1, res.shape[-1], idx)
        plt.imshow(np.abs(res[..., ii]))
        idx += 1
    plt.show()

    # Take a look
    # res = np.abs(np.sqrt(N**2)*np.fft.fftshift(np.fft.ifft2(
    #     np.fft.ifftshift(res, axes=ax), axes=ax), axes=ax))
    res0 = np.empty((2*N, 3*N))
    kk = 0
    for idx in np.ndindex((2, 3)):
        ii, jj = idx[:]
        res0[ii*N:(ii+1)*N, jj*N:(jj+1)*N] = np.abs(res[..., kk])
        kk += 1
    plt.imshow(res0, cmap='gray')
    plt.show()
