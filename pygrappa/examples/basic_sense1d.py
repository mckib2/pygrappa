'''Show basic usage of 1D SENSE.'''

import numpy as np
import matplotlib.pyplot as plt
from phantominator import shepp_logan

from pygrappa import sense1d
from utils import gaussian_csm

if __name__ == '__main__':
    N, nc = 128, 8
    im = shepp_logan(N)
    sens = gaussian_csm(N, N, nc)
    im = im[..., None]*sens
    kspace = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(
        im, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))

    # Undersample
    R = 4
    kspace[::R, ...] = 0
    kspace[1::R, ...] = 0
    kspace[2::R, ...] = 0

    # Do the SENSE recon
    res = sense1d(kspace, sens, Rx=R, coil_axis=-1, imspace=False)
    plt.imshow(np.abs(res))
    plt.show()
