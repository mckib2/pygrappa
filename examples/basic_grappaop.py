'''Basic usage of the GRAPPA operator.'''

import numpy as np
import matplotlib.pyplot as plt
from phantominator import shepp_logan

from pygrappa import grappaop
from utils import gaussian_csm

if __name__ == '__main__':

    N, nc = 128, 5
    ph = shepp_logan(N)[..., None]*gaussian_csm(N, N, nc)

    ax = (0, 1)
    kspace = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(
        ph, axes=ax), axes=ax), axes=ax)

    ctr = int(N/2)
    pad = 10
    calib = kspace[ctr-pad:ctr+pad, ctr-pad:ctr+pad, :].copy()

    # Undersample
    kspace[::2, ...] = 0

    # Get a GRAPPA operator
    G = grappaop(calib)

    dim = 'C'
    S = np.reshape(kspace[1::2, ...], (-1, nc), order=dim)
    T = S @ G

    kspace[::2, ...] = T.reshape((ctr, N, nc), order=dim)

    recon = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(
        kspace, axes=ax), axes=ax), axes=ax)
    plt.imshow(np.abs(recon[..., 0]))
    plt.show()
