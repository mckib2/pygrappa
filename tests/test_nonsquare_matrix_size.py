'''Test that nonsquare matrices work.'''

import numpy as np
import matplotlib.pyplot as plt
from phantominator import shepp_logan

from pygrappa import cgrappa
from utils import gaussian_csm

if __name__ == '__main__':

    # helpers
    fft = lambda x0: np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(
        x0, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))
    ifft = lambda x0: np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(
        x0, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))

    # Sizes we want
    N, M, nc = 300, 192, 8
    calib_lines = 20

    # Make square kspace
    assert N >= M, 'First must be largest dim!'
    N2 = int(N/2)
    imspace = np.flipud(
        shepp_logan(N))[..., None]*gaussian_csm(N, N, nc)

    # Trim down to make nonsquare
    # 1st > 2nd
    trim = int((N - M)/2)
    pad = int(calib_lines/2)
    imspace1 = imspace[:, trim:-trim, :]
    kspace1 = fft(imspace1)
    calib1 = kspace1[N2-pad:N2+pad, ...].copy()
    kspace1[::2, ...] = 0 # Undersample: R=2

    # 2nd > 1st
    imspace2 = imspace[trim:-trim, ...]
    kspace2 = fft(imspace2)
    calib2 = kspace2[:, N2-pad:N2+pad, :].copy()
    kspace2[:, ::2, :] = 0

    # Do the thing
    res1 = cgrappa(kspace1, calib1, kernel_size=(5, 5), coil_axis=-1)
    res2 = cgrappa(kspace2, calib2, kernel_size=(5, 5), coil_axis=-1)

    # Show sum-of-squares results
    sos1 = np.sqrt(np.sum(np.abs(ifft(res1))**2, axis=-1))
    sos2 = np.sqrt(np.sum(np.abs(ifft(res2))**2, axis=-1))

    plt.subplot(1, 2, 1)
    plt.imshow(sos1)
    plt.subplot(1, 2, 2)
    plt.imshow(sos2)
    plt.show()
