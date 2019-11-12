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

    # Make square kspace
    assert N >= M, 'First must be largest dim!'
    N2 = int(N/2)
    imspace = np.flipud(
        shepp_logan(N))[..., None]*gaussian_csm(N, N, nc)

    # Trim down to make nonsquare
    pad = int((N - M)/2)
    imspace = imspace[:, pad:-pad, :]
    kspace = fft(imspace)

    # Get calib region
    calib_lines = 20
    pd = int(calib_lines/2)
    calib = kspace[N2-pd:N2+pd, ...].copy()
    print(calib.shape)

    # Undersample: R=2
    kspace[::2, ...] = 0

    # Do the thing
    res = cgrappa(kspace, calib, kernel_size=(5, 5), coil_axis=-1)
    print(res.shape)

    sos = np.sqrt(np.sum(np.abs(ifft(res))**2, axis=-1))
    plt.imshow(sos)
    plt.show()
