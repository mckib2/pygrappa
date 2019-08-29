'''Example of how LIKE is called.'''

import numpy as np
import matplotlib.pyplot as plt
from phantominator import shepp_logan

from pygrappa import like
from utils import gaussian_csm

if __name__ == '__main__':

    N = 128
    ncoil = 5
    ph = shepp_logan(N)
    csm = gaussian_csm(N, N, ncoil)
    ph = ph[..., None]*csm


    kspace = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(
        ph, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))

    # Get one line for calibration and undersample R=2
    ctr = int(N/2)
    calib = (kspace[ctr-1:ctr+2, ...].copy())
    print(calib.shape)
    kspace[::2, ...] = 0

    res = like(kspace, calib)
    print(res.shape)

    plt.imshow(np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(
        res[..., 0], axes=(0, 1)), axes=(0, 1)), axes=(0, 1))))
    plt.show()
