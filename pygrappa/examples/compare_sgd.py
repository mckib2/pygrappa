'''Compare performance of stochastic gradient descent with standard.
'''

from time import time

import numpy as np
from phantominator import shepp_logan

from utils.gen_lstsq import isgd

if __name__ == '__main__':


    # Generate fake sensitivity maps: mps
    N = 128
    ncoils = 4
    xx = np.linspace(0, 1, N)
    x, y = np.meshgrid(xx, xx)
    mps = np.zeros((N, N, ncoils))
    mps[..., 0] = x**2
    mps[..., 1] = 1 - x**2
    mps[..., 2] = y**2
    mps[..., 3] = 1 - y**2

    # generate 4 coil phantom
    ph = shepp_logan(N)
    imspace = ph[..., None]*mps
    imspace = imspace.astype(np.complex64)
    ax = (0, 1)
    kspace = np.fft.fftshift(np.fft.fft2(
        np.fft.ifftshift(imspace, axes=ax), axes=ax), axes=ax)

    # crop window from the center of k-space for calibration
    pd = 10
    ctr = int(N/2)
    calib = kspace[ctr-pd:ctr+pd, ctr-pd:ctr+pd, :].copy()

    # calibrate a kernel
    kernel_size = (7, 7)

    # undersample by a factor of 2 in both kx and ky
    kspace[::2, 1::2, :] = 0
    kspace[1::2, ::2, :] = 0

    mask = np.abs(kspace[ctr-2:ctr+2+1, ctr-2:ctr+2+1, 0]) > 0
    np.random.seed(0)
    t0 = time()
    W = isgd(calib, mask, kernel_size=(5, 5), coil_axis=-1)
    print('Implicit stochastic GD took %g seconds' % (time() - t0))

    S = kspace[ctr-2:ctr+2+1, ctr-2:ctr+2+1, :]
    S = S[mask].flatten()[None, :]
    recon = S @ W
    print(recon.squeeze())
    print(kspace[ctr, ctr, :])
