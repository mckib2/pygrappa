'''Compare performance of stochastic gradient descent with standard.
'''

from time import time

import numpy as np
import matplotlib.pyplot as plt
from phantominator import shepp_logan

# from utils.gen_lstsq import isgd
from utils.gen_lstsq import reconstructor

if __name__ == '__main__':


    # Generate fake sensitivity maps: mps
    N = 256
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
    pd = 20
    ctr = int(N/2)
    calib = kspace[ctr-pd:ctr+pd, ctr-pd:ctr+pd, :].copy()

    # calibrate a kernel
    kernel_size = (7, 7)

    # undersample by a factor of 2 in both kx and ky
    kspace[::2, 1::2, :] = 0
    kspace[1::2, ::2, :] = 0

    # Reconstruct kspace
    weight_fun_args = {
        'thrash': 10,
        'batch_size': 20,
    }
    t0 = time()
    res = reconstructor(
        kspace, calib, kernel_size, coil_axis=-1, **weight_fun_args)
    print('Stochastic GD took %g seconds' % (time() - t0))
    res = np.fft.ifftshift(np.fft.ifft2(
        np.fft.fftshift(res, axes=ax), axes=ax), axes=ax)
    sos = np.sqrt(np.sum(np.abs(res)**2, axis=-1))
    plt.imshow(sos)
    plt.show()
