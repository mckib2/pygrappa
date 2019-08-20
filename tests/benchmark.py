'''Compare performance of grappa with and without C implementation.'''

from time import time

import numpy as np
from phantominator import shepp_logan

from pygrappa import cgrappa
from pygrappa import grappa
from utils import gaussian_csm

if __name__ == '__main__':

    # Generate fake sensitivity maps: mps
    N = 512
    ncoils = 32
    mps = gaussian_csm(N, N, ncoils)

    # generate 4 coil phantom
    ph = shepp_logan(N)
    imspace = ph[..., None]*mps
    imspace = imspace.astype('complex')
    ax = (0, 1)
    kspace = 1/np.sqrt(N**2)*np.fft.fftshift(np.fft.fft2(
        np.fft.ifftshift(imspace, axes=ax), axes=ax), axes=ax)

    # crop 20x20 window from the center of k-space for calibration
    pd = 10
    ctr = int(N/2)
    calib = kspace[ctr-pd:ctr+pd, ctr-pd:ctr+pd, :].copy()

    # calibrate a kernel
    kernel_size = (5, 5)

    # undersample by a factor of 2 in both x and y
    kspace[::2, 1::2, :] = 0
    kspace[1::2, ::2, :] = 0

    # Time both implementations
    t0 = time()
    recon0 = grappa(kspace, calib, (5, 5))
    print(' GRAPPA: %g' % (time() - t0))

    t0 = time()
    recon1 = cgrappa(kspace, calib, (5, 5))
    print('CGRAPPA: %g' % (time() - t0))

    assert np.allclose(recon0, recon1)
