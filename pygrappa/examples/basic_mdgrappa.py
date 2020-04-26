'''Basic usage of multidimensional GRAPPA.'''

from time import time

import numpy as np
import matplotlib.pyplot as plt
from phantominator import shepp_logan

from pygrappa import mdgrappa
from utils import gaussian_csm

if __name__ == '__main__':

    # Generate fake sensitivity maps: mps
    L, M, N = 128, 128, 32
    ncoils = 4
    mps = gaussian_csm(L, M, ncoils)[..., None, :]

    # generate 3D phantom
    ph = shepp_logan((L, M, N), zlims=(-.25, .25))
    imspace = ph[..., None]*mps
    ax = (0, 1, 2)
    kspace = np.fft.fftshift(np.fft.fftn(
        np.fft.ifftshift(imspace, axes=ax), axes=ax), axes=ax)

    # crop 20x20xN window from the center of k-space for calibration
    # (use all z-axis)
    pd = 10
    ctrs = [int(s/2) for s in kspace.shape[:2]]
    calib = kspace[tuple([slice(ctr-pd, ctr+pd) for ctr in ctrs] + [slice(None), slice(None)])].copy()
    # calib = kspace[ctr-pd:ctr+pd, ctr-pd:ctr+pd, :].copy()

    # calibrate a kernel
    kernel_size = (4, 5, 5)

    # undersample by a factor of 2 in both kx and ky
    kspace[::2, 1::2, ...] = 0
    kspace[1::2, ::2, ...] = 0

    # Do the recon
    t0 = time()
    res = mdgrappa(kspace, calib, kernel_size)
    print('Took %g sec' % (time() - t0))

    # Take a look at a single slice (z=-.25)
    res = np.abs(np.fft.fftshift(np.fft.ifftn(
        np.fft.ifftshift(res, axes=ax), axes=ax), axes=ax))
    res = res[..., 0, :]
    res0 = np.zeros((2*L, 2*M))
    kk = 0
    for idx in np.ndindex((2, 2)):
        ii, jj = idx[:]
        res0[ii*L:(ii+1)*L, jj*M:(jj+1)*M] = res[..., kk]
        kk += 1
    plt.imshow(res0, cmap='gray')
    plt.show()
