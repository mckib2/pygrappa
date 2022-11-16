'''Basic usage of multidimensional GRAPPA.'''

from time import time

import numpy as np
import matplotlib.pyplot as plt
from phantominator import shepp_logan

from pygrappa import mdgrappa
from pygrappa.utils import gaussian_csm

if __name__ == '__main__':

    # Generate fake sensitivity maps: mps
    L, M, N = 160, 92, 8
    ncoils = 15
    mps = gaussian_csm(L, M, ncoils)[..., None, :]

    # generate 3D phantom
    ph = shepp_logan((L, M, N), zlims=(-.25, .25))
    imspace = ph[..., None]*mps
    ax = (0, 1, 2)
    kspace = np.fft.fftshift(np.fft.fftn(
        np.fft.ifftshift(imspace, axes=ax), axes=ax), axes=ax)

    # calibrate a kernel
    kernel_size = (5, 5, 5)

    # undersample by a factor of 2 in both kx and ky
    mask = np.ones(kspace.shape, dtype=bool)
    mask[::2, 1::2, ...] = False
    mask[1::2, ::2, ...] = False

    # Include calib in data: 20x20xN window at center of k-space for
    # calibration (use all z-axis)
    ctrs = [int(s/2) for s in kspace.shape[:2]]
    pds = [20, 8, 4]
    mask[tuple([slice(ctr-pd, ctr+pd) for ctr, pd in zip(ctrs, pds)] +
               [slice(None), slice(None)])] = True
    kspace *= mask

    # Do the recon
    t0 = time()
    res = mdgrappa(kspace, kernel_size=kernel_size)
    print(f'Took {time() - t0} sec')

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
