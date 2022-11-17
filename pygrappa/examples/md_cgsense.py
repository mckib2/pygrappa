'''Multidimensional CG-SENSE.'''

from time import time

import numpy as np
import matplotlib.pyplot as plt
from phantominator import shepp_logan

from pygrappa import cgsense
from pygrappa.utils import gaussian_csm

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

    # undersample by a factor of 2 in both kx and ky
    kspace[::2, 1::2, ...] = 0
    kspace[1::2, ::2, ...] = 0

    # Do the recon
    t0 = time()
    res = cgsense(kspace, mps)
    print('Took %g sec' % (time() - t0))

    # Take a look at a single slice (z=-.25)
    plt.imshow(np.abs(res[..., 0]), cmap='gray')
    plt.show()
