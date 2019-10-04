'''Basic usage of Radial GRAPPA operator.'''

from time import time

import numpy as np
import matplotlib.pyplot as plt
from phantominator import radial, kspace_shepp_logan

from pygrappa import radialgrappaop, grog

if __name__ == '__main__':

    # Radially sampled Shepp-Logan
    N, spokes, nc = 128, 128, 8
    kx, ky = radial(N, spokes)
    kx = np.reshape(kx, (N, spokes), 'F').flatten()
    ky = np.reshape(ky, (N, spokes), 'F').flatten()
    k = kspace_shepp_logan(kx, ky, ncoil=nc)

    # Reduce dimensionality
    nc = 3
    U, S, Vh = np.linalg.svd(k, full_matrices=False)
    k = U[:, :nc] @ np.diag(S[:nc]) @ Vh[:nc, :nc]

    # Put in correct shape for radialgrappaop
    k = np.reshape(k, (N, spokes, nc))
    kx = np.reshape(kx, (N, spokes))
    ky = np.reshape(ky, (N, spokes))

    # Take a look at the sampling pattern:
    plt.scatter(kx, ky, .1)
    plt.title('Radial Sampling Pattern')
    plt.show()

    # Get the GRAPPA operators!
    t0 = time()
    Gx, Gy = radialgrappaop(kx, ky, k)
    print('Gx, Gy computed in %g seconds' % (time() - t0))

    # Put in correct order for GROG
    kx = kx.flatten()
    ky = ky.flatten()
    k = np.reshape(k, (-1, nc))
    t0 = time()
    res = grog(kx, ky, k, N, N, Gx, Gy)
    print('Gridded in %g seconds' % (time() - t0))

    # Make sure we gridded something recognizable
    im = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(
        res, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))
    sos = np.sqrt(np.sum(np.abs(im)**2, axis=-1))

    plt.imshow(np.abs(sos).T)
    plt.show()
