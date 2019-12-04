'''Do Cartesian to radial gridding.'''

from time import time

import numpy as np
from scipy.cluster.vq import whiten
import matplotlib.pyplot as plt
from phantominator import radial, kspace_shepp_logan

from pygrappa import radialgrappaop, grog
from utils import gridder

if __name__ == '__main__':

    # Helpers
    ifft = lambda x0: np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(
        x0, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))
    sos = lambda x0: np.sqrt(np.sum(np.abs(x0)**2, axis=-1))

    # Radially sampled Shepp-Logan
    N, spokes, nc = 288, 72, 8
    kx, ky = radial(N, spokes)
    kx = np.reshape(kx, (N, spokes), 'F').flatten()
    ky = np.reshape(ky, (N, spokes), 'F').flatten()
    k = kspace_shepp_logan(kx, ky, ncoil=nc)
    k = whiten(k) # whitening seems to help conditioning of Gx, Gy

    # Get the GRAPPA operators
    t0 = time()
    Gx, Gy = radialgrappaop(kx, ky, k, nspokes=spokes)
    print('Gx, Gy computed in %g seconds' % (time() - t0))

    # Do forward GROG (with oversampling)
    t0 = time()
    res_cart = grog(kx, ky, k, 2*N, 2*N, Gx, Gy)
    print('Gridded in %g seconds' % (time() - t0))

    # Now back to radial (inverse GROG)
    res_radial = grog(
        kx, ky, np.reshape(res_cart, (-1, nc), order='F'), 2*N, 2*N,
        Gx, Gy, inverse=True)

    # Make sure we gridded something recognizable
    nx, ny = 1, 3
    plt.subplot(nx, ny, 1)
    plt.imshow(sos(gridder(kx, ky, k, N, N)))
    plt.title('Radial Truth')

    plt.subplot(nx, ny, 2)
    N2 = int(N/2)
    plt.imshow(sos(ifft(res_cart))[N2:-N2, N2:-N2])
    plt.title('GROG Cartesian')

    plt.subplot(nx, ny, 3)
    plt.imshow(sos(gridder(kx, ky, res_radial, N, N)))
    plt.title('GROG Radial (Inverse)')

    plt.show()
