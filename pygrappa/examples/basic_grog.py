'''Example usage of GROG.'''

import numpy as np
import matplotlib.pyplot as plt
from phantominator import radial, kspace_shepp_logan

from pygrappa import grog

if __name__ == '__main__':

    # Simulate a radial trajectory
    N = 32
    sx, spokes, nc = N, N, 8
    kx, ky = radial(sx, spokes)
    kx = np.reshape(kx, (sx, spokes)).flatten('F')
    ky = np.reshape(ky, (sx, spokes)).flatten('F')

    # Sample Shepp-Logan at points (kx, ky) with nc coils:
    k = kspace_shepp_logan(kx, ky, ncoil=nc)

    res = grog(kx, ky, k, N, N)
    print(res.shape)

    plt.imshow(np.abs(res[..., 0]))
    plt.show()

    im = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(
        res, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))
    sos = np.sqrt(np.sum(np.abs(im)**2, axis=-1))
    plt.imshow(sos)
    plt.show()
