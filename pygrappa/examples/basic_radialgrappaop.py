'''Basic usage of Radial GRAPPA operator.'''

import numpy as np
from phantominator import radial, kspace_shepp_logan

from pygrappa import radialgrappaop

if __name__ == '__main__':

    N, spokes, nc = 64, 32, 8
    kx, ky = radial(N, spokes)
    kx = np.reshape(kx, (N, spokes)).flatten('F')
    ky = np.reshape(ky, (N, spokes)).flatten('F')
    k = kspace_shepp_logan(kx, ky, ncoil=nc)
    k = np.reshape(k, (N, spokes, nc))
    kx = np.reshape(kx, (N, spokes))
    ky = np.reshape(ky, (N, spokes))

    Gx, Gy = radialgrappaop(kx, ky, k)
