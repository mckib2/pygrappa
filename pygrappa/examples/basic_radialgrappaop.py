'''Basic usage of Radial GRAPPA operator.'''

from time import time

import numpy as np
import matplotlib.pyplot as plt
from phantominator import radial, kspace_shepp_logan

from pygrappa import radialgrappaop

if __name__ == '__main__':

    N, spokes, nc = 288, 62, 3
    kx, ky = radial(N, spokes)
    kx = np.reshape(kx, (N, spokes), 'F').flatten()
    ky = np.reshape(ky, (N, spokes), 'F').flatten()
    k = kspace_shepp_logan(kx, ky, ncoil=nc)
    k = np.reshape(k, (N, spokes, nc))
    kx = np.reshape(kx, (N, spokes))
    ky = np.reshape(ky, (N, spokes))

    plt.scatter(kx, ky, .1)
    plt.title('Radial Sampling Pattern')
    plt.show()

    t0 = time()
    Gx, Gy = radialgrappaop(kx, ky, k)
    print('Gx, Gy computed in %g seconds' % (time() - t0))
