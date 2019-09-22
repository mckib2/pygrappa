'''Do PARS using BART stuff.'''

from time import time

import numpy as np
import matplotlib.pyplot as plt
from bart import bart # pylint: disable=E0401

from pygrappa import pars
from utils import gridder

if __name__ == '__main__':

    sx, spokes, nc = 256, 256, 8
    traj = bart(1, 'traj -r -x%d -y%d' % (sx, spokes))
    kx, ky = traj[0, ...].real.flatten(), traj[1, ...].real.flatten()

    # Use BART to get Shepp-Logan and sensitivity maps
    t0 = time()
    k = bart(1, 'phantom -k -s%d -t' % nc, traj).reshape((-1, nc))
    print('Took %g seconds to simulate %d coils' % (time() - t0, nc))
    sens = bart(1, 'phantom -S%d -x%d' % (nc, sx)).squeeze()

    # Undersample
    ku = k.copy()
    ku[::2] = 0

    # Take a looksie
    sos = lambda x0: np.sqrt(np.sum(np.abs(x0)**2, axis=-1))
    plt.subplot(1, 3, 1)
    plt.imshow(sos(gridder(kx, ky, k, sx, sx)))
    plt.title('Truth')

    plt.subplot(1, 3, 2)
    plt.imshow(sos(gridder(kx, ky, ku, sx, sx)))
    plt.title('Undersampled')

    plt.subplot(1, 3, 3)
    res = pars(kx, ky, ku, sens, kernel_radius=.8)
    plt.imshow(sos(res))
    plt.title('PARS')

    plt.show()
