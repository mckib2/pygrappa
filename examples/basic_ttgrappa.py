'''Demo of Non-Cartesian GRAPPA.'''

import numpy as np
import matplotlib.pyplot as plt
from phantominator import radial
from bart import bart # pylint: disable=E0401

from pygrappa import ttgrappa
from utils import gridder

if __name__ == '__main__':

    # Define a radial trajectory
    sx, spokes, nc = 128, 128, 8
    kx, ky = radial(sx, spokes)

    # We need to reorder the samples like this for easier
    # undersampling...
    kx = np.reshape(kx, (sx, spokes)).flatten('F')
    ky = np.reshape(ky, (sx, spokes)).flatten('F')
    kxy = np.concatenate((kx[:, None], ky[:, None]), axis=1)

    # Make it look like BART trajectory so we can get samples...
    # traj = bart(1, 'traj -r -x %d -y %d' % (sx, spokes))
    traj = np.concatenate((
        kx.reshape((1, sx, spokes)),
        ky.reshape((1, sx, spokes)),
        np.zeros((1, sx, spokes))), axis=0)

    # Get phantom from BART since phantominator doesn't have
    # arbitrary sampling for multicoil Shepp-Logan yet...
    kspace = bart(1, 'phantom -k -s %d -t' % nc, traj)

    # Get the trajectory and kspace samples
    k = kspace.reshape((-1, nc))

    # Get some calibration data
    cx = kx.copy()
    cy = ky.copy()
    calib = k.copy()
    # calib = np.tile(calib[:, None, :], (1, 2, 1))
    calib = calib[:, None, :]

    # Undersample
    k[::4] = 0

    # Reconstruct with Non-Cartesian GRAPPA
    res = ttgrappa(
        kx, ky, k, cx, cy, calib, kernel_size=25, coil_axis=-1)

    # Let's take a look
    sos = lambda x0: np.sqrt(np.sum(np.abs(x0)**2, axis=-1))
    gridder0 = lambda x0: gridder(kx, ky, x0, sx=sx, sy=sx)
    plt.subplot(1, 3, 1)
    plt.imshow(sos(gridder0(k)))
    plt.title('Undersampled')

    plt.subplot(1, 3, 2)
    plt.imshow(sos(gridder0(kspace.reshape((-1, nc)))))
    plt.title('True')

    plt.subplot(1, 3, 3)
    plt.imshow(sos(gridder0(res)))
    plt.title('Through-time GRAPPA')
    plt.show()
