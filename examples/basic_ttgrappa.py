'''Demo of Non-Cartesian GRAPPA.'''

import numpy as np
import matplotlib.pyplot as plt

from bart import bart # pylint: disable=E0401

from pygrappa import ttgrappa

if __name__ == '__main__':

    # Get phantom from BART since phantominator doesn't have
    # arbitrary sampling yet...
    sx, spokes, nc = 128, 128, 8
    traj = bart(1, 'traj -r -x %d -y %d' % (sx, spokes))
    nufft = lambda x0: bart(
        1, 'nufft -i -t -d %d:%d:1' % (sx, sx),
        traj, x0.reshape((1, sx, spokes, nc))).squeeze()
    kspace = bart(1, 'phantom -k -s %d -t' % nc, traj)

    # Get the trajectory and kspace samples
    kx = traj[0, ...].real.flatten()
    ky = traj[1, ...].real.flatten()
    kx /= np.max(np.abs(kx))
    ky /= np.max(np.abs(ky))
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
    plt.subplot(1, 3, 1)
    plt.imshow(sos(nufft(k)))
    plt.title('Undersampled')

    plt.subplot(1, 3, 2)
    plt.imshow(sos(nufft(kspace)))
    plt.title('True')

    plt.subplot(1, 3, 3)
    plt.imshow(sos(nufft(res)))
    plt.title('Through-time GRAPPA')
    plt.show()
