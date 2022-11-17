'''Demo of Non-Cartesian GRAPPA.'''

import numpy as np
import matplotlib.pyplot as plt
from phantominator import radial, kspace_shepp_logan

from pygrappa import ttgrappa
from pygrappa.utils import gridder


def _sos(x0):
    return np.sqrt(np.sum(np.abs(x0)**2, axis=-1))


def _gridder0(x0):
    return gridder(kx, ky, x0, sx=sx, sy=sx)


if __name__ == '__main__':

    # Simulate a radial trajectory
    sx, spokes, nc = 128, 128, 8
    kx, ky = radial(sx, spokes)

    # We reorder the samples like this for easier undersampling later
    kx = np.reshape(kx, (sx, spokes)).flatten('F')
    ky = np.reshape(ky, (sx, spokes)).flatten('F')

    # Sample Shepp-Logan at points (kx, ky) with nc coils:
    kspace = kspace_shepp_logan(kx, ky, ncoil=nc)
    k = kspace.copy()

    # Get some calibration data -- ideally we would want to simulate
    # something other than the image we're going to reconstruct, but
    # since this is just proof of concept, we'll go ahead
    cx = kx.copy()
    cy = ky.copy()
    calib = k.copy()
    # calib = np.tile(calib[:, None, :], (1, 2, 1))
    calib = calib[:, None, :]  # middle axis is the through-time dim

    # Undersample: R=2
    k[::2] = 0

    # Reconstruct with Non-Cartesian GRAPPA
    res = ttgrappa(
        kx, ky, k, cx, cy, calib, kernel_size=25, coil_axis=-1)

    # Let's take a look
    plt.subplot(1, 3, 1)
    plt.imshow(_sos(_gridder0(k)))
    plt.title('Undersampled')

    plt.subplot(1, 3, 2)
    plt.imshow(_sos(_gridder0(kspace.reshape((-1, nc)))))
    plt.title('True')

    plt.subplot(1, 3, 3)
    plt.imshow(_sos(_gridder0(res)))
    plt.title('Through-time GRAPPA')
    plt.show()
