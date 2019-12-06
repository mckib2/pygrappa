'''Demo of Non-Cartesian GRAPPA using PARS.'''

import numpy as np
import matplotlib.pyplot as plt
from phantominator import radial, kspace_shepp_logan
from phantominator.kspace import _kspace_ellipse_sens
from phantominator.sens_coeffs import _sens_coeffs

from pygrappa import pars
from utils import gridder

if __name__ == '__main__':

    # Helper functions
    ifft = lambda x0: np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(
        x0, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))
    sos = lambda x0: np.sqrt(np.sum(np.abs(x0)**2, axis=-1))

    # Simulate a radial trajectory
    sx, spokes, nc = 256, 256, 8
    kx, ky = radial(sx, spokes)

    # We reorder the samples like this for easier undersampling later
    kx = np.reshape(kx, (sx, spokes)).flatten('F')
    ky = np.reshape(ky, (sx, spokes)).flatten('F')

    # Sample Shepp-Logan at points (kx, ky) with nc coils:
    kspace = kspace_shepp_logan(kx, ky, ncoil=nc)
    k = kspace.copy()

    # Get some calibration data -- for PARS, we train using coil
    # sensitivity maps.  Here's a hacky way to get those:
    coeffs = []
    for ii in range(nc):
        coeffs.append(_sens_coeffs(ii))
    coeffs = np.array(coeffs)
    tx, ty = np.meshgrid(
        np.linspace(np.min(kx), np.max(kx), sx),
        np.linspace(np.min(ky), np.max(ky), sx))
    tx, ty = tx.flatten(), ty.flatten()
    calib = _kspace_ellipse_sens(
        tx/2 + 1j*ty/2, 0, 0, 1, .95, .95, 0, coeffs).T
    sens = ifft(calib.reshape((sx, sx, nc)))

    # BART's phantom function has a better way to simulate coil
    # sensitivity maps, see examples/bart_pars.py

    # Undersample: R=2
    k[::2] = 0

    # Reconstruct with PARS by setting kernel_radius
    res = pars(kx, ky, k, sens, kernel_radius=.8)

    # Let's take a look
    gridder0 = lambda x0: gridder(kx, ky, x0, sx=sx, sy=sx)

    plt.subplot(1, 3, 1)
    plt.imshow(sos(gridder0(kspace.reshape((-1, nc)))))
    plt.title('Truth')

    plt.subplot(1, 3, 2)
    plt.imshow(sos(gridder0(k)))
    plt.title('Undersampled')

    plt.subplot(1, 3, 3)
    plt.imshow(sos(res))
    plt.title('PARS')
    plt.show()
