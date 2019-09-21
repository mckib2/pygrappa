'''Demo of Non-Cartesian GRAPPA using PARS.'''

import numpy as np
import matplotlib.pyplot as plt
from phantominator import radial, kspace_shepp_logan
from phantominator.kspace import _kspace_ellipse_sens
from phantominator.sens_coeffs import _sens_coeffs

# from pygrappa import ttgrappa # ttgrappa and PARS are basically same
from pygrappa import pars
from utils import gridder

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
        tx, ty, 0, 0, 1, .98, .98, 0, coeffs).T
    sens = calib.reshape((sx, sx, nc))

    # Undersample: R=2
    k[::2] = 0

    # Reconstruct with PARS by setting kernel_radius
    res = pars(kx, ky, k, sens)

    # Let's take a look
    sos = lambda x0: np.sqrt(np.sum(np.abs(x0)**2, axis=-1))
    gridder0 = lambda x0: gridder(kx, ky, x0, sx=sx, sy=sx)
    ifft = lambda x0: np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(
        x0, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))
    plt.subplot(1, 3, 1)
    plt.imshow(sos(gridder0(k)))
    plt.title('Undersampled')

    plt.subplot(1, 3, 2)
    plt.imshow(sos(gridder0(kspace.reshape((-1, nc)))))
    plt.title('True')

    plt.subplot(1, 3, 3)
    plt.imshow(sos(ifft(res)))
    # plt.imshow(sos(ifft(sens)))
    plt.title('PARS')
    plt.show()
