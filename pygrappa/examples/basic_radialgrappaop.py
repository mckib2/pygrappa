'''Basic usage of Radial GRAPPA operator.'''

from time import time

import numpy as np
from scipy.cluster.vq import whiten
import matplotlib.pyplot as plt
from phantominator import radial, kspace_shepp_logan
try:
    from skimage.metrics import normalized_root_mse as compare_nrmse  # pylint: disable=E0611,E0401
    from skimage.metrics import structural_similarity as compare_ssim  # pylint: disable=E0611,E0401
except ImportError:
    from skimage.measure import compare_nrmse, compare_ssim
from skimage.morphology import convex_hull_image
from skimage.filters import threshold_li

from pygrappa import radialgrappaop, grog


def ifft(x0):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(
        x0, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))


def sos(x0):
    return np.sqrt(np.sum(np.abs(x0)**2, axis=-1))


if __name__ == '__main__':

    # Radially sampled Shepp-Logan
    N, spokes, nc = 288, 72, 8
    kx, ky = radial(N, spokes)
    kx = np.reshape(kx, (N, spokes), 'F').flatten().astype(np.float32)
    ky = np.reshape(ky, (N, spokes), 'F').flatten().astype(np.float32)
    k = kspace_shepp_logan(kx, ky, ncoil=nc).astype(np.complex64)
    k = whiten(k)  # whitening seems to help conditioning of Gx, Gy

    # # Instead of whitening, maybe you prefer to reduce coils:
    # nc = 4
    # U, S, Vh = np.linalg.svd(k, full_matrices=False)
    # k = U[:, :nc] @ np.diag(S[:nc]) @ Vh[:nc, :nc]

    # Take a look at the sampling pattern:
    plt.scatter(kx, ky, .1)
    plt.title('Radial Sampling Pattern')
    plt.show()

    # Get the GRAPPA operators!
    t0 = time()
    Gx, Gy = radialgrappaop(kx, ky, k, nspokes=spokes)
    print('Gx, Gy computed in %g seconds' % (time() - t0))

    # Do GROG
    t0 = time()
    res, Dx, Dy = grog(kx, ky, k, N, N, Gx, Gy, ret_dicts=True)
    print('Gridded in %g seconds' % (time() - t0))

    # We can do it faster again if we pass back in the dictionaries!
    # t0 = time()
    # res = grog(kx, ky, k, N, N, Gx, Gy, Dx=Dx, Dy=Dy)
    # print('Gridded in %g seconds' % (time() - t0))

    # Get the Cartesian grid
    tx, ty = np.meshgrid(
        np.linspace(np.min(kx), np.max(kx), N),
        np.linspace(np.min(ky), np.max(ky), N))
    tx, ty = tx.flatten(), ty.flatten()
    kc = kspace_shepp_logan(tx, ty, ncoil=nc)
    kc = whiten(kc)
    outside = np.argwhere(
        np.sqrt(tx**2 + ty**2) > np.max(kx)).squeeze()
    kc[outside] = 0  # keep region of support same as radial
    kc = np.reshape(kc, (N, N, nc), order='F')

    # Make sure we gridded something recognizable
    nx, ny = 1, 3
    plt.subplot(nx, ny, 1)
    true = sos(ifft(kc))
    true /= np.max(true.flatten())
    thresh = threshold_li(true)
    mask = convex_hull_image(true > thresh)
    true *= mask
    plt.imshow(true)
    plt.title('Cartesian Sampled')

    plt.subplot(nx, ny, 2)
    scgrog = sos(ifft(res))
    scgrog /= np.max(scgrog.flatten())
    scgrog *= mask
    plt.imshow(scgrog)
    plt.title('SC-GROG')

    plt.subplot(nx, ny, 3)
    plt.imshow(true - scgrog)
    plt.title('Residual')
    nrmse = compare_nrmse(true, scgrog)
    ssim = compare_ssim(true, scgrog)
    plt.xlabel('NRMSE: %g, SSIM: %g' % (nrmse, ssim))
    # print(nrmse, ssim)

    plt.show()
