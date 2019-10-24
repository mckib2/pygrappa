'''Basic usage of the GRAPPA operator.'''

import numpy as np
import matplotlib.pyplot as plt
from phantominator import shepp_logan
from skimage.metrics import normalized_root_mse as compare_nrmse # pylint: disable=E0611,E0401

from pygrappa import cgrappa, grappaop
from utils import gaussian_csm

if __name__ == '__main__':

    # Make a simple phantom -- note that GRAPPA operator only works
    # well with pretty well separated coil sensitivities, so using
    # these simple maps we don't expect GRAPPA operator to work as
    # well as GRAPPA when trying to do "GRAPPA" things
    N, nc = 256, 16
    ph = shepp_logan(N)[..., None]*gaussian_csm(N, N, nc)

    # Put into kspace
    ax = (0, 1)
    kspace = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(
        ph, axes=ax), axes=ax), axes=ax)

    # 20x20 calibration region
    ctr = int(N/2)
    pad = 10
    calib = kspace[ctr-pad:ctr+pad, ctr-pad:ctr+pad, :].copy()

    # Undersample: R=4
    kspace4x1 = kspace.copy()
    kspace4x1[1::4, ...] = 0
    kspace4x1[2::4, ...] = 0
    kspace4x1[3::4, ...] = 0

    # Compare to regular ol' GRAPPA
    grecon4x1 = cgrappa(kspace4x1, calib, kernel_size=(4, 5))

    # Get a GRAPPA operator and do the recon
    Gx, Gy = grappaop(calib)
    recon4x1 = kspace4x1.copy()
    recon4x1[1::4, ...] = recon4x1[0::4, ...] @ Gx
    recon4x1[2::4, ...] = recon4x1[1::4, ...] @ Gx
    recon4x1[3::4, ...] = recon4x1[2::4, ...] @ Gx

    # Try different undersampling factors: Rx=2, Ry=2.  Same Gx, Gy
    # will work since we're using the same calibration region!
    kspace2x2 = kspace.copy()
    kspace2x2[1::2, ...] = 0
    kspace2x2[:, 1::2, :] = 0
    grecon2x2 = cgrappa(kspace2x2, calib, kernel_size=(4, 5))
    recon2x2 = kspace2x2.copy()
    recon2x2[1::2, ...] = recon2x2[::2, ...] @ Gx
    recon2x2[:, 1::2, :] = recon2x2[:, ::2, :] @ Gy

    # Bring everything back into image space, coil combine, and
    # normalize for comparison
    fft = lambda x0: np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(
        x0, axes=ax), axes=ax), axes=ax)
    sos = lambda x0: np.sqrt(np.sum(np.abs(x0)**2, axis=-1))
    normalize = lambda x0: x0/np.max(x0.flatten())

    ph = normalize(shepp_logan(N))
    aliased4x1 = normalize(sos(fft(kspace4x1)))
    aliased2x2 = normalize(sos(fft(kspace2x2)))
    grappa4x1 = normalize(sos(fft(grecon4x1)))
    grappa2x2 = normalize(sos(fft(grecon2x2)))
    grappa_op4x1 = normalize(sos(fft(recon4x1)))
    grappa_op2x2 = normalize(sos(fft(recon2x2)))

    # Let's take a gander
    nx, ny = 2, 3
    plt.subplot(nx, ny, 1)
    plt.imshow(aliased4x1, cmap='gray')
    plt.title('Aliased')
    plt.ylabel('Rx=4')

    plt.subplot(nx, ny, 2)
    plt.imshow(grappa4x1, cmap='gray')
    plt.title('GRAPPA')
    plt.xlabel('NRMSE: %.4f' % compare_nrmse(ph, grappa4x1))

    plt.subplot(nx, ny, 3)
    plt.imshow(grappa_op4x1, cmap='gray')
    plt.title('GRAPPA operator')
    plt.xlabel('NRMSE: %.4f' % compare_nrmse(ph, grappa_op4x1))

    plt.subplot(nx, ny, 4)
    plt.imshow(aliased2x2, cmap='gray')
    plt.ylabel('Rx=2, Ry=2')

    plt.subplot(nx, ny, 5)
    plt.imshow(grappa2x2, cmap='gray')
    plt.xlabel('NRMSE: %.4f' % compare_nrmse(ph, grappa2x2))

    plt.subplot(nx, ny, 6)
    plt.imshow(grappa_op2x2, cmap='gray')
    plt.xlabel('NRMSE: %.4f' % compare_nrmse(ph, grappa_op2x2))

    plt.show()
