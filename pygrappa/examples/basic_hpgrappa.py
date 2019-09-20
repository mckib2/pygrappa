'''Basic hp-GRAPPA usage.'''


import numpy as np
from mpl_toolkits.mplot3d import Axes3D # pylint: disable=W0611
import matplotlib.pyplot as plt
from phantominator import shepp_logan

from pygrappa import hpgrappa, cgrappa
from utils import gaussian_csm

if __name__ == '__main__':

    # The much abused Shepp-Logan phantom
    N, ncoil = 128, 5
    ph = shepp_logan(N)[..., None]*gaussian_csm(N, N, ncoil)
    fov = (10e-2, 10e-2) # 10cm x 10cm FOV

    # k-space-ify it
    ax = (0, 1)
    kspace = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(
        ph, axes=ax), axes=ax), axes=ax)

    # Get an ACS region
    pad = 12
    ctr = int(N/2)
    calib = kspace[ctr-pad:ctr+pad, ...].copy()

    # Undersample: R=3
    kspace[0::3, ...] = 0
    kspace[1::3, ...] = 0

    # Run hp-GRAPPA and GRAPPA to compare results
    res_hpgrappa, F2 = hpgrappa(
        kspace, calib, fov=fov, ret_filter=True)
    res_grappa = cgrappa(kspace, calib)

    # Into image space
    imspace_hpgrappa = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(
        res_hpgrappa, axes=ax), axes=ax), axes=ax)
    imspace_grappa = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(
        res_grappa, axes=ax), axes=ax), axes=ax)

    # Coil combine (sum-of-squares)
    cc_hpgrappa = np.sqrt(
        np.sum(np.abs(imspace_hpgrappa)**2, axis=-1))
    cc_grappa = np.sqrt(np.sum(np.abs(imspace_grappa)**2, axis=-1))
    ph = shepp_logan(N)

    # Take a look
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(
        np.linspace(-1, 1, N),
        np.linspace(-1, 1, N))
    ax.plot_surface(X, Y, F2, linewidth=0, antialiased=False)
    plt.title('High Pass Filter')

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(cc_hpgrappa, cmap='gray')
    plt.title('hp-GRAPPA')

    plt.subplot(1, 2, 2)
    plt.imshow(cc_grappa, cmap='gray')
    plt.title('GRAPPA')

    plt.show()
