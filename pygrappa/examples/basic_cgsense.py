'''Basic usage of CG-SENSE implementation.'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from phantominator import shepp_logan

from pygrappa import cgsense
from pygrappa.utils import gaussian_csm

if __name__ == '__main__':

    N, nc = 128, 4
    sens = gaussian_csm(N, N, nc)

    im = shepp_logan(N) + np.finfo('float').eps
    im = im[..., None]*sens
    kspace = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(
        im, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))

    # Undersample
    kspace[::2, 1::2, :] = 0
    kspace[1::2, ::2, :] = 0

    # SOS of the aliased image
    aliased = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(
        kspace, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))
    aliased = np.sqrt(np.sum(np.abs(aliased)**2, axis=-1))

    # Reconstruct from undersampled data and coil sensitivities
    res = cgsense(kspace, sens, coil_axis=-1)

    # Take a look
    nx, ny = 1, 3
    plt.subplot(nx, ny, 1)
    plt.imshow(aliased)
    plt.title('Aliased')
    plt.axis('off')

    plt.subplot(nx, ny, 2)
    plt.imshow(np.abs(res))
    plt.title('CG-SENSE')
    plt.axis('off')

    plt.subplot(nx, ny, 3)
    true = np.abs(shepp_logan(N))
    true /= np.max(true)
    res = np.abs(res)
    res /= np.max(res)
    plt.imshow(true - res)
    plt.title('|True - CG-SENSE|')
    plt.axis('off')
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(cax=cax)

    plt.show()
