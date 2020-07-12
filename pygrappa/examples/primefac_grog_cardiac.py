'''ISMRM abstract code for prime factorization speed-up for SC-GROG.
'''

from time import time

import numpy as np
import matplotlib.pyplot as plt
try:
    from skimage.metrics import normalized_root_mse as compare_nrmse  # pylint: disable=E0611,E0401
except ImportError:
    from skimage.measure import compare_nrmse

from pygrappa import grog, radialgrappaop

if __name__ == '__main__':

    # Load in cardiac data
    path = 'data/meas_MID34_CV_Radial7Off_2.4ml_FID1789_Kspace.npy.npz'
    data = np.load(path)

    time_pt, sl = 20, 0
    k = data['kSpace'][:, :, time_pt, :, sl]
    kx = data['kx'][..., time_pt].astype(np.float32)
    ky = data['ky'][..., time_pt].astype(np.float32)
    N, spokes, nc = k.shape[:]
    print(k.shape, kx.shape, ky.shape)

    # Get the GRAPPA operators!
    t0 = time()
    Gx, Gy = radialgrappaop(kx, ky, k)
    print('Gx, Gy computed in %g seconds' % (time() - t0))

    # Put in correct order for GROG
    kx = kx.flatten()
    ky = ky.flatten()
    k = np.reshape(k, (-1, nc))

    # Do GROG without primefac
    t0 = time()
    res = grog(kx, ky, k, N, N, Gx, Gy, use_primefac=False)
    print('Gridded in %g seconds' % (time() - t0))

    # Do GROG with primefac
    t0 = time()
    res_prime = grog(kx, ky, k, N, N, Gx, Gy, use_primefac=True)
    print('Gridded in %g seconds (primefac)' % (time() - t0))

    res = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(
        res, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))
    res = np.sqrt(np.sum(np.abs(res)**2, axis=-1))

    res_prime = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(
        res_prime, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))
    res_prime = np.sqrt(np.sum(np.abs(res_prime)**2, axis=-1))

    # So Ed doesn't get mad at me...
    res = np.flipud(np.fliplr(res))
    res_prime = np.flipud(np.fliplr(res_prime))

    nx, ny = 1, 3
    plt_opts = {
        'vmin': 0,
        'vmax': np.max(np.concatenate((res, res_prime)).flatten()),
        'cmap': 'gray'
    }
    fig = plt.figure()
    plt.subplot(nx, ny, 1)
    plt.imshow(res, **plt_opts)
    plt.title('SC-GROG')

    plt.subplot(nx, ny, 2)
    plt.imshow(res_prime, **plt_opts)
    plt.title('Proposed')

    residual = np.abs(res - res_prime)
    scale_fac = int(plt_opts['vmax']/np.max(residual.flatten()) + .5)

    plt.subplot(nx, ny, 3)
    plt.imshow(residual*scale_fac, **plt_opts)
    plt.title('Residual (x%d)' % scale_fac)

    msg0 = 'NRMSE: %.3e' % compare_nrmse(res, res_prime)
    plt.annotate(
        msg0, xy=(1, 0), xycoords='axes fraction',
        fontsize=10, xytext=(-5, 5),
        textcoords='offset points', color='white',
        ha='right', va='bottom')

    # Remove ticks
    allaxes = fig.get_axes()
    for ax in allaxes:
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
