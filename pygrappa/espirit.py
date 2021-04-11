'''ESPIRiT—an eigenvalue approach to autocalibrating parallel MRI.

Where SENSE meets GRAPPA.
'''

import logging
from time import time
import pathlib

import numpy as np
from skimage.util import view_as_windows


logger = logging.getLogger(name=pathlib.Path(__file__).name)
logging.basicConfig(level=logging.INFO)


def _fft(x, s=None, axes=None):
    return np.fft.fftshift(
        np.fft.fft2(
            np.fft.ifftshift(x, axes=axes),
            s=s, axes=axes), axes=axes)


def espirit(kspace, calib, kernel_size=None, eig_thresh_1=0.02, eig_thresh_2=0.95, coil_axis=-1):
    '''ESPIRiT Python implementation.'''

    kspace = np.moveaxis(kspace, coil_axis, -1)
    calib = np.moveaxis(calib, coil_axis, -1)
    sx, sy, nc = kspace.shape[:]

    # default kernel size if None given
    if kernel_size is None:
        kernel_size = (6, 6)
        logger.info('Choosing default kernel size %s', kernel_size)
    kx, ky = kernel_size

    # train kernel
    A = view_as_windows(calib, (kx, ky, nc)).reshape((-1, kx*ky*nc))
    t0 = time()
    _u, s, vh = np.linalg.svd(A, full_matrices=False, compute_uv=True)
    v = np.conj(vh).T
    logger.info('First SVD size %s took %g seconds',
                A.shape, time()-t0)
    k = np.reshape(v, (kx, ky, nc, v.shape[1]))
    nv = s[s >= eig_thresh_1*s[0]].size
    k = k[..., :nv]
    logger.info('Found %d significant (thresh=%g) singular values',
                nv, eig_thresh_1*s[0])

    k = np.moveaxis(k, 3, 2)
    k = np.reshape(k, (kx*ky*nv, nc))
    t0 = time()
    _u, _s, vh = np.linalg.svd(
        k, full_matrices=k.shape[0] >= k.shape[1], compute_uv=True)
    v = np.conj(vh).T
    logger.info('Second SVD size %s took %g seconds',
                k.shape, time()-t0)
    k = np.einsum('ij,jk->ik', k, v)
    k = np.reshape(k, (kx, ky, nv, nc))
    k = np.moveaxis(k, 3, 2)
    ax = (0, 1)
    t0 = time()
    print(k.shape)
    # k = _fft(np.conj(k[::-1, ::-1, ...]), s=(sx, sy), axes=ax)/np.sqrt(kx*ky)  # TODO: why is this not equivalent to following?
    xpd = sx // 2 - kx // 2
    ypd = sy // 2 - ky // 2  # TODO: might need to consider odd/even cases
    KERNEL = np.zeros((sx, sy, k.shape[2], k.shape[3]), dtype=k.dtype)
    for n in range(k.shape[3]):
        KERNEL[..., n] = _fft(
            np.pad(np.conj(k[::-1, ::-1, :, n]), ((xpd, xpd), (ypd, ypd), (0, 0))),
            axes=ax)
    KERNEL /= np.sqrt(kx*ky)
    k = KERNEL
    logger.info('FFT took %g seconds', time()-t0)
    
    t0 = time()
    c, d, _vh = np.linalg.svd(k, full_matrices=False, compute_uv=True)
    logger.info('Third SVD size %s took %g seconds',
                k.shape, time()-t0)
    ph = np.exp(-1j*np.angle(c[..., 0, :]))[..., None, :]  # TODO: something fishy is happening here -- phase of M doesn't look like MATLAB output
    M = np.einsum('ij,xyjk->xyik', v, c*ph)[..., ::-1]
    W = np.real(d)[..., ::-1]
    
    print('M is', M.shape)

    import matplotlib.pyplot as plt

    idx = 1
    for ii in range(min(nc, nv)):
        plt.subplot(1, nc, idx)
        plt.imshow(np.abs(W[..., ii]))
        idx += 1
    plt.title('W')
    plt.show()
    
    idx = 1
    for ii in range(nc):
        for jj in range(min(nc, nv)):
            plt.subplot(nc, nc, idx)
            plt.imshow(np.abs(M[..., jj, ii]))
            idx += 1
    plt.show()

    idx = 1
    for ii in range(nc):
        for jj in range(min(nc, nv)):
            plt.subplot(nc, nc, idx)
            plt.imshow(np.angle(M[..., jj, ii]))
            idx += 1
    plt.show()


if __name__ == '__main__':
    # load data
    import matplotlib.pyplot as plt
    from scipy.io import loadmat
    # from pygrappa import find_acs
    matdata = loadmat('/home/nmckibben/Documents/SPIRiT_v0.3/ESPIRiT/data/brain_8ch.mat')
    data = matdata['DATA']
    mask = matdata['mask_unif']
    # data = data*mask[..., None]
    # plt.imshow(np.abs(_fft(data[..., 0], axes=(0, 1))))
    # plt.show()
    # plt.imshow(np.abs(data[..., 0]*mask))
    # plt.show()
    # calib = find_acs(data, coil_axis=-1)
    xctr, yctr = data.shape[0] // 2, data.shape[1] // 2
    sz = 24
    sz2 = sz // 2
    calib = data[xctr-sz2:xctr+sz2, yctr-sz2:yctr+sz2, :].copy()
    print('calib:', calib.shape)
    espirit(data, calib)
