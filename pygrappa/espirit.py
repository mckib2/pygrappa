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
        np.fft.fftn(
            np.fft.ifftshift(x, axes=axes),
            s=s, axes=axes), axes=axes)


def _espirit_kernel(kspace_sh, axes, calib, kernel_size, eig_thresh_1, eig_thresh_2):
    # train kernel
    nc = calib.shape[-1]
    A = view_as_windows(calib, kernel_size + (nc,)).reshape((-1, np.prod(kernel_size)*nc))
    t0 = time()
    _u, s, vh = np.linalg.svd(A, full_matrices=False, compute_uv=True)
    v = np.conj(vh).T
    logger.info('First SVD size %s took %g seconds',
                A.shape, time()-t0)
    k = np.reshape(v, kernel_size + (nc, v.shape[1]))
    nv = s[s >= eig_thresh_1*s[0]].size
    k = k[..., :nv]
    logger.info('Found %d significant (thresh=%g) singular values',
                nv, eig_thresh_1*s[0])

    k = np.moveaxis(k, -1, -2)
    k = np.reshape(k, (np.prod(kernel_size)*nv, nc))
    t0 = time()
    _u, _s, vh = np.linalg.svd(
        k, full_matrices=k.shape[0] >= k.shape[1], compute_uv=True)
    v = np.conj(vh).T
    logger.info('Second SVD size %s took %g seconds',
                k.shape, time()-t0)
    k = np.einsum('ij,jk->ik', k, v)
    k = np.reshape(k, kernel_size + (nv, nc))
    k = np.moveaxis(k, -1, -2)
    t0 = time()
    pds = (ss//2 - kk//2 for ss, kk in zip(kspace_sh[:-1], kernel_size))  # TODO: consider odd/even cases
    k = _fft(np.pad(np.conj(k[::-1, ::-1, ...]),
                    tuple((pd, pd) for pd in pds) + ((0, 0), (0, 0))),
             axes=axes)/np.sqrt(np.prod(kspace_sh[:-1]))
    #KERNEL = np.zeros((sx, sy, k.shape[2], k.shape[3]), dtype=k.dtype)
    #for n in range(k.shape[3]):
    #    KERNEL[..., n] = _fft(
    #        np.pad(np.conj(k[::-1, ::-1, :, n]), ((xpd, xpd), (ypd, ypd), (0, 0))),
    #        axes=ax)
    #KERNEL /= np.sqrt(kx*ky)
    #k = KERNEL
    logger.info('FFT took %g seconds', time()-t0)

    t0 = time()
    c, d, _vh = np.linalg.svd(k, full_matrices=True, compute_uv=True)
    logger.info('Third SVD size %s took %g seconds',
                k.shape, time()-t0)
    # TODO: something fishy is happening here -- phase of M doesn't look like MATLAB output
    #ph = np.exp(-1j*np.angle(c[..., 0, :]))[..., None, :]
    ph = 1
    M = np.einsum('ij,xyjk->xyik', v, c*ph)[..., ::-1]
    W = np.real(d)[..., ::-1]

    #import matplotlib.pyplot as plt
    #idx = 1
    #for ii in range(min(nc, nv)):
    #    plt.subplot(1, nc, idx)
    #    plt.imshow(np.abs(W[..., ii]))
    #    idx += 1
    #plt.title('W')
    #plt.show()
    #idx = 1
    #for ii in range(nc):
    #    for jj in range(min(nc, nv)):
    #        plt.subplot(nc, nc, idx)
    #        plt.imshow(np.abs(M[..., jj, ii]))
    #        idx += 1
    #plt.show()
    #idx = 1
    #for ii in range(nc):
    #    for jj in range(min(nc, nv)):
    #        plt.subplot(nc, nc, idx)
    #        plt.imshow(np.angle(M[..., jj, ii]))
    #        idx += 1
    #plt.show()

    return M, W


def espirit(kspace, calib, kernel_size=None, eig_thresh_1=0.02, eig_thresh_2=0.95, axes=None, coil_axis=-1):
    '''ESPIRiT Python implementation.'''

    # default is all axes except coil
    if axes is None:
        axes = tuple(range(kspace.ndim-1))

    kspace = np.moveaxis(kspace, coil_axis, -1)
    calib = np.moveaxis(calib, coil_axis, -1)

    # convert axes to match rearranged dimensions
    new_axes = list(axes)
    if coil_axis < 0:
        coil_axis += kspace.ndim
    for ii, ax in enumerate(axes):
        if ax < 0:
            new_axes[ii] += kspace.ndim
        if ax > coil_axis:
            new_axes[ii] -= 1
    axes = tuple(new_axes)
    sh, nc = kspace.shape[:-1], kspace.shape[-1]

    # default kernel size if None given
    if kernel_size is None:
        kernel_size = tuple([6,]*len(axes))
        logger.info('Choosing default kernel size %s', kernel_size)

    # do kernel training and calculate eigenvalues/vectors
    M, _W = _espirit_kernel(
        kspace.shape, axes, calib, kernel_size, eig_thresh_1, eig_thresh_2)

    # apply kernel to reconstruct
    imspace = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(
        kspace, axes=axes), axes=axes), axes=axes)*np.sqrt(np.prod(sh))
    recon = np.sum(imspace[..., None]*np.conj(M), axis=-2)

    return recon


if __name__ == '__main__':
    # load data
    import matplotlib.pyplot as plt
    from scipy.io import loadmat
    # from pygrappa import find_acs
    matdata = loadmat('/home/nmckibben/Documents/SPIRiT_v0.3/ESPIRiT/data/brain_8ch.mat')
    #matdata = loadmat('/home/nmckibben/Documents/SPIRiT_v0.3/ESPIRiT/data/brain_alias_8ch.mat')
    data = matdata['DATA']
    #mask = matdata['mask_unif']
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
    recon = espirit(data, calib)

    idx = 1
    for ii in range(recon.shape[-1]):
        plt.subplot(1, recon.shape[-1], idx)
        plt.imshow(np.abs(recon[..., ii]))
        idx += 1
    plt.show()
