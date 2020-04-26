'''Naive implementation to make sure we know what's going on.'''

import numpy as np
from skimage.util import view_as_windows
from scipy.linalg import null_space
from scipy.signal import convolve2d

import matplotlib.pyplot as plt
from phantominator import shepp_logan


def simple_pruno(
        kspace, calib, kernel_size=(5, 5), coil_axis=-1,
        sens=None, ph=None, kspace_ref=None):
    '''PRUNO.'''

    # Coils to da back
    kspace = np.moveaxis(kspace, coil_axis, -1)
    calib = np.moveaxis(calib, coil_axis, -1)
    nx, ny, _nc = kspace.shape[:]

    # Make a calibration matrix
    kx, ky = kernel_size[:]
    # kx2, ky2 = int(kx/2), int(ky/2)
    nc = calib.shape[-1]

    # Pull out calibration matrix
    C = view_as_windows(
        calib, (kx, ky, nc)).reshape((-1, kx*ky*nc))

    # Get the nulling kernels
    n = null_space(C, rcond=1e-3)
    print(n.shape)

    # Test to see if nulling kernels do indeed null
    if sens is not None:
        ws = 8  # full width of sensitivity map spectra
        wd = kx
        wm = wd + ws - 1

        # Choose a target
        xx, yy = int(nx/3), int(ny/3)

        # Get source
        wm2 = int(wm/2)
        S = ph[xx-wm2:xx+wm2, yy-wm2:yy+wm2].copy()
        assert (wm, wm) == S.shape

        sens_spect = 1/np.sqrt(N**2)*np.fft.fftshift(np.fft.fft2(
            np.fft.ifftshift(sens, axes=(0, 1)),
            axes=(0, 1)), axes=(0, 1))

        # Get the target
        ctr = int(sens_spect.shape[0]/2)
        ws2 = int(ws/2)
        T = []
        for ii in range(nc):
            sens0 = sens_spect[
                ctr-ws2:ctr+ws2, ctr-ws2:ctr+ws2, ii].copy()
            T.append(convolve2d(S, sens0, mode='valid'))
        T = np.moveaxis(np.array(T), 0, -1)
        assert (wd, wd, nc) == T.shape

        # Find local encoding matrix
        #     E S = T
        #
        ShS = S.conj().T @ S
        print(ShS.shape, T.shape)
        ShT = S.conj().T @ T
        print(ShS.shape, ShT.shape)
        E = np.linalg.solve(ShS, ShT).T
        print(E.shape)


if __name__ == '__main__':

    # Generate fake sensitivity maps: mps
    N = 32
    ncoils = 4
    xx = np.linspace(0, 1, N)
    x, y = np.meshgrid(xx, xx)
    mps = np.zeros((N, N, ncoils))
    mps[..., 0] = x**2
    mps[..., 1] = 1 - x**2
    mps[..., 2] = y**2
    mps[..., 3] = 1 - y**2

    # generate 4 coil phantom
    ph = shepp_logan(N)
    imspace = ph[..., None]*mps
    imspace = imspace.astype('complex')
    ax = (0, 1)
    kspace = 1/np.sqrt(N**2)*np.fft.fftshift(np.fft.fft2(
        np.fft.ifftshift(imspace, axes=ax), axes=ax), axes=ax)
    kspace_ref = kspace.copy()

    ph = 1/np.sqrt(N**2)*np.fft.fftshift(np.fft.fft2(
        np.fft.ifftshift(ph, axes=ax), axes=ax), axes=ax)

    # crop 20x20 window from the center of k-space for calibration
    pd = 10
    ctr = int(N/2)
    calib = kspace[ctr-pd:ctr+pd, ctr-pd:ctr+pd, :].copy()

    # calibrate a kernel
    kernel_size = (5, 5)

    # undersample by a factor of 2 in both kx and ky
    kspace[::2, 1::2, :] = 0
    kspace[1::2, ::2, :] = 0

    # reconstruct:
    res = simple_pruno(
        kspace, calib, kernel_size, coil_axis=-1,
        sens=mps, ph=ph, kspace_ref=kspace_ref)
    assert False

    # Take a look
    res = np.abs(np.sqrt(N**2)*np.fft.fftshift(np.fft.ifft2(
        np.fft.ifftshift(res, axes=ax), axes=ax), axes=ax))
    res0 = np.zeros((2*N, 2*N))
    kk = 0
    for idx in np.ndindex((2, 2)):
        ii, jj = idx[:]
        res0[ii*N:(ii+1)*N, jj*N:(jj+1)*N] = res[..., kk]
        kk += 1
    plt.imshow(res0, cmap='gray')
    plt.show()
