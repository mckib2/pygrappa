'''Python implementation of the PRUNO algorithm.'''

import numpy as np
from skimage.util import pad, view_as_windows
from scipy.linalg import null_space
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import LinearOperator
from scipy.signal import convolve2d
from scipy.signal import lfilter
from tqdm import trange


def pruno(kspace, calib, kernel_size=(5, 5), coil_axis=-1):
    '''Parallel Reconstruction Using Null Operations (PRUNO).

    Parameters
    ----------

    Returns
    -------

    References
    ----------
    .. [1] Zhang, Jian, Chunlei Liu, and Michael E. Moseley.
           "Parallel reconstruction using null operations." Magnetic
           resonance in medicine 66.5 (2011): 1241-1253.
    '''

    # Coils to da back
    kspace = np.moveaxis(kspace, coil_axis, -1)
    calib = np.moveaxis(calib, coil_axis, -1)
    nx, ny, _nc = kspace.shape[:]

    # Make a calibration matrix
    kx, ky = kernel_size[:]
    kx2, ky2 = int(kx/2), int(ky/2)
    nc = calib.shape[-1]

    # Pad and pull out calibration matrix
    kspace = pad(  # pylint: disable=E1102
        kspace, ((kx2, kx2), (ky2, ky2), (0, 0)), mode='constant')
    calib = pad(  # pylint: disable=E1102
        calib, ((kx2, kx2), (ky2, ky2), (0, 0)), mode='constant')
    C = view_as_windows(
        calib, (kx, ky, nc)).reshape((-1, kx*ky*nc))

    # Get the nulling kernels
    n = null_space(C, rcond=1e-3)  # TODO: automate selection of rcond
    print(n.shape)

    # Calculate composite kernels
    # TODO: not sure if this is doing the right thing...
    n = np.reshape(n, (-1, nc, n.shape[-1]))
    nconj = np.conj(n).T
    eta = np.zeros((nc, nc, n.shape[0]*2 - 1), dtype=kspace.dtype)
    print(n.shape)
    for ii in trange(nc):
        for jj in range(nc):
            eta[ii, jj, :] = np.sum(convolve2d(
                nconj[:, ii, :], n[:, jj, :], mode='full'), -1)
    print(eta.shape)

    # Solve for b (setting up Ax = b):
    #     b = -Im @ NhN @ Ia @ d
    # Treat NhN as a filer using composite kernels as weights.
    # TODO: include ACS
    # TODO: Not sure if this is doing the right thing...
    b = np.zeros((np.prod(kspace.shape[:2]), nc), dtype=kspace.dtype)
    for ii in range(nc):
        res = np.zeros(b.shape[0], dtype=b.dtype)
        for jj in range(nc):
            res += lfilter(
                eta[ii, jj, :], a=1, x=kspace[..., jj].flatten())
        b[:, ii] = res
    Im = (np.abs(kspace[..., 0]) == 0).flatten()
    b = -1*b*Im[..., None]
    b = b.flatten()

    # import matplotlib.pyplot as plt
    # b = np.reshape(b[:, 1], kspace.shape[:2])
    # plt.imshow(np.abs(b))
    # # im = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(
    # #     b, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))
    # # plt.imshow(np.abs(im))
    # plt.show()

    # Initial guess is zeros
    # TODO: include ACS
    x0 = np.zeros(kspace.size, dtype=kspace.dtype)

    # Conjugate gradient iterations to solve.  To use scipy's CG
    # solver, must be in the form Ax = b, A is a linear operator.
    # Need to do some stuff to create A since nulling operation
    # is filtering
    nx, ny = kspace.shape[:2]

    def _mv(v):
        v = np.reshape(v, (-1, nc))
        res = np.zeros((v.shape[0], nc), dtype=v.dtype)
        for ii in range(nc):
            for jj in range(nc):
                res[..., ii] += lfilter(
                    eta[ii, jj, :], a=1, x=v[:, jj])
        res = res*Im[..., None]
        return res.flatten()

    A = LinearOperator(
        dtype=kspace.dtype, shape=(nc*nx*ny, nc*nx*ny), matvec=_mv)

    d, info = cg(A, b, x0=x0, maxiter=100)
    print(info)
    print(d.shape)

    return np.moveaxis(np.reshape(d, (nx, ny, nc)), -1, coil_axis)


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from phantominator import shepp_logan
    from pygrappa.utils import gaussian_csm

    N, nc = 128, 8
    ph = shepp_logan(N)[..., None]*gaussian_csm(N, N, nc)
    kspace = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(
        ph, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))

    # Get calibration region (20x20)
    pd = 10
    ctr = int(N/2)
    calib = kspace[ctr-pd:ctr+pd, ctr-pd:ctr+pd, :].copy()

    # undersample by a factor of 2 in both kx and ky
    kspace[::2, 1::2, :] = 0
    kspace[1::2, ::2, :] = 0

    res = pruno(kspace, calib)

    im = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(
        res, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))
    sos = np.sqrt(np.sum(np.abs(im)**2, axis=-1))
    plt.imshow(sos)
    plt.show()
