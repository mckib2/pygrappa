'''Python GRAPPA implementation.'''

import numpy as np
from skimage.util import pad, view_as_windows

def grappa(
        kspace, calib, kernel_size=(5, 5), coil_axis=-1, lamda=0.01,
        disp=False, memmap=False, memmap_filename='out.memmap'):
    '''Now in Python.'''

    # Put the coil dimension at the end
    kspace = np.moveaxis(kspace, coil_axis, -1)
    calib = np.moveaxis(calib, coil_axis, -1)

    # Get displays up and running if we need them
    if disp:
        import matplotlib.pyplot as plt

    # Get shape of kernel
    kx, ky = kernel_size[:]
    kx2, ky2 = int(kx/2), int(ky/2)
    nc = calib.shape[-1]

    # Find the holes
    idx = np.argwhere(kspace == 0)

    # Pad kspace
    kspace = pad( # pylint: disable=E1102
        kspace, ((kx2, kx2), (ky2, ky2), (0, 0)), mode='constant')
    mask = np.abs(kspace) > 0

    # Get all overlapping patches from the mask and find the unique
    # patches -- these are all the kernel geometries.  We will also
    # return mapping of each hole to the unique kernels
    P = view_as_windows(mask, (kx, ky, nc)).reshape((-1, kx, ky, nc))
    P, Pidx = np.unique(P, return_inverse=True, axis=0)
    # print(P.shape)

    # Now that we know the kernel geometries, we will train the
    # weights!
    lamda = .01
    W = np.zeros((P.shape[0], kx*ky*nc), dtype=np.complex64)
    for ii in range(P.shape[0]):

        # Gather up all the sources in the calibration data
        # corresponding to the current kernel geometry
        A = 
        AtA = np.dot(A.conj().T, A)

        # Now we need to current hole target
        hole =

        # Solve the least squares problem: AhA W = hole
        # With Tikhonov regularization, of course...
        W[ii, :] = np.linalg.solve(
            AtA + lamda*np.eye(kx*ky*nc), A.conj().T.dot(hole))[0]


    # # We need weights for each of the holes
    # lookup = dict()
    # for ii, idx0 in enumerate(idx):
    #     xx, yy, cc = idx0[:]
    #
    #     # Kernel mask for this hole
    #     kernel = mask[xx:xx+kx, yy:yy+ky, cc].astype(int)
    #     key = str(kernel.flatten().tolist())
    #
    #     # Associate this hole with this kernel geometry
    #     try:
    #         lookup[key].append(ii)
    #     except KeyError:
    #         lookup[key] = [ii]



    # # Now train weights for each kernel geometry
    # for key in


if __name__ == '__main__':

    from phantominator import shepp_logan

    # Generate fake sensitivity maps: mps
    N = 128
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

    # crop 20x20 window from the center of k-space for calibration
    pd = 10
    ctr = int(N/2)
    calib = kspace[ctr-pd:ctr+pd, ctr-pd:ctr+pd, :].copy()

    # calibrate a kernel
    kernel_size = (5, 5)

    # undersample by a factor of 2 in both x and y
    kspace[::2, 1::2, :] = 0
    kspace[1::2, ::2, :] = 0

    # reconstruct:
    res = grappa(
        kspace, calib, kernel_size, coil_axis=-1, lamda=0.01,
        disp=False, memmap=False)
