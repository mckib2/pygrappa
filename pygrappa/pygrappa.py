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

    # Pad kspace
    kspace = pad( # pylint: disable=E1102
        kspace, ((kx2, kx2), (ky2, ky2), (0, 0)), mode='constant')
    mask = np.abs(kspace) > 0

    # Get all overlapping patches from the mask and find the unique
    # patches -- these are all the kernel geometries.  We will also
    # return mapping of each hole to the unique kernels
    P = view_as_windows(mask, (kx, ky, nc)).reshape((-1, kx, ky, nc))
    P, Pidx = np.unique(P, return_inverse=True, axis=0)

    # Filter out geometries that don't have a hole at the center
    P = np.moveaxis(P, -1, 0)
    test = P[..., kx2, ky2] == 0
    idx = np.argwhere(P[..., kx2, ky2][0] == 0).squeeze().tolist()
    Pidx = Pidx.squeeze().tolist()
    Pidx = np.array([x for x in Pidx if x not in idx])
    P = P[test, ...].reshape((nc, -1, kx, ky))
    P = np.moveaxis(P, 0, -1)


    # Get the sources of defined by the geometries by masking all
    # patches of the ACS, targets by taking the center.  Source and
    # targets will have the following sizes:
    #     S : (# samples, N possible patches in ACS)
    #     T : (# coils, N possible patches in ACS)
    # Solve the equation for the weights:
    #     WS = T
    #     WSS^H = TS^H
    #  -> W = TS^H (SS^H)^-1
    A = view_as_windows(calib, (kx, ky, nc)).reshape((-1, kx, ky, nc))
    print(A.shape)
    recon = np.zeros(kspace.shape, dtype=kspace.dtype)
    for ii in range(P.shape[0]):
        S = A[:, P[ii, ...]].T # transpose to get correct shape
        T = A[:, kx2, ky2, :].T
        TSh = T @ S.conj().T
        SSh = S @ S.conj().T
        W = TSh @ np.linalg.pinv(SSh)

        # Now that we know the weights, let's apply them!  Find all
        # holes corresponding to this geometry and fill them all in

        # We need to find all the holes that have geometry P[ii, ...].
        # This is not a great way to do it...
        for xx in range(kx, mask.shape[0]-kx+1):
            for yy in range(ky, mask.shape[1]-ky+1):
                mask0 = mask[xx-kx2:xx+kx2+1, yy-ky2:yy+ky2+1, :]
                if np.all(mask0 == P[ii, ...]):
                    S = kspace[xx-kx2:xx+kx2+1, yy-ky2:yy+ky2+1, :]
                    S = S[P[ii, ...]]
                    recon[xx, yy, :] = (W @ S[:, None]).squeeze()

    # Fill in known data and send it on back
    return recon + kspace

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

    # Take a gander
    from mr_utils import view
    view(res, log=True)
    view(res, fft=True)
