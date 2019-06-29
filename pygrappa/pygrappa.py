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

    # # Find the holes
    # idx = np.argwhere(kspace == 0)

    # Pad kspace
    kspace = pad( # pylint: disable=E1102
        kspace, ((kx2, kx2), (ky2, ky2), (0, 0)), mode='constant')
    mask = np.abs(kspace) > 0

    # Get all overlapping patches from the mask and find the unique
    # patches -- these are all the kernel geometries.  We will also
    # return mapping of each hole to the unique kernels
    P = view_as_windows(mask, (kx, ky, nc)).reshape((-1, kx, ky, nc))
    # print(P.shape)
    P, uidx, Pidx = np.unique(
        P, return_index=True, return_inverse=True, axis=0)
    # print(P.shape)
    # print(Pidx)
    # print(uidx)

    # Filter out geometries that don't have a hole at the center
    P = np.moveaxis(P, -1, 0)
    test = P[..., kx2, ky2] == 0
    uidx = uidx[test[0, ...]]
    P = P[test, ...].reshape((nc, -1, kx, ky))
    P = np.moveaxis(P, 0, -1)
    # print(P.shape)
    # print(uidx)

    # Get the sources of defined by the geometries by masking all
    # patches of the ACS, targets by taking the center, and weights
    # by solving:
    #     WS = T => WSS^H = TS^H => W = TS^H (SS^H)^-1
    A = view_as_windows(calib, (kx, ky, nc)).reshape((-1, kx, ky, nc))
    print(A.shape)
    # W = np.zeros((P.shape[0], nc, (kx-1)*(ky-1)), dtype=kspace.dtype)
    W = []
    for ii in range(P.shape[0]):
        # S = A[:, P[ii, ...]]
        # T = A[:, kx2, ky2, :]
        # TSh = T[..., None] @ S[:, None, :].conj()
        # SSh = S[..., None] @ S[:, None, :].conj()
        # W0 = TSh @ np.linalg.pinv(SSh)
        # W0 = W0.sum(0)/np.count_nonzero(W0, axis=0)
        # W.append(W0)
        # # test = W[0, ...] @ S[0, :, None]
        # # assert np.allclose(test, T[0, :, None])

        S = A[:, P[ii, ...]].T
        T = A[:, kx2, ky2, :].T
        print(S.shape, T.shape)
        TSh = T @ S.conj().T
        SSh = S @ S.conj().T
        W0 = TSh @ np.linalg.pinv(SSh)
        W.append(W0)

        # Well, the weights don't seem to be shift invariant, but
        # hopefully averaging the nonzero weights will do the job...
        # for jj in range(A.shape[0]):
        #     S = A[jj, P[ii, ...]]
        #     T = A[jj, kx2, ky2, :]
        #     TSh = T[:, None] @ S[None, :].conj()
        #     SSh = S[:, None] @ S[None, :].conj()
        #     W = TSh @ np.linalg.pinv(SSh)
        #     # test = W @ S[:, None]
        #     # assert np.allclose(test, T[:, None])

    # Now that we know the weights, let's apply them!
    recon = np.zeros(kspace.shape, dtype=kspace.dtype)

    # Find all holes with each geometry and fill them all in
    for ii in range(P.shape[0]):

        for xx in range(kx, mask.shape[0]-kx+1):
            for yy in range(ky, mask.shape[1]-ky+1):

                mask0 = mask[xx-kx2:xx+kx2+1, yy-ky2:yy+ky2+1, :]
                if np.all(mask0 == P[ii, ...]):
                    S = kspace[xx-kx2:xx+kx2+1, yy-ky2:yy+ky2+1, :]
                    S = S[P[ii, ...]]
                    recon[xx, yy, :] = (W[ii] @ S[:, None]).squeeze()
    recon += kspace

    from mr_utils import view
    view(recon, log=True)
    view(recon, fft=True)

    # # Now that we know the kernel geometries, we will train the
    # # weights!
    # lamda = .01
    # W = np.zeros((P.shape[0], kx*ky*nc), dtype=np.complex64)
    # A = view_as_windows(calib, (kx, ky, nc)).reshape((-1, kx, ky, nc))
    # print(A.shape, P.shape)
    # for ii in range(P.shape[0]):
    #
    #     # Gather up all the sources in the calibration data
    #     # corresponding to the current kernel geometry
    #     A0 = np.zeros(A.shape, dtype=A.dtype)
    #     A0[:, P[ii, ...]] = A[:, P[ii, ...]]
    #     A0 = A0.reshape((A.shape[0], -1))
    #     AtA = np.dot(A0.conj().T, A0)
    #     print(AtA.shape)
    #
    #     # Now we need to current hole targets
    #     # hole = kspace[np.unravel_index(idx[Pidx], kspace.shape)]
    #     # hole = kspace[idx[Pidx[np.argwhere(Pidx == ii)]]]
    #     # hole = A[np.argwhere(Pidx == ii)
    #     hole = A[:, kx2, ky2, 0] # for the first coil im for now
    #
    #     # Solve the least squares problem: AhA W = hole
    #     # With Tikhonov regularization, of course...
    #     W[ii, :] = np.linalg.solve(
    #         AtA + lamda*np.eye(kx*ky*nc),
    #         np.dot(A0.conj().T, hole))[0]
    #
    # # Now do the interpolation by applying the weights (first coil)
    # P = view_as_windows(kspace, (kx, ky, nc)).reshape((-1, kx*ky*nc))
    # for ii, jj in enumerate(Pidx):
    #     xx, yy, cc = idx[ii]
    #     kspace[xx, yy, cc] = np.dot(P[ii, ...], W[jj, ...])
    #
    # print(P.shape)
    #
    # from mr_utils import view
    # view(kspace, fft=True)
    #

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
    N = 64
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
