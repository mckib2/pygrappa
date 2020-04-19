'''3D implementation of GRAPPA.'''

from collections import defaultdict

import numpy as np
from skimage.util import view_as_windows
from tqdm import tqdm

def _find_acs(kspace, mask):
    '''Given kspaces for each coil and a mask, find the ACS region.'''
    raise NotImplementedError()

def grappa(kspace, calib=None, kernel_size=None, coil_axis=-1, lamda=0.01):
    '''Multidimensional GRAPPA.'''

    # coils to the back
    kspace = np.moveaxis(kspace, coil_axis, -1)
    nc = kspace.shape[-1]

    # User can supply calibration region separately or we can find it
    if calib is not None:
        calib = np.moveaxis(calib, coil_axis, -1)
    else:
        # Find the calibration region and split it out from kspace
        kspace, calib = _find_acs(kspace)

    # Pad the arrays
    pads = [int(k/2) for k in kernel_size]
    adjs = [np.mod(k, 2) for k in kernel_size]
    kspace = np.pad(kspace, [(pd, pd) for pd in pads] + [(0, 0)], mode='constant')
    calib = np.pad(calib, [(pd, pd) for pd in pads] + [(0, 0)], mode='constant')

    # Find all the unique sampling patterns
    mask = np.abs(kspace[..., 0]) > 0
    P = defaultdict(list)
    for idx in np.argwhere(~mask[tuple([slice(pd, -pd) for pd in pads])]):
        p0 = tuple(mask[tuple([slice(ii, ii+2*pd+adj) for ii, pd, adj in zip(idx, pads, adjs)])].flatten().astype(int))
        P[p0].append(idx)

    # We need all overlapping patches from calibration data
    A = view_as_windows(
        calib, tuple(kernel_size) + (nc,)).reshape((-1, np.prod(kernel_size), nc,))

    # Train and apply kernels
    ctr = np.ravel_multi_index([pd for pd in pads], dims=kernel_size)
    recon = np.empty(kspace.shape, dtype=kspace.dtype)
    for key, holes in tqdm(P.items(), desc='Train/apply weights', leave=False):

        # Get sampling pattern from key
        p0 = np.array(p0, dtype=bool)

        # Train kernels
        S = A[:, p0, :].reshape(A.shape[0], -1)
        T = A[:, ctr, :]
        ShS = S.conj().T @ S
        ShT = S.conj().T @ T
        lamda0 = lamda*np.linalg.norm(ShS)/ShS.shape[0]
        W = np.linalg.solve(
            ShS + lamda0*np.eye(ShS.shape[0]), ShT)

        # Apply kernel to each hole
        for idx in holes:
            S = kspace[tuple([slice(ii, ii+2*pd+adj) for ii, pd, adj in zip(idx, pads, adjs)] + [slice(None)])].reshape((-1, nc))[p0, :].flatten()
            recon[tuple([ii + pd for ii, pd in zip(idx, pads)] + [slice(None)])] = S @ W

    # Add back in the measured voxels, put axis back where it goes
    recon[mask] += kspace[mask]
    return np.moveaxis(
        recon[tuple([slice(pd, -pd) for pd in pads] + [slice(None)])], -1, coil_axis)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
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
    #imspace = imspace[:, :-5, :]
    print(imspace.shape)
    ax = (0, 1)
    kspace = 1/np.sqrt(N**2)*np.fft.fftshift(np.fft.fft2(
        np.fft.ifftshift(imspace, axes=ax), axes=ax), axes=ax)

    # crop 20x20 window from the center of k-space for calibration
    pd = 10
    ctr = int(N/2)
    calib = kspace[ctr-pd:ctr+pd, ctr-pd:ctr+pd, :].copy()

    # calibrate a kernel
    kernel_size = (7, 8)

    # undersample by a factor of 2 in both kx and ky
    kspace[::2, 1::2, :] = 0
    kspace[1::2, ::2, :] = 0

    # Do the recon
    res = grappa(kspace, calib, kernel_size)


    # Take a look
    res = np.abs(np.sqrt(N**2)*np.fft.fftshift(np.fft.ifft2(
        np.fft.ifftshift(res, axes=ax), axes=ax), axes=ax))
    M, N = res.shape[:2]
    res0 = np.zeros((2*M, 2*N))
    kk = 0
    for idx in np.ndindex((2, 2)):
        ii, jj = idx[:]
        res0[ii*M:(ii+1)*M, jj*N:(jj+1)*N] = res[..., kk]
        kk += 1
    plt.imshow(res0, cmap='gray')
    plt.show()
