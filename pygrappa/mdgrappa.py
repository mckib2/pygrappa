'''Multidimensional implementation of GRAPPA.'''

from collections import defaultdict

import numpy as np
from skimage.util import view_as_windows
from tqdm import tqdm

def _find_acs(kspace, mask):
    '''Given kspaces for each coil and a mask, find the ACS region.'''
    raise NotImplementedError()

def grappa(kspace, calib=None, kernel_size=None, coil_axis=-1, lamda=0.01, nnz=None):
    '''Multidimensional GRAPPA.

    Parameters
    ----------
    nnz : int or None, optional
        Number of nonzero elements in a multidimensional patch
        required to train/apply a kernel.
        Default: `sqrt(prod(kernel_size))`.

    Notes
    -----
    All axes (except coil axis) are used for GRAPPA reconstruction.
    If you desire to exlude an axis, say `ignored_axis`, set
    `kernel_size[ignored_axis] = 1`.
    '''

    # coils to the back
    kspace = np.moveaxis(kspace, coil_axis, -1)
    nc = kspace.shape[-1]

    if kernel_size is None:
        kernel_size = tuple([5]*(kspace.ndim-1))

    # Only consider sampling patterns that have at least nnz samples
    if nnz is None:
        nnz = int(np.sqrt(np.prod(kernel_size)))

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
        p0 = mask[tuple([slice(ii, ii+2*pd+adj) for ii, pd, adj in zip(idx, pads, adjs)])].flatten()
        if np.sum(p0) >= nnz: # only counts if it has enough samples
            P[tuple(p0.astype(int))].append(idx)

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

        # Doesn't seem to be a big difference in speed?
        # Try gathering all sources and doing single matrix multiply
        #S = np.empty((len(holes), W.shape[0]), dtype=kspace.dtype)
        #targets = np.empty((kspace.ndim-1, len(holes)), dtype=int)
        #for jj, idx in enumerate(holes):
        #    S[jj, :] = kspace[tuple([slice(ii, ii+2*pd+adj) for ii, pd, adj in zip(idx, pads, adjs)] + [slice(None)])].reshape((-1, nc))[p0, :].flatten()
        #    targets[:, jj] = [ii + pd for ii, pd in zip(idx, pads)]
        #recon = np.reshape(recon, (-1, nc))
        #targets = np.ravel_multi_index(targets, dims=kspace.shape[:-1])
        #recon[targets, :] = S @ W
        #recon = np.reshape(recon, kspace.shape)

        # Apply kernel to each hole
        for idx in holes:
            S = kspace[tuple([slice(ii, ii+2*pd+adj) for ii, pd, adj in zip(idx, pads, adjs)] + [slice(None)])].reshape((-1, nc))[p0, :].flatten()
            recon[tuple([ii + pd for ii, pd in zip(idx, pads)] + [slice(None)])] = S @ W

    # Add back in the measured voxels, put axis back where it goes
    recon[mask] += kspace[mask]
    return np.moveaxis(
        recon[tuple([slice(pd, -pd) for pd in pads] + [slice(None)])], -1, coil_axis)

if __name__ == '__main__':
    from time import time
    import matplotlib.pyplot as plt
    from phantominator import shepp_logan
    from utils import gaussian_csm

    # Generate fake sensitivity maps: mps
    L, M, N = 128, 128, 32
    ncoils = 4
    mps = gaussian_csm(L, M, ncoils)[..., None, :]

    # generate phantom
    ph = shepp_logan((L, M, N), zlims=(-.25, .25))
    imspace = ph[..., None]*mps
    #imspace = imspace[:-1, :-1, :]
    print(imspace.shape)
    ax = (0, 1, 2)
    kspace = 1/np.sqrt(N**2)*np.fft.fftshift(np.fft.fftn(
        np.fft.ifftshift(imspace, axes=ax), axes=ax), axes=ax)

    # crop 20x20 window from the center of k-space for calibration
    # (use all z-axis)
    pd = 10
    ctrs = [int(s/2) for s in kspace.shape[:2]]
    calib = kspace[tuple([slice(ctr-pd, ctr+pd) for ctr in ctrs] + [slice(None), slice(None)])].copy()
    #calib = kspace[ctr-pd:ctr+pd, ctr-pd:ctr+pd, :].copy()

    # calibrate a kernel
    kernel_size = (4, 5, 4)

    # undersample by a factor of 2 in both kx and ky
    kspace[::2, 1::2, ...] = 0
    kspace[1::2, ::2, ...] = 0

    # Do the recon
    t0 = time()
    res = grappa(kspace, calib, kernel_size)
    print('Took %g sec' % (time() - t0))

    # Take a look at a single slice
    res = np.abs(np.sqrt(N**2)*np.fft.fftshift(np.fft.ifftn(
        np.fft.ifftshift(res, axes=ax), axes=ax), axes=ax))
    res = res[..., 0, :]
    res0 = np.zeros((2*L, 2*M))
    kk = 0
    for idx in np.ndindex((2, 2)):
        ii, jj = idx[:]
        res0[ii*L:(ii+1)*L, jj*M:(jj+1)*M] = res[..., kk]
        kk += 1
    plt.imshow(res0, cmap='gray')
    plt.show()
