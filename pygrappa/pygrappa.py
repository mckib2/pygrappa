'''Python GRAPPA implementation.

More efficient Python implementation of GRAPPA.

Notes
-----
view_as_windows uses numpy.lib.stride_tricks.as_strided which may
use up a lot of memory.  This is more efficient as we get all the
patches in one go as opposed to looping over the image in multiple
dimensions.  These are be stored in temporary memmaps so we don't
crash anyone's computer.

We are looping over unique sampling patterns, similar to Miki Lustig's
key-lookup table for kernels.  It might be nice to train multiple
kernel geometries simultaneously if possible, or at least have an
option to do chunks at a time.

Currently each hole in kspace is being looped over when applying
weights for a single kernel type.  It would be nice to apply the
weights for all corresponding holes simultaneously.

Tikhonov regularization is not being performed in the least squares
fit.  Probably should try to implement the least squares to be solved
using numpy.linalg.solve and not explicitly compute the inverse.
'''

from time import time
from tempfile import NamedTemporaryFile as NTF

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

    # Pad kspace data
    kspace = pad( # pylint: disable=E1102
        kspace, ((kx2, kx2), (ky2, ky2), (0, 0)), mode='constant')
    calib = pad( # pylint: disable=E1102
        calib, ((kx2, kx2), (ky2, ky2), (0, 0)), mode='constant')
    mask = np.abs(kspace) > 0

    # Store windows in temporary files so we don't overwhelm memory
    with NTF() as fP, NTF() as fA:

        # Start the clock...
        t0 = time()

        # Get all overlapping patches from the mask
        P = np.memmap(fP, dtype=mask.dtype, mode='w+', shape=(
            mask.shape[0]-2*kx2, mask.shape[1]-2*ky2, 1, kx, ky, nc))
        P = view_as_windows(mask, (kx, ky, nc))
        Psh = P.shape[:] # save shape for unflattening indices later
        P = P.reshape((-1, kx, ky, nc))

        # Find the unique patches and associate them with indices
        P, iidx = np.unique(P, return_inverse=True, axis=0)

        # Filter out geometries that don't have a hole at the center.
        # These are all the kernel geometries we actually need to
        # compute weights for. Notice that all coils have same
        # sampling pattern, so choose the 0th one arbitrarily
        validP = np.argwhere(P[:, kx2, ky2, 0] == 0).squeeze()

        # Get all overlapping patches of ACS
        A = np.memmap(fA, dtype=calib.dtype, mode='w+', shape=(
            calib.shape[0]-2*kx, calib.shape[1]-2*ky, 1, kx, ky, nc))
        A = view_as_windows(
            calib, (kx, ky, nc)).reshape((-1, kx, ky, nc))

        # Report on how long it took to construct windows
        print('Data set up took %g seconds' % (time() - t0))

        # Train weights and apply them for each hole we have in
        # kspace data:
        recon = np.zeros(kspace.shape, dtype=kspace.dtype)
        t0 = time()
        for ii in validP:
            # Get the sources by masking all patches of the ACS and
            # get targets by taking the center of each patch. Source
            # and targets will have the following sizes:
            #     S : (# samples, N possible patches in ACS)
            #     T : (# coils, N possible patches in ACS)
            # Solve the equation for the weights:
            #     WS = T
            #     WSS^H = TS^H
            #  -> W = TS^H (SS^H)^-1
            S = A[:, P[ii, ...]].T # transpose to get correct shape
            T = A[:, kx2, ky2, :].T
            TSh = T @ S.conj().T
            SSh = S @ S.conj().T
            W = TSh @ np.linalg.pinv(SSh) # NOTE: inv won't work here

            # Now that we know the weights, let's apply them!  Find
            # all holes corresponding to current geometry.
            # Currently we're looping through all the points
            # associated with the current geometry.  It would be nice
            # to find a way to apply the weights to everything at
            # once.  Right now I don't know how to simultaneously
            # pull all source patches from kspace faster than a
            # forloop

            # x, y define where top left corner is, so move to ctr,
            # also make sure they are iterable by enforcing atleast_1d
            idx = np.unravel_index(
                np.argwhere(iidx == ii), Psh[:2])
            x, y = idx[0]+kx2, idx[1]+ky2
            x = np.atleast_1d(x.squeeze())
            y = np.atleast_1d(y.squeeze())
            for xx, yy in zip(x, y):
                # Collect sources for this hole and apply weights
                S = kspace[xx-kx2:xx+kx2+1, yy-ky2:yy+ky2+1, :]
                S = S[P[ii, ...]]
                recon[xx, yy, :] = (W @ S[:, None]).squeeze()

    # Report on how long it took to train and apply weights
    print(('Training and application of weights took %g seconds'
           '' % (time() - t0)))

    # Fill in known data, crop, move coil axis back
    return np.moveaxis(
        (recon + kspace)[kx2:-kx2, ky2:-ky2, :], -1, coil_axis)

if __name__ == '__main__':

    from phantominator import shepp_logan

    # Generate fake sensitivity maps: mps
    N = 256
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
