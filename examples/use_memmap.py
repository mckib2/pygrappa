'''Example demonstrating how process datasets stored in memmap.'''

from tempfile import NamedTemporaryFile as NTF

import numpy as np
import matplotlib.pyplot as plt
from phantominator import shepp_logan

from pygrappa import grappa

if __name__ == '__main__':

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

    # Use NamedTemporaryFiles for kspace and reconstruction results
    with NTF() as kspace_file, NTF() as res_file:
        # Make a memmap
        kspace = np.memmap(
            kspace_file, mode='w+', shape=(N, N, ncoils),
            dtype='complex')

        # Fill the memmap with kspace data (remember the [:]!!!)
        ax = (0, 1)
        kspace[:] = 1/np.sqrt(N**2)*np.fft.fftshift(np.fft.fft2(
            np.fft.ifftshift(imspace, axes=ax), axes=ax), axes=ax)

        # crop 20x20 window from the center of k-space for calibration
        pd = 10
        ctr = int(N/2)
        calib = np.array(kspace[ctr-pd:ctr+pd, ctr-pd:ctr+pd, :])
        # Make sure calib is not referencing kspace, it should be a
        # copy so it doesn't change in place!

        # undersample by a factor of 2 in both x and y
        kspace[::2, 1::2, :] = 0
        kspace[1::2, ::2, :] = 0

        # Close the memmap
        del kspace

        # ========================================================== #
        # Open up a new readonly memmap -- this is where you would
        # likely start with data you really wanted to process
        kspace = np.memmap(
            kspace_file, mode='r', shape=(N, N, ncoils),
            dtype='complex')

        # calibrate a kernel
        kernel_size = (5, 5)

        # reconstruct, write res out to a memmap with name res_file
        grappa(
            kspace, calib, kernel_size, coil_axis=-1, lamda=0.01,
            memmap=True, memmap_filename=res_file)

        # Take a look by opening up the memmap
        res = np.memmap(
            res_file, mode='r', shape=(N, N, ncoils),
            dtype='complex')
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
