'''Example demonstrating how to use TGRAPPA.'''

import numpy as np
from phantominator import dynamic

from pygrappa import tgrappa
from utils import gaussian_csm

if __name__ == '__main__':

    # Simulation parameters
    N = 128 # in-plane resolution: (N, N)
    nt = 40 # number of time frames
    ncoil = 4 # number of coils
    R = 4 # undersampling factor

    # Make a simple phantom
    ph = dynamic(N, nt)

    # Apply coil sensitivities
    csm = gaussian_csm(N, N, ncoil)
    ph = ph[:, :, None, :]*csm[..., None]

    # Throw into kspace
    kspace = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(
        ph, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))

    # Undersample
    for ii in range(R):
        kspace[ii::R, ..., ii::R] = 0

    # Reconstuct using TGRAPPA algorithm
    res = tgrappa(kspace)
