'''Basic usage of NL-GRAPPA.'''

import numpy as np
from phantominator import shepp_logan

from pygrappa import nlgrappa
from utils import gaussian_csm

if __name__ == '__main__':

    N, nc = 256, 16
    ph = shepp_logan(N)[..., None]*gaussian_csm(N, N, nc)

    # Put into kspace
    ax = (0, 1)
    kspace = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(
        ph, axes=ax), axes=ax), axes=ax)

    # 20x20 calibration region
    ctr = int(N/2)
    pad = 10
    calib = kspace[ctr-pad:ctr+pad, ctr-pad:ctr+pad, :].copy()

    # Undersample: R=4
    kspace4x1 = kspace.copy()
    kspace4x1[1::4, ...] = 0
    kspace4x1[2::4, ...] = 0
    kspace4x1[3::4, ...] = 0

    nlgrappa(kspace, calib)
