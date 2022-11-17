'''Show basic usage of NL-GRAPPA MATLAB port.'''

import numpy as np
import matplotlib.pyplot as plt
from phantominator import shepp_logan

from pygrappa import nlgrappa_matlab
from pygrappa.utils import gaussian_csm

if __name__ == '__main__':

    # Generate data
    N, nc = 128, 8
    sens = gaussian_csm(N, N, nc)
    im = shepp_logan(N)
    im = im[..., None]*sens
    sos = np.sqrt(np.sum(np.abs(im)**2, axis=-1))

    off = 0  # starting sampling location

    # The number of ACS lines
    R = 5
    nencode = 42

    # The convolution size
    num_block = 2
    num_column = 15  # make smaller to go quick during development

    # Obtain ACS data and undersampled data
    sx, sy, nc = im.shape[:]
    sx2 = int(sx/2)
    nencode2 = int(nencode/2)
    acs_line_loc = np.arange(sx2 - nencode2, sx2 + nencode2)
    calib = np.fft.fftshift(np.fft.fft2(
        im, axes=(0, 1)), axes=(0, 1))[acs_line_loc, ...].copy()

    # Obtain uniformly undersampled locations
    pe_loc = np.arange(off, sx-off, R)
    kspace_u = np.zeros((pe_loc.size, sy, nc), dtype=im.dtype)
    kspace_u = np.fft.fftshift(np.fft.fft2(
        im, axes=(0, 1)), axes=(0, 1))[pe_loc, ...].copy()  # why do this?

    # Net reduction factor
    acq_idx = np.zeros(sx, dtype=bool)
    acq_idx[pe_loc] = True
    acq_idx[acs_line_loc] = True
    NetR = sx / np.sum(acq_idx)

    # Nonlinear GRAPPA Reconstruction
    times_comp = 3  # The number of times of the first-order terms
    full_fourier_data1, ImgRecon1, coef1 = nlgrappa_matlab(
        kspace_u, R, pe_loc, calib, acs_line_loc, num_block,
        num_column, times_comp)

    plt.figure()
    plt.imshow(np.abs(np.fft.fftshift(ImgRecon1, axes=(0, 1))))
    plt.show()
