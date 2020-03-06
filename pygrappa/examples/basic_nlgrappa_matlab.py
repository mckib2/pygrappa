'''Show basic usage of NL-GRAPPA.'''

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

from pygrappa import nlgrappa_matlab

if __name__ == '__main__':

    # Load mat file
    data = loadmat(
        '/home/nicholas/Downloads/NLGRAPPA/rawdata_brain.mat')[
            'raw_data']
    im = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(
        data, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))
    sos = np.sqrt(np.sum(np.abs(im)**2, axis=-1))

    # plt.imshow(sos, cmap='gray')
    # plt.show()

    off = 0 # starting sampling location

    # The number of ACS lines
    R = 5
    nencode = 42

    # The convolution size
    num_block = 2
    num_column = 5#15 # make smaller to go quick during development

    # Obtain ACS data and undersampled data
    sx, sy, nc = data.shape[:]
    sx2 = int(sx/2)
    nencode2 = int(nencode/2)
    acs_line_loc = np.arange(sx2 - nencode2, sx2 + nencode2)
    calib = np.fft.fftshift(np.fft.fft2(
        im, axes=(0, 1)), axes=(0, 1))[acs_line_loc, ...].copy()

    # Obtain uniformly undersampled locations
    pe_loc = np.arange(off, sx-off, R)
    kspace_u = np.zeros((pe_loc.size, sy, nc), dtype=data.dtype)
    kspace_u = np.fft.fftshift(np.fft.fft2(
        im, axes=(0, 1)), axes=(0, 1))[pe_loc, ...].copy() # why do this?

    # Net reduction factor
    acq_idx = np.zeros(sx, dtype=bool)
    acq_idx[pe_loc] = True
    acq_idx[acs_line_loc] = True
    NetR = sx / np.sum(acq_idx)

    # Nonlinear GRAPPA Reconstruction
    times_comp = 3 # The number of times of the first-order terms
    full_fourier_data1, ImgRecon1, coef1 = nlgrappa_matlab(
        kspace_u, R, pe_loc, calib, acs_line_loc, num_block,
        num_column, times_comp)

    cmp = loadmat('/home/nicholas/Downloads/NLGRAPPA/data.mat')
    plt.imshow(np.abs(ImgRecon1 - cmp['ImgRecon1']))
    plt.show()
    #assert np.allclose(ImgRecon1, cmp['ImgRecon1'])

    plt.figure()
    plt.imshow(np.abs(np.fft.fftshift(cmp['ImgRecon1'])))
    plt.show(block=False)

    plt.figure()
    plt.imshow(np.abs(np.fft.fftshift(ImgRecon1, axes=(0, 1))))
    plt.show()
