'''Basic demo of Split-Slice-GRAPPA.'''

import numpy as np
from phantominator import shepp_logan
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from pygrappa import splitslicegrappa
from utils import gaussian_csm

if __name__ == '__main__':

    # Get slices of 3D Shepp-Logan phantom
    N = 128
    ns = 2
    ph = shepp_logan((N, N, ns), zlims=(-.3, 0))

    # Apply some coil sensitivities
    ncoil = 8
    csm = gaussian_csm(N, N, ncoil)
    ph = ph[..., None, :]*csm[..., None]

    # Shift one slice FOV/2 (SMS-CAIPI)
    ph[..., -1] = np.fft.fftshift(ph[..., -1], axes=0)

    # Put into kspace
    ax = (0, 1)
    kspace = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(
        ph, axes=ax), axes=ax), axes=ax)

    # Calibration data is individual slices
    calib = kspace.copy()

    # Simulate SMS by simply adding slices together
    kspace_sms = np.sum(kspace, axis=-1)

    # Make identical time frames
    nt = 5
    kspace_sms = np.tile(kspace_sms[..., None], (1, 1, 1, nt))

    # Separate the slices using Split-Slice-GRAPPA
    res = splitslicegrappa(
        kspace_sms, calib, kernel_size=(5, 5), prior='kspace')

    # IFFT and stitch slices together
    res = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(
        res, axes=ax), axes=ax), axes=ax)
    res0 = np.zeros((ns*N, N, nt))
    for ii in range(ns):
        res0[ii*N:(ii+1)*N, ...] = np.sqrt(
            np.sum(np.abs(res[..., ii])**2, axis=2))

    # Some code to look at the animation
    fig = plt.figure()
    ax = plt.imshow(np.abs(res0[..., 0]), cmap='gray')

    def init():
        '''Initialize ax data.'''
        ax.set_array(np.abs(res0[..., 0]))
        return(ax,)

    def animate(frame):
        '''Update frame.'''
        ax.set_array(np.abs(res0[..., frame]))
        return(ax,)

    anim = FuncAnimation(
        fig, animate, init_func=init, frames=nt,
        interval=40, blit=True)
    plt.show()
