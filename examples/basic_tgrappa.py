'''Example demonstrating how to use TGRAPPA.'''

import numpy as np
from phantominator import dynamic
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from pygrappa import tgrappa
from utils import gaussian_csm

if __name__ == '__main__':

    # Simulation parameters
    N = 128 # in-plane resolution: (N, N)
    nt = 40 # number of time frames
    ncoil = 4 # number of coils

    # Make a simple phantom
    ph = dynamic(N, nt)

    # Apply coil sensitivities
    csm = gaussian_csm(N, N, ncoil)
    ph = ph[:, :, None, :]*csm[..., None]

    # Throw into kspace
    ax = (0, 1)
    kspace = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(
        ph, axes=ax), axes=ax), axes=ax)

    # Undersample by factor 2 in both kx and ky, alternating with time
    kspace[0::2, 1::2, :, 0::2] = 0
    kspace[1::2, 0::2, :, 1::2] = 0

    # Reconstuct using TGRAPPA algorithm:
    #    Use 20x20 calibration region
    #    Kernel size: (4, 5)
    res = tgrappa(kspace, calib_size=(20, 20), kernel_size=(4, 5))

    # IFFT and stitch coil images together
    res = np.abs(np.sqrt(N**2)*np.fft.fftshift(np.fft.ifft2(
        np.fft.ifftshift(res, axes=ax), axes=ax), axes=ax))
    res0 = np.zeros((2*N, 2*N, nt))
    kk = 0
    for idx in np.ndindex((2, 2)):
        ii, jj = idx[:]
        res0[ii*N:(ii+1)*N, jj*N:(jj+1)*N, :] = res[..., kk, :]
        kk += 1

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
