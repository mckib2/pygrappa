'''Demonstrate how to grid non-Cartesian data.

Notes
-----
In general, you should probably be using a fast NUFFT implementation,
such as that in BART or NFFT [1]_ [2]_.  Unfortunately, most of these
implementations do not work "out-of-the-box" with Python (i.e., can't
pip install them) and/or they aren't cross-platform solutions (e.g.,
BART doesn't officially support Microsoft Windows).  In the case of
BART, I find the Python interface to be rather clumsy. Those that you
can install through pip, e.g., pynufft, would be great alternatives,
but since this package isn't meant to be a showcase of fast NDFTs,
we're just going to use some simple interpolation methods provided by
scipy.  We just want to get a taste of what non-Cartesian datasets
look like.

References
----------
.. [1] Uecker, Martin, et al. "Berkeley advanced reconstruction
       toolbox." Proc. Intl. Soc. Mag. Reson. Med. Vol. 23. 2015.
.. [2] Keiner, Jens, Stefan Kunis, and Daniel Potts. "Using
       NFFT 3---a software library for various nonequispaced fast
       Fourier transforms." ACM Transactions on Mathematical Software
       (TOMS) 36.4 (2009): 19.
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from phantominator import kspace_shepp_logan, radial

try:
    from bart import bart # pylint: disable=E0401
    FOUND_BART = True
except ModuleNotFoundError:
    FOUND_BART = False

if __name__ == '__main__':

    # Demo params
    sx, spokes, nc = 128, 128, 4
    of = 2 # oversampling factor for gridding
    method = 'linear' # interpolation strategy, see scipy.griddata()

    # Helper functions for sum-of-squares coil combine and ifft2
    sos = lambda x0: np.sqrt(np.sum(np.abs(x0)**2, axis=-1))
    ifft = lambda x0: np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(
        np.nan_to_num(x0), axes=(0, 1)), axes=(0, 1)), axes=(0, 1))

    # If you have BART installed, you could replicate this demo with
    # the following:
    # Make a radial trajectory, we'll have to mess with it later to
    # get it to look like pygrappa usually assumes it is
    if FOUND_BART:
        traj = bart(1, 'traj -r -x %d -y %d' % (sx, spokes))

        # Define a wrapper function for BART's nufft function,
        # assumes 2D
        bart_nufft = lambda x0: bart(
            1, 'nufft -i -t -d %d:%d:1' % (sx, sx),
            traj, x0.reshape((1, sx, spokes, nc))).squeeze()

        # Multicoil Shepp-Logan phantom kspace measurements
        kspace = bart(1, 'phantom -k -s %d -t' % nc, traj)

        # Make kx, ky, k look like they do for pygrappa
        bart_kx = traj[0, ...].real.flatten()
        bart_ky = traj[1, ...].real.flatten()
        bart_k = kspace.reshape((-1, nc))

        # Check it out
        plt.figure()
        plt.imshow(sos(bart_nufft(bart_k)))
        plt.title('BART NUFFT')
        plt.show(block=False)


    # The phantominator module also supports arbitrary kspace
    # sampling:
    kx, ky = radial(sx, spokes)
    k = kspace_shepp_logan(kx, ky)[..., None]

    # We will prefer a gridding approach to keep things simple:
    pad = int(sx*(of - 1)/2)
    xx, yy = np.meshgrid(
        np.linspace(np.min(kx), np.max(kx), sx*of),
        np.linspace(np.min(ky), np.max(ky), sx*of))
    grid_kspace = griddata((kx, ky), k, (xx, yy), method=method)
    grid_imspace = ifft(grid_kspace)[pad:-pad, pad:-pad, :]
    grid_imspace = np.flipud(grid_imspace) # match BART orientation

    # Take a gander
    plt.figure()
    plt.imshow(sos(grid_imspace))
    plt.title('scipy.griddata')
    plt.show()
