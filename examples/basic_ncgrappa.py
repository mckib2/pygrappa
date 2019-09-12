'''Demo of Non-Cartesian GRAPPA.'''

import numpy as np
import matplotlib.pyplot as plt

from bart import bart # pylint: disable=E0401

from pygrappa import ncgrappa

if __name__ == '__main__':

    # Get phantom from BART since phantominator doesn't have
    # arbitrary sampling yet...
    sx, spokes, nc = 128, 64, 8
    traj = bart(1, 'traj -r -x %d -y %d -a 2' % (sx, spokes))
    kspace = bart(1, 'phantom -k -s %d -t' % nc, traj)
    print(kspace[..., 0])

    # Do inverse gridding with NUFFT so we can get fully sampled
    # cartesian ACS
    igrid = bart(
        1, 'nufft -i -t -d %d:%d:1' % (sx, sx),
        traj, kspace).squeeze()
    # plt.imshow(np.abs(igrid[..., 0]))
    # plt.show()
    ax = (0, 1)
    igrid = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(
        igrid, axes=ax), axes=ax), axes=ax)

    # 20x20 calibration region at the center
    ctr = int(sx/2)
    pad = 10
    calib = igrid[ctr-pad:ctr+pad, ctr-pad:ctr+pad, :].copy()

    # Get the trajectory and kspace samples
    kx = traj[0, ...].real.flatten()
    ky = traj[1, ...].real.flatten()
    kx /= np.max(np.abs(kx))
    ky /= np.max(np.abs(ky))
    k = kspace.reshape((-1, nc))

    # Reconstruct with Non-Cartesian GRAPPA
    ncgrappa(kx, ky, k, calib, kernel_size=.1, coil_axis=-1)
