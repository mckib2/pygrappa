'''Naive NDFT.'''

import numpy as np
import matplotlib.pyplot as plt
from bart import bart # pylint: disable=E0401

from pygrappa import idft2d


if __name__ == '__main__':

    # Get some stuff
    sx, spokes, nc = 64, 64, 8
    traj = bart(1, 'traj -r -x %d -y %d' % (sx, spokes))
    nufft = lambda x0: bart(
        1, 'nufft -i -t -d %d:%d:1' % (sx, sx),
        traj, x0.reshape((1, sx, spokes, nc))).squeeze()
    kspace = bart(1, 'phantom -k -s %d -t' % nc, traj)

    kx = traj[0, ...].real.flatten()
    ky = traj[1, ...].real.flatten()
    k = kspace.reshape((-1, nc))

    res = np.zeros((sx, sx, nc), dtype='complex')
    for cc in range(nc):
        res[..., cc] = idft2d(kx, ky, k[..., cc], M=sx, N=sx)
        print(cc)
    print(res.shape)

    sos = lambda x0: np.sqrt(np.sum(np.abs(x0)**2, axis=-1))
    plt.subplot(1, 2, 1)
    plt.imshow(sos(nufft(kspace)))

    plt.subplot(1, 2, 2)
    plt.imshow(sos(res))
    plt.show()
