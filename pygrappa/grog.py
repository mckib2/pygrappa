'''Python implmentation of the GROG algorithm.'''

from time import time

import numpy as np
from scipy.spatial import cKDTree  # pylint: disable=E0611
from scipy.optimize import least_squares

niter = 0
def grog(kx, ky, k, N, M):
    '''GRAPPA operator gridding.'''

    _ns, nc = k.shape[:]

    # We have samples at (kx, ky).  We want new samples on a
    # Cartesian grid at, say, (tx, ty).
    tx, ty = np.meshgrid(
        np.linspace(np.min(kx), np.max(kx), N),
        np.linspace(np.min(ky), np.max(ky), M))
    tx, ty = tx.flatten(), ty.flatten()
    outside = np.argwhere(np.sqrt(tx**2 + ty**2) > np.max(kx)).squeeze()
    inside = np.argwhere(np.sqrt(tx**2 + ty**2) <= np.max(kx)).squeeze()
    tx = np.delete(tx, outside)
    ty = np.delete(ty, outside)
    txy = np.concatenate((tx[:, None], ty[:, None]), axis=-1)

    # import matplotlib.pyplot as plt
    # plt.scatter(kx, ky)
    # plt.scatter(tx, ty)
    # plt.show()

    # We can find GRAPPA operators Gx, Gy from the data itself by
    # considering s(kx+dx, ky+dy) = Gx^(dx) Gy^(dy) s(kx, ky)
    # dx, dy = tx - kx, ty - ky
    dkx, dky = np.diff(kx), np.diff(ky)

    def _obj(x):
        global niter
        S = k[:-1, :]
        T = k[1:, :]
        Gxr = np.reshape(x[0*nc**2:1*nc**2], (nc, nc))
        Gxi = np.reshape(x[1*nc**2:2*nc**2], (nc, nc))
        Gyr = np.reshape(x[2*nc**2:3*nc**2], (nc, nc))
        Gyi = np.reshape(x[3*nc**2:4*nc**2], (nc, nc))
        Gx = Gxr + 1j*Gxi
        Gy = Gyr + 1j*Gyi
        T0 = [Gx**dx0 @ Gy**dy0 @ S[ii, :] for ii,
              (dx0, dy0) in enumerate(zip(dkx, dky))]
        res = np.linalg.norm(T - np.array(T0))
        niter += 1
        if not np.mod(niter, 100):
            print(res)
        return res

    G0 = np.ones(4*nc**2)*.1
    print('Starting least_squares...')
    t0 = time()
    res = least_squares(_obj, G0, '3-point')
    print(res)
    print('Found Gx, Gy in %g seconds' % (time() - t0))
    Gxr = np.reshape(res['x'][0*nc**2:1*nc**2], (nc, nc))
    Gxi = np.reshape(res['x'][1*nc**2:2*nc**2], (nc, nc))
    Gyr = np.reshape(res['x'][2*nc**2:3*nc**2], (nc, nc))
    Gyi = np.reshape(res['x'][3*nc**2:4*nc**2], (nc, nc))
    Gx = Gxr + 1j*Gxi
    Gy = Gyr + 1j*Gyi

    # # Testing:
    # Gx = np.zeros((nc, nc), dtype=k.dtype)
    # Gy = Gx.copy()

    # Grid
    kxy = np.concatenate((kx[:, None], ky[:, None]), axis=-1)
    kdtree = cKDTree(kxy)
    _, idx = kdtree.query(txy, k=1)
    # print(idx)

    res = [Gx**(tx0 - kx[idx[ii]]) @ Gy**(ty0 - ky[idx[ii]]) @ k[idx[ii], :] for ii, (tx0, ty0) in enumerate(zip(tx, ty))]
    res0 = np.zeros((N*M, nc), dtype=k.dtype)
    res0[inside, :] = res
    return res0.reshape((N, M, nc))
    # return np.array(res)#.reshape((N, M, nc))

    # We then build a dictionary of desired Gx^(dx), Gy^(dy).  We can
    # further factor Gx^(dx), Gy^(dy) into Gi^(di0 + di1 + ...) to
    # further reduce computations desired, but this is an NP-hard
    # problem (coin change problem)

    # For radial we could use the log method described in the paper,
    # but we could also try to numerically solve the system of
    # nonlinear equations?
