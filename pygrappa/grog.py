'''Python implmentation of the GROG algorithm.'''

import numpy as np
from scipy.spatial import cKDTree  # pylint: disable=E0611
from scipy.linalg import fractional_matrix_power as fmp
from tqdm import tqdm

def _make_key(key):
    '''Dictionary keys.'''
    return round(key, 1)

def grog(kx, ky, k, N, M, Gx, Gy):
    '''GRAPPA operator gridding.'''

    # Coils to the back
    _ns, nc = k.shape[:]

    # We have samples at (kx, ky).  We want new samples on a
    # Cartesian grid at, say, (tx, ty).
    tx, ty = np.meshgrid(
        np.linspace(np.min(kx), np.max(kx), N),
        np.linspace(np.min(ky), np.max(ky), M))
    tx, ty = tx.flatten(), ty.flatten()
    outside = np.argwhere(
        np.sqrt(tx**2 + ty**2) > np.max(kx)).squeeze()
    inside = np.argwhere(
        np.sqrt(tx**2 + ty**2) <= np.max(kx)).squeeze()
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
    # dkx, dky = np.diff(kx), np.diff(ky)

    # Grid
    kxy = np.concatenate((kx[:, None], ky[:, None]), axis=-1)
    kdtree = cKDTree(kxy)
    _, idx = kdtree.query(txy, k=1)

    res = []
    Dx, Dy = {}, {}
    for ii, (tx0, ty0) in tqdm(
            enumerate(zip(tx, ty)), total=idx.size, leave=False):
        dx = tx0 - kx[idx[ii]]
        key = _make_key(dx)
        if key not in Dx:
            Dx[key] = fmp(Gx, np.abs(key))
            if np.sign(key) < 0:
                Dx[key] = np.linalg.matrix_power(Dx[key], -1)
        Gxf = Dx[key]

        dy = ty0 - ky[idx[ii]]
        key = _make_key(dy)
        if key not in Dy:
            Dy[key] = fmp(Gy, np.abs(key))
            if np.sign(key) < 0:
                Dy[key] = np.linalg.matrix_power(Dy[key], -1)
        Gyf = Dy[key]

        res.append(Gxf @ Gyf @ k[idx[ii], :])

    res0 = np.zeros((N*M, nc), dtype=k.dtype)
    res0[inside, :] = res
    return res0.reshape((N, M, nc))
