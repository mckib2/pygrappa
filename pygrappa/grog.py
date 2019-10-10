'''Python implmentation of the GROG algorithm.'''

import numpy as np
from scipy.spatial import cKDTree  # pylint: disable=E0611
from scipy.linalg import fractional_matrix_power as fmp
from tqdm import tqdm

def _make_key(key, precision):
    '''Dictionary keys.'''
    return np.around(key, decimals=int(precision))

def grog(
        kx, ky, k, N, M, Gx, Gy, precision=2, radius=.75,
        coil_axis=-1, ret_image=False, flip_flop=False):
    '''GRAPPA operator gridding.

    Parameters
    ----------
    kx, ky : array_like
        k-space coordinates (kx, ky) of measured data k.  kx, ky
        should each be a 1D array.
    k : array_like
        Measured  k-space data at points (kx, ky).
    N, M : int
        Desired resolution of Cartesian grid.
    Gx, Gy : array_like
        Unit GRAPPA operators.
    precision : int, optional
        Number of decimal places to round fractional matrix powers to.
    radius : float, optional
        Radius of ball in k-space to from Cartesian targets from
        which to select source points.
    coil_axis : int, optional
        Axis holding coil data.
    ret_image : bool, optional
        Return image space result instead of k-space.
    flip_flop : bool, optional
        Randomly shift the order of Gx and Gy application to achieve
        commutativity on average.

    Returns
    -------
    res : array_like
        Cartesian gridded k-space (or image).

    Notes
    -----
    Implements the GROG algorithm as described in [1]_.

    References
    ----------
    .. [1] Seiberlich, Nicole, et al. "Self‐calibrating GRAPPA
           operator gridding for radial and spiral trajectories."
           Magnetic Resonance in Medicine: An Official Journal of the
           International Society for Magnetic Resonance in Medicine
           59.4 (2008): 930-935.
    '''

    # Coils to the back
    k = np.moveaxis(k, coil_axis, -1)
    _ns, nc = k.shape[:]

    # We have samples at (kx, ky).  We want new samples on a
    # Cartesian grid at, say, (tx, ty). Let's also oversample by a
    # factor of 2:
    N, M = 2*N, 2*M
    tx, ty = np.meshgrid(
        np.linspace(np.min(kx), np.max(kx), N),
        np.linspace(np.min(ky), np.max(ky), M))
    tx, ty = tx.flatten(), ty.flatten()

    # We only want to do work inside the region of support: estimate
    # as a circle for now, works well with radial
    outside = np.argwhere(
        np.sqrt(tx**2 + ty**2) > np.max(kx)).squeeze()
    inside = np.argwhere(
        np.sqrt(tx**2 + ty**2) <= np.max(kx)).squeeze()
    tx = np.delete(tx, outside)
    ty = np.delete(ty, outside)
    txy = np.concatenate((tx[:, None], ty[:, None]), axis=-1)

    # Grid
    kxy = np.concatenate((kx[:, None], ky[:, None]), axis=-1)
    kdtree = cKDTree(kxy)
    idx = kdtree.query_ball_point(txy, r=radius)
    cnts = [len(idx0) if idx0 else 1 for idx0 in idx]
    res = np.zeros((N*M, nc), dtype=k.dtype)
    Dx, Dy = {}, {}
    for ii, (cnts0, tx0, ty0) in tqdm(
            enumerate(zip(cnts, tx, ty)),
            total=idx.size, leave=False):

        # Each Cartesian target may have many source points.
        # Accumulate all of these and then average:
        for idx0 in idx[ii]:
            # Construct dictionary for Gx
            dx = tx0 - kx[idx0]
            key = _make_key(dx, precision)
            if key not in Dx:
                Dx[np.abs(key)] = fmp(Gx, np.abs(key))
                if np.sign(key) < 0:
                    Dx[key] = np.linalg.pinv(Dx[np.abs(key)])
            Gxf = Dx[key]

            # Construct dictionary for Gy
            dy = ty0 - ky[idx0]
            key = _make_key(dy, precision)
            if key not in Dy:
                Dy[np.abs(key)] = fmp(Gy, np.abs(key))
                if np.sign(key) < 0:
                    Dy[key] = np.linalg.pinv(Dy[np.abs(key)])
            Gyf = Dy[key]

            # Expect to do an equal number of Gxf @ Gyf and Gyf @ Gxf,
            # that is, expect commutativity
            if flip_flop and np.random.rand(1) > .5:
                Gxf, Gyf = Gyf, Gxf

            # Start the averaging (accumulation step)
            res[inside[ii], :] += Gxf @ Gyf @ k[idx0, :]

        # Finish the averaging (dividing step)
        res[inside[ii], :] /= cnts0

    # Remove the oversampling factor and return in kspace
    N4, M4 = int(N/4), int(M/4)
    ax = (0, 1)
    im = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(
        res.reshape((N, M, nc), order='F'),
        axes=ax), axes=ax), axes=ax)[N4:-N4, M4:-M4, :]
    if ret_image:
        return im
    return np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(
        im, axes=ax), axes=ax), axes=ax)
