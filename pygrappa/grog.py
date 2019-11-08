'''Python implmentation of the GROG algorithm.'''

from time import time

import numpy as np
from scipy.spatial import cKDTree  # pylint: disable=E0611
from scipy.linalg import fractional_matrix_power as fmp
from tqdm import tqdm

from pygrappa.grog_powers import grog_powers
from pygrappa.grog_gridding import grog_gridding

def _make_key(key, precision):
    '''Dictionary keys.'''
    return np.around(key, decimals=int(precision))

def grog(
        kx, ky, k, N, M, Gx, Gy, precision=2, radius=.75, Dx=None,
        Dy=None, coil_axis=-1, ret_image=False, ret_dicts=False,
        use_primefac=False):
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
    Dx, Dy : dict, optional
        Dictionaries of precomputed fractional matrix powers.
    coil_axis : int, optional
        Axis holding coil data.
    ret_image : bool, optional
        Return image space result instead of k-space.
    ret_dicts : bool, optional
        Return dictionaries of fractional matrix powers.
    use_primefac : bool, optional
        Use prime factorization to speed-up fractional matrix
        power precomputations.

    Returns
    -------
    res : array_like
        Cartesian gridded k-space (or image).
    Dx, Dy : dict, optional
        Fractional matrix power dictionary for both Gx and Gy.

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
    res = np.zeros((N*M, nc), dtype=k.dtype)

    t0 = time()
    key_x, key_y = grog_powers(tx, ty, kx, ky, idx, precision)
    print('Took %g seconds to find required powers' % (time() - t0))

    # If we have provided dictionaries, whitle down the work to only
    # those powers not already computed
    t0 = time()
    if Dx:
        key_x = key_x - set(Dx.keys())
    else:
        Dx = {}
    if Dy:
        key_y = key_y - set(Dy.keys())
    else:
        Dy = {}

    # Precompute deficient matrix powers
    if use_primefac:
        # Precompute matrix powers using prime factorization
        from primefac import factorint # pylint: disable=E0401
        scale_fac = 10**precision

        # Start a dictionary of fractional matrix powers
        frac_mats_x = {}
        frac_mats_y = {}

        # First thing we need is the scale factor, note that we will
        # assume the inverse!
        lscale_fac = np.log(scale_fac)
        frac_mats_x[lscale_fac] = np.linalg.pinv(fmp(Gx, lscale_fac))
        frac_mats_y[lscale_fac] = np.linalg.pinv(fmp(Gy, lscale_fac))

        for keyx0, keyy0 in tqdm(
                zip(key_x, key_y), total=len(key_x), leave=False,
                desc='Dxy'):

            dx0 = np.exp(np.abs(keyx0))*scale_fac
            dy0 = np.exp(np.abs(keyy0))*scale_fac
            rx = factorint(int(dx0))
            ry = factorint(int(dy0))

            # Component fractional powers are log of prime factors;
            # add in the scale_fac term here so we get it during the
            # multi_dot later.  We explicitly cast to integer because
            # sometimes we run into an MPZ object that doesn't play
            # nice with numpy
            lpx = np.log(np.array(
                [int(r) for r in rx.keys()] + [scale_fac])).squeeze()
            lpy = np.log(np.array(
                [int(r) for r in ry.keys()] + [scale_fac])).squeeze()
            lpx_unique = np.unique(lpx)
            lpy_unique = np.unique(lpy)

            # Compute new fractional matrix powers we haven't seen
            for lpxu in lpx_unique:
                if lpxu not in frac_mats_x:
                    frac_mats_x[lpxu] = fmp(Gx, lpxu)
            for lpyu in lpy_unique:
                if lpyu not in frac_mats_y:
                    frac_mats_y[lpyu] = fmp(Gy, lpyu)

            # Now compose all the matrices together for this point
            nx = list(rx.values()) + [1] # +1 to account for scale_fac
            ny = list(ry.values()) + [1]
            Dx[np.abs(keyx0)] = np.linalg.multi_dot([
                np.linalg.matrix_power(frac_mats_x[lpx0], n0) for
                lpx0, n0 in zip(lpx, nx)])
            Dy[np.abs(keyy0)] = np.linalg.multi_dot([
                np.linalg.matrix_power(frac_mats_y[lpy0], n0) for
                lpy0, n0 in zip(lpy, ny)])

            if np.sign(keyx0) < 0:
                Dx[keyx0] = np.linalg.pinv(Dx[np.abs(keyx0)])
            if np.sign(keyy0) < 0:
                Dy[keyy0] = np.linalg.pinv(Dy[np.abs(keyy0)])

    else:
        for key0 in tqdm(key_x, leave=False, desc='Dx'):
            Dx[np.abs(key0)] = fmp(Gx, np.abs(key0))
            if np.sign(key0) < 0:
                Dx[key0] = np.linalg.pinv(Dx[np.abs(key0)])
        for key0 in tqdm(key_y, leave=False, desc='Dy'):
            Dy[np.abs(key0)] = fmp(Gy, np.abs(key0))
            if np.sign(key0) < 0:
                Dy[key0] = np.linalg.pinv(Dy[np.abs(key0)])
    print(
        'Took %g seconds to precompute fractional matrix powers' % (
            time() - t0))

    # res is modified inplace
    grog_gridding(
        tx, ty, kx, ky, k, idx, res, inside.astype(np.uint32),
        Dx, Dy, precision)

    # Remove the oversampling factor and return in kspace
    N4, M4 = int(N/4), int(M/4)
    ax = (0, 1)
    im = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(
        res.reshape((N, M, nc), order='F'),
        axes=ax), axes=ax), axes=ax)[N4:-N4, M4:-M4, :]
    if ret_image:
        retVal = im
    else:
        retVal = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(
            im, axes=ax), axes=ax), axes=ax)

    # If the user asked for the precomputed dictionaries back, add
    # them to the tuple of returned values
    if ret_dicts:
        retVal = (retVal, Dx, Dy)

    return retVal
