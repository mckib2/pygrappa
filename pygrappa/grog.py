'''Python implmentation of the GROG algorithm.'''

from time import time

import numpy as np
from scipy.spatial import cKDTree  # pylint: disable=E0611
from scipy.linalg import fractional_matrix_power as fmp
from tqdm import tqdm

from pygrappa.grog_powers import grog_powers_double, grog_powers_float # pylint: disable=E0611
from pygrappa.grog_gridding import (
    grog_gridding_double, grog_gridding_float) # pylint: disable=E0611

def _make_key(key, precision):
    '''Dictionary keys.'''
    return np.around(key, decimals=int(precision))

def grog(
        kx, ky, k, N, M, Gx, Gy, precision=2, radius=.75, Dx=None,
        Dy=None, coil_axis=-1, ret_image=False, ret_dicts=False,
        use_primefac=False, remove_os=True, inverse=False):
    '''GRAPPA operator gridding.

    Parameters
    ----------
    kx, ky : array_like
        k-space coordinates (kx, ky) of measured data k.  kx, ky
        should each be a 1D array.  Must both be either float or
        double.
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
    remove_os : bool, optional
        Remove oversampling factor.
    inverse : bool, optional
        Do the inverse gridding operation, i.e., Cartesian points to
        (kx, ky).

    Returns
    -------
    res : array_like
        Cartesian gridded k-space (or image).
    Dx, Dy : dict, optional
        Fractional matrix power dictionary for both Gx and Gy.

    Raises
    ------
    AssertionError
        When (kx, ky) have different types.
    AssertionError
        When (kx, ky) and k do not have matching types, i.e.,
        if (kx, ky) are float32, k must be complex64.

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

    # Make sure types are consistent before calling grog funcs
    assert kx.dtype == ky.dtype, (
        '(kx, ky) must both be either double or float!')
    assert (k.dtype == np.complex64 if
            kx.dtype == np.float32 else k.dtype == np.complex128), (
                '(kx, ky) and k must have matching types!')

    # Coils to the back
    k = np.moveaxis(k, coil_axis, -1)
    _ns, nc = k.shape[:]

    if not inverse:
        # We have samples at (kx, ky).  We want new samples on a
        # Cartesian grid at, say, (tx, ty). Let's also oversample:
        N, M = 2*N, 2*M

    # Create the target grid (or source grid for inverse gridding)
    tx, ty = np.meshgrid(
        np.linspace(np.min(kx), np.max(kx), N, dtype=kx.dtype),
        np.linspace(np.min(ky), np.max(ky), M, dtype=kx.dtype))
    tx, ty = tx.flatten(), ty.flatten()

    # We only want to do work inside the region of support:
    # estimate as a circle for now, works well with radial
    outside = np.argwhere(
        np.sqrt(tx**2 + ty**2) > np.max(kx)).squeeze()
    inside = np.argwhere(
        np.sqrt(tx**2 + ty**2) <= np.max(kx)).squeeze()
    tx = np.delete(tx, outside)
    ty = np.delete(ty, outside)

    if inverse:
        # We want to fill all non-cartesian locations, so the region
        # of support is the whole thing (all indices)
        k = np.delete(k, outside, axis=0)
        inside = np.arange(kx.size, dtype=int)

    # Swap coordinates if doing inverse (cartesian to radial)
    if inverse:
        kx, tx = tx, kx
        ky, ty = ty, ky
    kxy = np.concatenate((kx[:, None], ky[:, None]), axis=-1)
    txy = np.concatenate((tx[:, None], ty[:, None]), axis=-1)

    # Find all targets within radius of source points
    kdtree = cKDTree(kxy)
    idx = kdtree.query_ball_point(txy, r=radius)

    # The result will be shaped different if we are doing inverse
    # gridding:
    if inverse:
        res = np.zeros((tx.size, nc), dtype=k.dtype)
    else:
        res = np.zeros((N*M, nc), dtype=k.dtype)

    t0 = time()
    # Handle both single and double floating point calculations,
    # have to do it in separate functions because Cython...
    if tx.dtype == np.float32:
        key_x, key_y = grog_powers_float(
            tx, ty, kx, ky, idx, precision)
    else:
        key_x, key_y = grog_powers_double(
            tx, ty, kx, ky, idx, precision)
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
        frac_mats_x[lscale_fac] = np.linalg.pinv(
            fmp(Gx, lscale_fac)).astype(k.dtype)
        frac_mats_y[lscale_fac] = np.linalg.pinv(
            fmp(Gy, lscale_fac)).astype(k.dtype)

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
                np.linalg.matrix_power(
                    frac_mats_x[lpx0], n0).astype(k.dtype) for
                lpx0, n0 in zip(lpx, nx)])
            Dy[np.abs(keyy0)] = np.linalg.multi_dot([
                np.linalg.matrix_power(
                    frac_mats_y[lpy0], n0).astype(k.dtype) for
                lpy0, n0 in zip(lpy, ny)])

            if np.sign(keyx0) < 0:
                Dx[keyx0] = np.linalg.pinv(
                    Dx[np.abs(keyx0)]).astype(k.dtype)
            if np.sign(keyy0) < 0:
                Dy[keyy0] = np.linalg.pinv(
                    Dy[np.abs(keyy0)]).astype(k.dtype)

    else:
        for key0 in tqdm(key_x, leave=False, desc='Dx'):
            Dx[np.abs(key0)] = fmp(Gx, np.abs(key0)).astype(k.dtype)
            if np.sign(key0) < 0:
                Dx[key0] = np.linalg.pinv(
                    Dx[np.abs(key0)]).astype(k.dtype)
        for key0 in tqdm(key_y, leave=False, desc='Dy'):
            Dy[np.abs(key0)] = fmp(Gy, np.abs(key0)).astype(k.dtype)
            if np.sign(key0) < 0:
                Dy[key0] = np.linalg.pinv(
                    Dy[np.abs(key0)]).astype(k.dtype)
    print(
        'Took %g seconds to precompute fractional matrix powers' % (
            time() - t0))

    # res is modified inplace
    if res.dtype == np.complex64:
        grog_gridding_float(
            tx, ty, kx, ky, k, idx, res, inside, Dx, Dy, precision)
    else:
        grog_gridding_double(
            tx, ty, kx, ky, k, idx, res, inside, Dx, Dy, precision)

    if inverse:
        retVal = res
    else:
        # Remove the oversampling factor and return in kspace or
        # imspace
        ax = (0, 1)
        im = None
        if remove_os:
            N4, M4 = int(N/4), int(M/4)
            im = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(
                res.reshape((N, M, nc), order='F'),
                axes=ax), axes=ax), axes=ax)[N4:-N4, M4:-M4, :]
            res = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(
                im, axes=ax), axes=ax), axes=ax)

        if ret_image:
            if im is None:
                retVal = np.fft.fftshift(np.fft.ifft2(
                    np.fft.ifftshift(res.reshape(
                        (N, M, nc), order='F'), axes=ax),
                    axes=ax), axes=ax)
            else:
                retVal = im
        else:
            retVal = res

    # If the user asked for the precomputed dictionaries back, add
    # them to the tuple of returned values
    if ret_dicts:
        retVal = (retVal, Dx, Dy)

    return retVal
